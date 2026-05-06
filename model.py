import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        # AdaGN condition projection
        self.emb_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(x)

        # Adaptive Group Normalization (AdaGN) modulation
        emb_out = self.emb_proj(t_emb)
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = h * (1 + scale[..., None, None]) + shift[..., None, None]

        h = self.conv2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = rearrange(
            qkv,
            "b (qkv heads c) h w -> qkv b heads (h w) c",
            qkv=3,
            heads=self.num_heads,
        )

        scale = 1.0 / math.sqrt(C // self.num_heads)
        attn = torch.einsum("b h i c, b h j c -> b h i j", q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("b h i j, b h j c -> b h i c", attn, v)
        out = rearrange(out, "b heads (h w) c -> b (heads c) h w", h=H, w=W)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_classes=24):
        super().__init__()
        time_emb_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.label_mlp = nn.Sequential(
            nn.Linear(num_classes, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsample steps: resolutions 64 -> 32 -> 16 -> 8
        self.down1 = ResnetBlock(base_channels, base_channels, time_emb_dim)
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), ResnetBlock(base_channels, base_channels * 2, time_emb_dim)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ResnetBlock(base_channels * 2, base_channels * 2, time_emb_dim),
        )
        self.attn3 = AttentionBlock(base_channels * 2)
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            ResnetBlock(base_channels * 2, base_channels * 4, time_emb_dim),
        )
        self.attn4 = AttentionBlock(base_channels * 4)

        self.mid1 = ResnetBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid_attn = AttentionBlock(base_channels * 4)
        self.mid2 = ResnetBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Upsample steps
        self.up1 = ResnetBlock(base_channels * 6, base_channels * 2, time_emb_dim)
        self.attn_up1 = AttentionBlock(base_channels * 2)
        self.up2 = ResnetBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.attn_up2 = AttentionBlock(base_channels * 2)
        self.up3 = ResnetBlock(base_channels * 3, base_channels, time_emb_dim)
        self.up4 = ResnetBlock(base_channels * 2, base_channels, time_emb_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time, context):
        t_emb = self.time_mlp(time)
        c_emb = self.label_mlp(context)
        emb = t_emb + c_emb

        x0 = self.init_conv(x)

        d1 = self.down1(x0, emb)
        d2 = self.down2[1](self.down2[0](d1), emb)

        d3 = self.down3[1](self.down3[0](d2), emb)
        d3 = self.attn3(d3)

        d4 = self.down4[1](self.down4[0](d3), emb)
        d4 = self.attn4(d4)

        m = self.mid1(d4, emb)
        m = self.mid_attn(m)
        m = self.mid2(m, emb)

        u1 = self.up1(
            torch.cat([F.interpolate(m, scale_factor=2, mode="nearest"), d3], dim=1),
            emb,
        )
        u1 = self.attn_up1(u1)

        u2 = self.up2(
            torch.cat([F.interpolate(u1, scale_factor=2, mode="nearest"), d2], dim=1),
            emb,
        )
        u2 = self.attn_up2(u2)

        u3 = self.up3(
            torch.cat([F.interpolate(u2, scale_factor=2, mode="nearest"), d1], dim=1),
            emb,
        )

        u4 = self.up4(torch.cat([u3, x0], dim=1), emb)

        return self.out_conv(u4)
