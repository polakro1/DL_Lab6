import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(a, t, x_shape):
    """Extracts values from a 1D tensor for a batch of indices."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'.
    Prevents abrupt noise injection at early stages.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999)


class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register buffers to avoid moving them to device manually
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """Forward pass: Add noise to the image."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, context, p_uncond=0.1):
        """Computes MSE loss for training with unconditional dropout for CFG."""
        b, c, h, w = x_start.shape
        device = x_start.device

        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        # CFG conditional dropout
        if p_uncond > 0:
            mask = torch.rand(b, 1, device=device) < p_uncond
            context = torch.where(mask, torch.zeros_like(context), context)

        predicted_noise = self.model(x_t, t, context)
        return torch.nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, shape, context, ddim_steps=50, cfg_scale=3.0, eta=0.0):
        """
        Fast DDIM sampling with Classifier-Free Guidance.
        """
        device = context.device
        b = shape[0]
        x = torch.randn(shape, device=device)

        step_size = self.timesteps // ddim_steps
        time_steps = torch.arange(0, self.timesteps, step_size, device=device).flip(0)

        null_context = torch.zeros_like(context)

        # Disable gradients strictly to avoid memory leaks during inference
        with torch.no_grad():
            for i, t in enumerate(tqdm(time_steps, desc="DDIM Sampling")):
                t_b = torch.full((b,), t, device=device, dtype=torch.long)

                t_prev = t - step_size
                t_prev_b = torch.full((b,), t_prev, device=device, dtype=torch.long)

                # CFG: Batch conditioning and unconditional passes together
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t_b, t_b], dim=0)
                c_in = torch.cat([context, null_context], dim=0)

                pred_noise_both = self.model(x_in, t_in, c_in)
                pred_noise_cond, pred_noise_uncond = pred_noise_both.chunk(2)

                # Extrapolate
                pred_noise = pred_noise_uncond + cfg_scale * (
                    pred_noise_cond - pred_noise_uncond
                )

                # DDIM Math
                alpha = extract(self.alphas_cumprod, t_b, x.shape)
                alpha_prev = (
                    extract(self.alphas_cumprod, t_prev_b, x.shape)
                    if t_prev >= 0
                    else torch.ones_like(alpha)
                )

                sigma = eta * torch.sqrt(
                    (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
                )

                # Predict x_0 and clamp for stability (dynamic thresholding)
                pred_x0 = (x - torch.sqrt(1 - alpha) * pred_noise) / torch.sqrt(alpha)
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

                dir_xt_radicand = 1 - alpha_prev - sigma**2
                dir_xt = torch.sqrt(torch.clamp(dir_xt_radicand, min=1e-7)) * pred_noise
                noise = torch.randn_like(x) if t > 0 and eta > 0 else 0.0

                x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma * noise

        return x
