import torch
import torch.nn as nn
import copy
from torchvision.utils import make_grid, save_image
import os


class EMA:
    """
    Exponential Moving Average with Dynamic Decay.
    Optimized for memory efficiency (in-place operations) and includes buffer synchronization.
    """

    def __init__(self, model, target_decay=0.9999, warmup_steps=100):
        self.model = model
        self.target_decay = target_decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.shadow = {}
        self.original_weights = {}
        self._register_shadow()

    @torch.no_grad()
    def _register_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        for name, buffer in self.model.named_buffers():
            self.shadow[name] = buffer.data.clone()

    @torch.no_grad()
    def update(self):
        self.step += 1
        decay = min(
            self.target_decay, (1.0 + self.step) / (self.warmup_steps + self.step)
        )

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # In-place linear interpolation prevents memory fragmentation
                # shadow = shadow + (1 - decay) * (param - shadow)
                self.shadow[name].lerp_(param.data, 1.0 - decay)

        for name, buffer in self.model.named_buffers():
            if buffer.is_floating_point():
                self.shadow[name].lerp_(buffer.data, 1.0 - decay)
            else:
                self.shadow[name].copy_(buffer.data)

    @torch.no_grad()
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.original_weights[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

        for name, buffer in self.model.named_buffers():
            self.original_weights[name] = buffer.data.clone()
            buffer.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original_weights[name])

        for name, buffer in self.model.named_buffers():
            buffer.data.copy_(self.original_weights[name])


def save_result_grids(images, path, nrow=8):
    """
    Saves a grid of 32 images as required: 8 images per row, 4 rows.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, path)


def save_individual_images(images, folder_path):
    """
    Saves individual PNG files in the exact structure required for submission.
    Structure: images/test/<id>.png or images/new_test/<id>.png
    """
    os.makedirs(folder_path, exist_ok=True)
    for i, img in enumerate(images):
        # Image IDs in i-CLEVR are typically 1-indexed for final submission
        save_path = os.path.join(folder_path, f"{i+1}.png")
        # Ensure images are denormalized from [-1, 1] to [0, 1]
        save_image((img + 1.0) / 2.0, save_path)


def get_denoising_progress_grid(denoising_steps, path):
    """
    Saves the denoising process as a horizontal grid.
    Expects a list of tensors from noise to clean.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Stack steps along a new dimension and create grid
    grid = make_grid(
        torch.stack(denoising_steps),
        nrow=len(denoising_steps),
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(grid, path)
