import torch
import torch.nn as nn
import copy
from torchvision.utils import make_grid, save_image
import os


class EMA:
    """
    Exponential Moving Average of model weights.
    Crucial for stabilizing diffusion models and reaching high accuracy.
    """

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = self._make_shadow()

    def _make_shadow(self):
        shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                shadow[name] = param.data.clone()
        return shadow

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Copies shadow weights to the actual model for evaluation."""
        self.original_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.original_weights[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore_weights(self):
        """Restores training weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original_weights[name])


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
