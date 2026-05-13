import torch
import json
import os
import argparse
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from model import UNet
from diffusion import DDPM
from dataset import ICLEVRDataset
from utils import save_result_grids, save_individual_images
from evaluator import evaluation_model

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.models._utils"
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="best_model.pth", help="Path to trained model"
    )
    parser.add_argument(
        "--output_dir", type=str, default="images", help="Base directory for results"
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=50, help="Number of DDIM steps"
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=3.0, help="Classifier-Free Guidance scale"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )

    default_device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=str(default_device),
        help="Device to run inference on",
    )

    return parser.parse_args()


@torch.inference_mode()
def generate_denoising_process(
    diffusion, obj2idx, device, cfg_scale=3.0, ddim_steps=50
):
    labels = ["red sphere", "cyan cylinder", "cyan cube"]
    context = torch.zeros(1, 24, device=device)
    for label in labels:
        context[0, obj2idx[label]] = 1.0

    null_context = torch.zeros_like(context)

    shape = (1, 3, 64, 64)
    x = torch.randn(shape, device=device)
    step_size = diffusion.timesteps // ddim_steps
    time_steps = torch.arange(0, diffusion.timesteps, step_size, device=device).flip(0)

    all_steps = []
    curr_x = x

    for t in tqdm(time_steps, desc="Denoising Process"):
        # Save current noisy state (normalized for visual)
        all_steps.append((curr_x.cpu().squeeze(0) + 1.0) / 2.0)

        t_b = torch.full((1,), t, device=device, dtype=torch.long)

        # Standard batched CFG
        x_in = torch.cat([curr_x, curr_x], dim=0)
        t_in = torch.cat([t_b, t_b], dim=0)
        c_in = torch.cat([context, torch.zeros_like(context)], dim=0)

        noise_pred_both = diffusion.model(x_in, t_in, c_in)
        noise_cond, noise_uncond = noise_pred_both.chunk(2)
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

        # Exact DDIM math
        alpha = diffusion.alphas_cumprod[t]
        t_prev = t - step_size
        alpha_prev = (
            diffusion.alphas_cumprod[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=device)
        )

        pred_x0 = (curr_x - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred
        curr_x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt

    all_steps.append((curr_x.cpu().squeeze(0) + 1.0) / 2.0)
    indices = torch.linspace(0, len(all_steps) - 1, 10).long()
    progress = [all_steps[idx] for idx in indices]

    grid = make_grid(torch.stack(progress), nrow=10)
    save_path = "results/denoising_process.png"
    os.makedirs("results", exist_ok=True)
    save_image(grid, save_path)
    print(f"Denoising process saved to {save_path}")


def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    with open("data/objects.json", "r") as f:
        obj2idx = json.load(f)

    model = UNet(base_channels=64).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = DDPM(model).to(device)
    evaluator = evaluation_model()

    for json_name in ["test.json", "new_test.json"]:
        print(f"\nGenerating images for {json_name}...")
        test_dataset = ICLEVRDataset(
            data_dir="data/iclevr",
            json_file=f"data/{json_name}",
            objects_file="data/objects.json",
            mode="test",
        )

        test_labels = torch.stack(
            [test_dataset[i] for i in range(len(test_dataset))]
        ).to(device)

        all_samples = []

        with torch.inference_mode():
            for i in range(0, len(test_labels), args.batch_size):
                batch_labels = test_labels[i : i + args.batch_size]

                print(
                    f"Processing batch {i//args.batch_size + 1}/{(len(test_labels)-1)//args.batch_size + 1}"
                )

                batch_samples = diffusion.ddim_sample(
                    shape=(len(batch_labels), 3, 64, 64),
                    context=batch_labels,
                    ddim_steps=args.ddim_steps,
                    cfg_scale=args.cfg_scale,
                )
                all_samples.append(batch_samples)

                # Clear intermediate metal graphs to free memory
                if device.type == "mps":
                    torch.mps.empty_cache()

        # Combine batches
        samples = torch.cat(all_samples, dim=0)

        folder_tag = json_name.replace(".json", "")
        save_individual_images(samples, os.path.join(args.output_dir, folder_tag))
        save_result_grids(samples, f"results/grid_{folder_tag}.png", nrow=8)

        acc = evaluator.eval(samples, test_labels)
        print(f"Accuracy for {json_name}: {acc:.4f}")

    print("\nGenerating mandatory denoising process visualization...")
    generate_denoising_process(diffusion, obj2idx, device, args.cfg_scale)


if __name__ == "__main__":
    main()
