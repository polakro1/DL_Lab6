import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import os
import argparse

from dataset import ICLEVRDataset
from model import UNet
from diffusion import DDPM
from utils import EMA, save_result_grids
from evaluator import evaluation_model


def train(args):
    # Hyperparameters
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    base_channels = args.base_channels
    timesteps = args.timesteps
    cfg_dropout = args.cfg_dropout
    ema_decay = args.ema_decay
    eval_epoch = args.eval_epoch

    # Weights & Biases initialization
    logger = wandb.init(
        project="Deep-Learning-Lab6",
        name=args.run_name,
        config=args,
        save_code=True,
    )

    # Data loading
    train_dataset = ICLEVRDataset(
        data_dir="data/iclevr",
        json_file="data/train.json",
        objects_file="data/objects.json",
        mode="train",
    )
    test_dataset = ICLEVRDataset(
        data_dir="data/iclevr",
        json_file="data/test.json",
        objects_file="data/objects.json",
        mode="test",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Model, Diffusion, EMA and Evaluator
    model = UNet(base_channels=base_channels).to(device)
    diffusion = DDPM(model, timesteps=timesteps).to(device)
    ema = EMA(model, target_decay=ema_decay, warmup_steps=args.ema_warmup_steps)
    evaluator = evaluation_model()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = GradScaler(
        device=device,
    )  # For AMP

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast(
                device_type=device.type, dtype=torch.float16
            ):  # Mixed Precision
                loss = diffusion.compute_loss(imgs, labels, p_uncond=cfg_dropout)

            scaler.scale(loss).backward()
            # Gradient clipping to prevent exploding gradients in early stages
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scale_before = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            scale_after = scaler.get_scale()

            if scale_before <= scale_after:
                ema.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Logging
        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch + 1})

        # Periodic Evaluation (e.g., every 10 epochs)
        if (epoch + 1) % eval_epoch == 0:
            ema.apply_shadow()  # Use stable weights for sampling
            model.eval()

            # Prepare test labels from dataset
            test_labels = torch.stack(
                [test_dataset[i] for i in range(len(test_dataset))]
            ).to(device)

            # DDIM Fast Sampling (50 steps)
            with torch.no_grad():
                samples = diffusion.ddim_sample(
                    shape=(len(test_labels), 3, 64, 64),
                    context=test_labels,
                    ddim_steps=50,
                    cfg_scale=3.0,
                )

                # ResNet18 Accuracy Evaluation[cite: 1, 3]
                acc = evaluator.eval(samples, test_labels)
                wandb.log({"test_acc": acc, "epoch": epoch + 1})
                print(f"\nEpoch {epoch+1} Test Accuracy: {acc:.4f}")

                # Save best model
                if acc > best_acc:
                    best_acc = acc
                    logger.summary["best_acc"] = best_acc
                    torch.save(
                        {
                            "model_state_dict": ema.shadow,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch + 1,
                            "acc": acc,
                        },
                        "best_model.pth",
                    )
                    wandb.save("best_model.pth")

                # Periodic visual check
                save_result_grids(samples, f"results/epoch_{epoch+1}.png")

            ema.restore_weights()  # Return to training weights

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_warmup_steps", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--run_name", type=str, default="Deep-Learning-Lab6")
    parser.add_argument(
        "--eval_epoch",
        type=int,
        default=5,
        help="Epoch interval for evaluation",
    )
    args = parser.parse_args()

    train(args=args)
