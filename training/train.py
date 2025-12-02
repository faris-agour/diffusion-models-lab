import os
import torch
from torch import optim
from tqdm import tqdm

from models.model import StrongUNet
from methods.ddpm import DDPM
from data import load_dataloader


def train_ddpm(
    epochs: int = 20,
    timesteps: int = 200,
    lr: float = 2e-4,
    img_size: int = 64,
    batch_size: int = 8,
    dataset_name: str = "tanganke/stanford_cars",
    checkpoint_dir: str = "checkpoints",
):
    """
    High-level training loop for DDPM on the Stanford Cars dataset.

    This:
      - builds the dataloader
      - creates UNet + DDPM wrapper
      - runs the training loop
      - saves the final model weights
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------------
    # 1) Dataloader
    # ------------------------------------------------------------------
    dataloader = load_dataloader(
        dataset_name=dataset_name,
        img_size=img_size,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # 2) Model + DDPM wrapper
    # ------------------------------------------------------------------
    unet = StrongUNet().to(device)
    ddpm = DDPM(
        model=unet,
        timesteps=timesteps,
        beta_start=1e-4,
        beta_end=2e-2,
        device=device,
    )

    optimizer = optim.Adam(ddpm.model.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float("inf")
    best_path = os.path.join(checkpoint_dir, "ddpm_best.pt")

    # ------------------------------------------------------------------
    # 3) Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        ddpm.model.train()
        epoch_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for x_batch in loop:
            # if your DataLoader returns (x, y), adjust accordingly:
            if isinstance(x_batch, (list, tuple)) and len(x_batch) == 2:
                x_batch, _ = x_batch

            x_batch = x_batch.to(device)

            loss = ddpm.loss(x_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"[INFO] Epoch {epoch} | Avg Loss: {avg_loss:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ddpm.model.state_dict(), best_path)
            print(f"[INFO] Saved new best model to: {best_path}")

    # Also save final model
    final_path = os.path.join(checkpoint_dir, "ddpm_final.pt")
    torch.save(ddpm.model.state_dict(), final_path)
    print(f"[INFO] Training finished. Final model saved to: {final_path}")


if __name__ == "__main__":
    train_ddpm()
