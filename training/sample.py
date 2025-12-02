import os
import torch
import matplotlib.pyplot as plt

from models.model import StrongUNet
from methods.ddpm import DDPM
from utils import show_tensor_batch


def main(
    checkpoint_path: str = "checkpoints/ddpm_best.pt",
    timesteps: int = 200,
    img_size: int = 64,
    batch_size: int = 4,
):
    """
    Load a trained DDPM model and generate samples.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Train the model first."
        )

    # Build UNet and DDPM wrapper
    unet = StrongUNet().to(device)
    ddpm = DDPM(
        model=unet,
        timesteps=timesteps,
        beta_start=1e-4,
        beta_end=2e-2,
        device=device,
    )

    # Load weights
    state = torch.load(checkpoint_path, map_location=device)
    ddpm.model.load_state_dict(state)
    ddpm.model.eval()

    # Sample images
    with torch.no_grad():
        samples = ddpm.sample(
            batch_size=batch_size,
            img_channels=3,
            img_size=img_size,
        )

    # Show or save images
    show_tensor_batch(samples, n=batch_size, cols=batch_size, title="DDPM Samples")
    plt.show()


if __name__ == "__main__":
    main()
