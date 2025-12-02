import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def denorm_to_numpy(x: torch.Tensor):
    """
    Convert a tensor in [-1, 1] to a NumPy array in [0, 1] (H, W, C).
    Accepts either [C,H,W] or [B,C,H,W] and always returns HWC.
    """
    if x.ndim == 4:
        x = x[0]  # take first image in batch

    x = x.detach().cpu().clamp(-1, 1)

    if x.ndim == 2:
        x = x.unsqueeze(0)  # [H,W] -> [1,H,W]

    if x.shape[0] == 1:
        # grayscale -> fake 3-channel RGB for visualization
        x = x.repeat(3, 1, 1)

    x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
    x = x.permute(1, 2, 0)  # [C,H,W] -> [H,W,C]
    return x.numpy()


def show_tensor_image(x: torch.Tensor, title: str = None):
    """
    Show a single image tensor ([-1,1]) as a matplotlib image.
    """
    img = denorm_to_numpy(x)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")


def show_tensor_batch(xb: torch.Tensor, n: int = 16, cols: int = 4, title: str = None):
    """
    Show a grid of images from a batch tensor in [-1,1].

    Args:
        xb:   batch tensor [B,C,H,W].
        n:    number of images to show.
        cols: how many columns in the grid.
    """
    xb = xb[:n].detach().cpu().clamp(-1, 1)
    grid = make_grid((xb + 1) / 2, nrow=cols)  # [C,H,W] in [0,1]
    grid = grid.permute(1, 2, 0).numpy()      # HWC

    plt.figure(figsize=(2.5 * cols, 2.5 * ((n + cols - 1) // cols)))
    if title:
        plt.title(title)
    plt.imshow(grid)
    plt.axis("off")
    plt.tight_layout()
