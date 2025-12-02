import math
import torch
import torch.nn.functional as F

def linear_beta_schedule(timesteps: int, start: float = 1e-4, end: float = 2e-2):
    return torch.linspace(start, end, timesteps, dtype=torch.float32)
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from Improved DDPM paper.
    Produces much better sample quality than pure linear.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


def get_index_from_list(vals, t, x_shape):
    t = t.to(vals.device).long()          # [B]
    out = vals.gather(0, t)               # [B]

    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)           # [B,1,1,1,...]

    return out