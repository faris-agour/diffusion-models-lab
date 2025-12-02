import torch
from diffusion.scheduler import get_index_from_list

def forward_diffusion_sample(x0, t, betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x0)
    s1 = get_index_from_list(sqrt_alphas_cumprod, t, x0.shape)
    s2 = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return s1 * x0 + s2 * noise, noise