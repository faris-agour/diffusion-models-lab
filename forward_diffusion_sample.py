import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


T = 600
betas = linear_beta_schedule(T).to(device)
alphas = (1. - betas)
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=device), alphas_cumprod[:-1]])

sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def get_index_from_list(vals, t, x_shape):
    out = vals.gather(-1, t.cpu())
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alpha = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
    return x_t, noise
