import torch
from scheduler import get_index_from_list

def forward_diffusion_sample(x0, t, betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x0)
    s1 = get_index_from_list(sqrt_alphas_cumprod, t, x0.shape)
    s2 = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return s1 * x0 + s2 * noise, noise

@torch.no_grad()
def sample_timestep(x, t, model, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_recip = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    soc = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)

    model_mean = sqrt_recip * (x - betas_t * model(x, t) / soc)
    var = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(var) * noise
