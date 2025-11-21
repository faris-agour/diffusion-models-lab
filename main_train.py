import torch
import torch.nn.functional as F
from data import load_dataloader
from unet import StrongUNet
from train_loop import train_diffusion
from scheduler import linear_beta_schedule

T = 200

def precompute(T):
    betas = linear_beta_schedule(T)
    alphas = 1 - betas
    ac = torch.cumprod(alphas, dim=0)
    ac_prev = F.pad(ac[:-1], (1,0), value=1.0)
    return {
        "betas": betas,
        "sqrt_ac": torch.sqrt(ac),
        "sqrt_om_ac": torch.sqrt(1 - ac),
        "sqrt_recip_alphas": torch.sqrt(1.0/alphas),
        "posterior_variance": betas*(1 - ac_prev)/(1-ac),
    }

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    dl = load_dataloader()
    pcs = precompute(T)

    model = StrongUNet().to(device)

    train_diffusion(
        model=model,
        dataloader=dl,
        betas=pcs["betas"].to(device),
        sqrt_alphas_cumprod=pcs["sqrt_ac"].to(device),
        sqrt_one_minus_alphas_cumprod=pcs["sqrt_om_ac"].to(device),
        T=T,
        epochs=20,
        lr=2e-4,
        device=device
    )

    torch.save(model.state_dict(), "model_final.pt")
    print("Saved model_final.pt")
