import torch
from unet import StrongUNet
from scheduler import linear_beta_schedule
import torch.nn.functional as F
from diffusion_utils import sample_timestep
from data import IMG_SIZE
import matplotlib.pyplot as plt

T = 200

def show(img):
    arr = (img.clamp(-1,1)+1)/2
    arr = arr[0].permute(1,2,0).cpu().numpy()
    plt.imshow(arr)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = StrongUNet().to(device)
    model.load_state_dict(torch.load("model_final.pt", map_location=device))
    model.eval()

    betas = linear_beta_schedule(T)
    alphas = 1 - betas
    ac = torch.cumprod(alphas, dim=0)
    ac_prev = torch.cat([torch.tensor([1.0]), ac[:-1]])
    pcs = {
        "betas": betas.to(device),
        "sqrt_recip_alphas": torch.sqrt(1.0/alphas).to(device),
        "sqrt_om_ac": torch.sqrt(1-ac).to(device),
        "posterior_variance": (betas*(1-ac_prev)/(1-ac)).to(device)
    }

    img = torch.randn((1,3,IMG_SIZE,IMG_SIZE), device=device)

    for t in reversed(range(T)):
        tt = torch.tensor([t], device=device)
        img = sample_timestep(
            img, tt, model,
            pcs["betas"], pcs["sqrt_recip_alphas"],
            pcs["sqrt_om_ac"], pcs["posterior_variance"]
        )
        img = img.clamp(-1,1)

    show(img)
