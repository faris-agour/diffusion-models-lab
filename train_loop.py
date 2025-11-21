import torch
from tqdm import tqdm
import torch.nn.functional as F
from diffusion_utils import forward_diffusion_sample

def train_diffusion(model, dataloader, betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, T, epochs, lr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    model.train()
    for epoch in range(1, epochs+1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        total = 0

        for xb, _ in loop:
            xb = xb.to(device)
            t = torch.randint(0, T, (xb.size(0),), device=device).long()

            x_t, noise = forward_diffusion_sample(
                xb, t, betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
            )

            noise_pred = model(x_t, t)
            loss = mse(noise_pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch} Avg Loss: {total/len(dataloader):.6f}")
