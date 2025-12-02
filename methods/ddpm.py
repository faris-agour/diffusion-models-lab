from diffusion.utils import forward_diffusion_sample
import torch
import torch.nn.functional as F
from diffusion.scheduler import (
    linear_beta_schedule,
    cosine_beta_schedule,
    get_index_from_list,
)


class DDPM:
    """
    Class wrapper around a UNet-like noise-prediction model that implements
    the core DDPM logic: forward noising, reverse denoising, loss, and sampling.
    """

    def __init__(
        self,
        model,
        timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: str = "cosine",
        device: str = "cuda"
    ):
        """
        Args:
            model:      Noise prediction model ε_θ(x_t, t). Expects forward(x, t).
            timesteps:  Number of diffusion steps T.
            beta_start: Starting value for linear beta schedule.
            beta_end:   Final value for linear beta schedule.
            device:     "cuda" or "cpu".
        """
        self.model = model.to(device)
        self.device = device
        self.T = timesteps

        # --- choose beta schedule ---
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.betas = betas.to(device)  # [T]
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]],
            dim=0
        )

        # --- useful precomputed terms ---
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # p(x_{t-1} | x_t, x_0) posterior variance
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

    # ------------------------------------------------------------------
    # Forward process q(x_t | x_0)
    # ------------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Diffuse the data (add noise) at a given time step t.

        q(x_t | x_0) = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε

        Args:
            x0:    clean images, shape [B, C, H, W].
            t:     time steps, shape [B], int64.
            noise: optional noise tensor. If None, sampled from N(0, I).

        Returns:
            x_t:   noised images
            noise: the noise that was used
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ac = get_index_from_list(
            self.sqrt_alphas_cumprod, t, x0.shape
        )  # [B, 1, 1, 1]
        sqrt_om_ac = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )  # [B, 1, 1, 1]

        x_t = sqrt_ac * x0 + sqrt_om_ac * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Reverse step p(x_{t-1} | x_t)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor):
        """
        Single reverse denoising step: x_t -> x_{t-1}

        model predicts ε_θ(x_t, t), and we compute the mean of
        p(x_{t-1} | x_t) as in the DDPM paper.
        """
        betas_t = get_index_from_list(self.betas, t, x_t.shape)
        sqrt_recip_alpha_t = get_index_from_list(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_om_ac_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        # Predict noise with the model
        eps_theta = self.model(x_t, t)

        # Equation (11) in DDPM paper: compute the "model mean"
        model_mean = sqrt_recip_alpha_t * (
            x_t - betas_t * eps_theta / sqrt_om_ac_t
        )

        posterior_variance_t = get_index_from_list(
            self.posterior_variance, t, x_t.shape
        )

        # if t == 0, no noise is added (last step)
        # here we assume all batch elements share the same t
        if (t == 0).all():
            return model_mean

        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # ------------------------------------------------------------------
    # Full sampling loop
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, batch_size: int, img_channels: int, img_size: int) -> torch.Tensor:
        """
        Run the full reverse process starting from pure noise.

        Args:
            batch_size:  number of images to sample
            img_channels: number of channels (e.g., 3 for RGB)
            img_size:    spatial size H = W

        Returns:
            x_0 samples in [-1, 1], shape [B, C, H, W]
        """
        x = torch.randn(
            (batch_size, img_channels, img_size, img_size),
            device=self.device
        )

        for t in reversed(range(self.T)):
            t_batch = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long
            )
            x = self.p_sample(x, t_batch)
            x = x.clamp(-1.0, 1.0)

        return x

    # ------------------------------------------------------------------
    # Training loss (noise prediction objective)
    # ------------------------------------------------------------------
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Compute the DDPM training loss for a batch of clean images.

        We randomly choose a time step t, diffuse x0 -> x_t, then
        train the model to predict the noise ε that was used.

        Loss = E_{t, ε}[ || ε - ε_θ(x_t, t) ||^2 ]
        """
        x0 = x0.to(self.device)
        B = x0.shape[0]

        # Sample random timesteps for each image in the batch
        t = torch.randint(
            low=0, high=self.T, size=(B,), device=self.device, dtype=torch.long
        )

        # Diffuse the images
        x_t, noise = self.q_sample(x0, t)

        # Predict noise with the model
        noise_pred = self.model(x_t, t)

        return F.mse_loss(noise_pred, noise)
