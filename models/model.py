import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Sinusoidal time embedding
# -------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B] (long or float)
        returns: [B, dim]
        """
        if t.dtype != torch.float32:
            t = t.float()

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim
        )  # [half_dim]
        args = t[:, None] * freqs[None, :]  # [B, half_dim]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # pad if odd
            emb = F.pad(emb, (0, 1))
        return emb


# -------------------------
# Basic building blocks
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_dim = time_dim

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        if time_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_ch),
            )
        else:
            self.time_mlp = None

    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        if self.time_mlp is not None and t_emb is not None:
            # t_emb: [B, time_dim] -> [B, C, 1, 1]
            time_out = self.time_mlp(t_emb)
            h = h + time_out[:, :, None, None]

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.res1 = ResidualBlock(in_ch, out_ch, time_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res1 = ResidualBlock(out_ch * 2, out_ch, time_dim)  # concat skip
        self.res2 = ResidualBlock(out_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        # match spatial size (safety)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


# -------------------------
# StrongUNet v2
# -------------------------
class StrongUNet(nn.Module):
    """
    Stronger UNet for DDPM v2:
    - base channels = 64
    - depth = 4 levels (64 → 128 → 256 → 512)
    - residual blocks with time conditioning
    """

    def __init__(self, img_ch: int = 3, time_dim: int = 256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder
        self.in_conv = nn.Conv2d(img_ch, 64, 3, padding=1)

        self.down1 = DownBlock(64, 64, time_dim)    # 64x64  -> 32x32
        self.down2 = DownBlock(64, 128, time_dim)   # 32x32  -> 16x16
        self.down3 = DownBlock(128, 256, time_dim)  # 16x16  -> 8x8
        self.down4 = DownBlock(256, 512, time_dim)  # 8x8    -> 4x4

        # Bottleneck
        self.bot1 = ResidualBlock(512, 512, time_dim)
        self.bot2 = ResidualBlock(512, 512, time_dim)

        # Decoder
        self.up1 = UpBlock(512, 256, time_dim)      # 4x4  -> 8x8
        self.up2 = UpBlock(256, 128, time_dim)      # 8x8  -> 16x16
        self.up3 = UpBlock(128, 64, time_dim)       # 16x16 -> 32x32
        self.up4 = UpBlock(64, 64, time_dim)        # 32x32 -> 64x64

        self.out_conv = nn.Conv2d(64, img_ch, 1)

    def forward(self, x, t):
        """
        x: [B, 3, H, W]
        t: [B]
        """
        t_emb = self.time_mlp(t)  # [B, time_dim]

        x = self.in_conv(x)

        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)
        x, s4 = self.down4(x, t_emb)

        x = self.bot1(x, t_emb)
        x = self.bot2(x, t_emb)

        x = self.up1(x, s4, t_emb)
        x = self.up2(x, s3, t_emb)
        x = self.up3(x, s2, t_emb)
        x = self.up4(x, s1, t_emb)

        x = self.out_conv(x)
        return x
