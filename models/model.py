import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t[:,None] * freq[None,:]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        t_emb = self.time_mlp(t)[:, :, None, None]
        h = self.relu(self.norm1(self.conv1(x)) + t_emb)
        h = self.norm2(self.conv2(h))
        return self.relu(h + self.skip(x))

class StrongUNet(nn.Module):
    def __init__(self, img_ch=3, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)

        self.down1 = ResBlock(img_ch, 64, time_dim)
        self.down2 = ResBlock(64, 128, time_dim)
        self.down3 = ResBlock(128, 256, time_dim)

        self.bot1 = ResBlock(256, 256, time_dim)
        self.bot2 = ResBlock(256, 256, time_dim)

        self.up1 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.up_block1 = ResBlock(512, 256, time_dim)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_block2 = ResBlock(256, 128, time_dim)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_block3 = ResBlock(128, 64, time_dim)

        self.out = nn.Conv2d(64, img_ch, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)

        x1 = self.down1(x, t)
        x2 = self.down2(self.pool(x1), t)
        x3 = self.down3(self.pool(x2), t)

        b = self.bot1(self.pool(x3), t)
        b = self.bot2(b, t)

        u1 = self.up1(b)
        u1 = self.up_block1(torch.cat([u1, x3], dim=1), t)

        u2 = self.up2(u1)
        u2 = self.up_block2(torch.cat([u2, x2], dim=1), t)

        u3 = self.up3(u2)
        u3 = self.up_block3(torch.cat([u3, x1], dim=1), t)

        return self.out(u3)
