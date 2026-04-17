"""Small U-Net for epsilon prediction with boundary conditioning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal embedding + MLP (DDPM-style)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / max(half - 1, 1))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] indices or floats
        x = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        emb = torch.cat([x.sin(), x.cos()], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        elif emb.shape[1] > self.dim:
            emb = emb[:, : self.dim]
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, ch: int, tdim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.t_proj = nn.Linear(tdim, ch)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x))
        te_b = self.t_proj(te)[:, :, None, None]
        h = h + te_b
        h = F.silu(self.conv2(h))
        return x + h


class SmallUNet(nn.Module):
    """
    Predicts noise ε with inputs concatenated along channels:
    [noisy field | boundary] -> 2 channels.
    """

    def __init__(self, in_ch: int = 2, base: int = 32, time_dim: int = 128) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(time_dim)

        self.down1 = nn.Conv2d(in_ch, base, 3, padding=1)
        self.rb1 = ResBlock(base, time_dim)
        self.down = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.rb2 = ResBlock(base * 2, time_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.rb3 = ResBlock(base * 4, time_dim)

        self.mid = ResBlock(base * 4, time_dim)

        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.urb1 = ResBlock(base * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.urb2 = ResBlock(base, time_dim)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x_t: torch.Tensor, boundary: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t : [B, 1, H, W]
        boundary : [B, 1, H, W]
        t : [B] long, diffusion indices
        """
        te = self.time_embed(t)
        x = torch.cat([x_t, boundary], dim=1)

        h1 = self.rb1(F.silu(self.down1(x)), te)
        h2 = self.rb2(self.down(h1), te)
        h3 = self.rb3(self.down2(h2), te)
        h = self.mid(h3, te)
        h = self.urb1(self.up1(h), te)
        if h.shape[2:] != h2.shape[2:]:
            h = F.interpolate(h, size=h2.shape[2:], mode="bilinear", align_corners=False)
        h = h + h2
        h = self.urb2(self.up2(h), te)
        if h.shape[2:] != h1.shape[2:]:
            h = F.interpolate(h, size=h1.shape[2:], mode="bilinear", align_corners=False)
        h = h + h1
        return self.out(F.silu(h))
