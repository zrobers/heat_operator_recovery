"""Minimal DDPM-style diffusion utilities (linear beta schedule)."""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianDiffusion(nn.Module):
    """Stores β schedule and derived α̅ quantities for training and sampling."""

    def __init__(self, num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        if num_timesteps < 2:
            raise ValueError("num_timesteps must be at least 2")
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.num_timesteps = num_timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def device(self) -> torch.device:
        return self.betas.device

    def sample_noise(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """
        q(x_t | x_0) sample: x_t = sqrt(α̅_t) x_0 + sqrt(1-α̅_t) ε.

        Parameters
        ----------
        x0 : Tensor [B, ...]
        t : LongTensor [B] in [0, num_timesteps - 1]
        """
        if noise is None:
            noise = self.sample_noise(x0)
        sa = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        sn = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        return sa * x0 + sn * noise

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Tweedie-style estimate x̂_0 from x_t and predicted noise ε."""
        sa = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x_t.dim() - 1)))
        sn = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x_t.dim() - 1)))
        return (x_t - sn * noise) / sa.clamp_min(1e-8)

    def predict_denoised(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        boundary: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Run model to get ε(x_t, boundary, t), then return estimated x_0."""
        eps = model(x_t, boundary, t)
        return self.predict_x0_from_noise(x_t, t, eps)
