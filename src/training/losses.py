"""Denoising + trajectory-consistency losses for mechanism alignment."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.diffusion import GaussianDiffusion


def diffusion_to_pde_index(
    t: torch.Tensor,
    num_diffusion: int,
    num_pde: int,
) -> torch.Tensor:
    """Map diffusion timestep index (same scale as training) to PDE frame index."""
    if num_pde < 1:
        raise ValueError("num_pde must be positive")
    if num_diffusion < 2:
        return torch.zeros_like(t, dtype=torch.long)
    scale = (num_pde - 1).float() / float(num_diffusion - 1)
    return (t.float() * scale).round().long().clamp(0, num_pde - 1)


def base_noise_loss(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    x0: torch.Tensor,
    boundary: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """L_base = MSE(ε_pred, ε_true) for random (x0, t) pairs."""
    if noise is None:
        noise = diffusion.sample_noise(x0)
    x_t = diffusion.add_noise(x0, t, noise=noise)
    eps_pred = model(x_t, boundary, t)
    return torch.mean((eps_pred - noise) ** 2)


def trajectory_consistency_loss(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    trajectory: torch.Tensor,
    boundary: torch.Tensor,
    t_k: torch.Tensor,
) -> torch.Tensor:
    """
    Align denoising estimates with physical trajectory frames.

    For each diffusion index in ``t_k``, map to a PDE time index τ and compare
    the Tweedie estimate x̂_0(x_{t_k}, τ) to u(τ).

    Parameters
    ----------
    trajectory : [B, T_pde, H, W]
    boundary : [B, 1, H, W]
    t_k : LongTensor [K] diffusion timestep indices (shared across batch)
    """
    b, t_pde, _, _ = trajectory.shape
    device = trajectory.device
    losses: list[torch.Tensor] = []
    num_diff = diffusion.num_timesteps

    for idx in range(t_k.shape[0]):
        tt = t_k[idx].expand(b).to(device=device, dtype=torch.long)
        phys = diffusion_to_pde_index(tt, num_diff, t_pde)
        u_true = trajectory[torch.arange(b, device=device), phys].unsqueeze(1)
        noise = diffusion.sample_noise(u_true)
        x_t = diffusion.add_noise(u_true, tt, noise=noise)
        eps_pred = model(x_t, boundary, tt)
        x0_hat = diffusion.predict_x0_from_noise(x_t, tt, eps_pred)
        losses.append(torch.mean((x0_hat - u_true) ** 2))

    return torch.stack(losses).mean()


def total_loss(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    trajectory: torch.Tensor,
    boundary: torch.Tensor,
    t_diffusion: torch.Tensor,
    t_traj_k: torch.Tensor,
    lambda_traj: float,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L = L_base + λ * L_traj.

    Returns
    -------
    total, base, traj
    """
    b, t_pde, _, _ = trajectory.shape
    # Random PDE frame for base loss
    pde_idx = torch.randint(0, t_pde, (b,), device=trajectory.device)
    x0 = trajectory[torch.arange(b, device=trajectory.device), pde_idx].unsqueeze(1)

    l_base = base_noise_loss(model, diffusion, x0, boundary, t_diffusion, noise=noise)

    if lambda_traj > 0 and t_traj_k.numel() > 0:
        l_traj = trajectory_consistency_loss(model, diffusion, trajectory, boundary, t_traj_k)
    else:
        l_traj = torch.zeros((), device=trajectory.device, dtype=l_base.dtype)

    l_tot = l_base + lambda_traj * l_traj
    return l_tot, l_base, l_traj
