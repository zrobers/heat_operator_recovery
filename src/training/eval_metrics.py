"""Final-state denoising metrics (for comparing trajectory-aligned vs standard training)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import HeatTrajectoryDataset
from src.models.diffusion import GaussianDiffusion


@torch.no_grad()
def final_state_mse_batch(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    trajectory: torch.Tensor,
    boundary: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Mean squared error between Tweedie x̂_0 and ground-truth final field u(·, T).

    trajectory : [B, T, H, W] — uses last frame
    boundary : [B, 1, H, W]
    """
    b, t_pde, _, _ = trajectory.shape
    u_final = trajectory[:, -1].unsqueeze(1).to(device)
    boundary = boundary.to(device)
    t = torch.randint(0, diffusion.num_timesteps, (b,), device=device, dtype=torch.long)
    noise = diffusion.sample_noise(u_final)
    x_t = diffusion.add_noise(u_final, t, noise=noise)
    eps_pred = model(x_t, boundary, t)
    x0_hat = diffusion.predict_x0_from_noise(x_t, t, eps_pred)
    return float(torch.mean((x0_hat - u_final) ** 2).item())


@torch.no_grad()
def dataset_mean_final_mse(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    data_path: str | Path,
    batch_size: int,
    device: torch.device,
) -> float:
    """Average `final_state_mse_batch` over an entire saved split (.pt file)."""
    pack = torch.load(Path(data_path), map_location="cpu")
    boundaries = pack["boundaries"]
    trajectories = pack["trajectories"]
    u_final = pack.get("u_final_solve")
    ds = HeatTrajectoryDataset(boundaries, trajectories, u_final_solve=u_final)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        traj = batch["trajectory"]
        bsz = traj.shape[0]
        m = final_state_mse_batch(
            model,
            diffusion,
            traj,
            batch["boundary"],
            device,
        )
        total += m * bsz
        n += bsz
    return total / max(n, 1)
