"""PyTorch Dataset for boundary-conditioned heat trajectories."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class HeatTrajectoryDataset(Dataset):
    """
    Each item is one PDE sample.

    Returns
    -------
    boundary : Tensor [1, H, W]
    trajectory : Tensor [T, H, W]
    """

    def __init__(
        self,
        boundaries: torch.Tensor,
        trajectories: torch.Tensor,
    ) -> None:
        if boundaries.shape[0] != trajectories.shape[0]:
            raise ValueError("boundaries/trajectories batch dimension mismatch")
        self.boundaries = boundaries
        self.trajectories = trajectories

    def __len__(self) -> int:
        return self.boundaries.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        b = self.boundaries[idx]
        traj = self.trajectories[idx]
        return {
            "boundary": b.unsqueeze(0),
            "trajectory": traj,
        }

    @staticmethod
    def sample_random_pde_time(trajectory: torch.Tensor, generator: torch.Generator | None = None) -> int:
        """Sample a random PDE frame index along time dimension [0, T-1]."""
        t = trajectory.shape[0]
        return int(torch.randint(0, t, (1,), generator=generator).item())
