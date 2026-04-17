"""Sample synthetic Dirichlet boundary conditions on the unit square grid."""

from __future__ import annotations

import math

import torch


def sample_boundary(
    H: int,
    W: int,
    mode: str = "fourier",
    *,
    device=None,
    dtype=torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample boundary values on an H×W grid; interior set to 0.

    Parameters
    ----------
    H, W : int
        Grid size.
    mode : {"random", "fourier"}
        Sampling strategy for edge values.

    Returns
    -------
    grid : Tensor [H, W]
        Boundary pixels filled; interior zeros.
    """
    if mode not in ("random", "fourier"):
        raise ValueError(f"Unknown mode {mode!r}, expected 'random' or 'fourier'")

    dev = device or "cpu"
    grid = torch.zeros(H, W, dtype=dtype, device=dev)

    if mode == "random":
        # Random corners, linear blends along edges, then edge noise
        c00 = torch.rand((), generator=generator, dtype=dtype, device=dev)
        c10 = torch.rand((), generator=generator, dtype=dtype, device=dev)
        c01 = torch.rand((), generator=generator, dtype=dtype, device=dev)
        c11 = torch.rand((), generator=generator, dtype=dtype, device=dev)

        j = torch.linspace(0, 1, W, dtype=dtype, device=dev)
        i = torch.linspace(0, 1, H, dtype=dtype, device=dev)
        top = c00 * (1 - j) + c01 * j
        bottom = c10 * (1 - j) + c11 * j
        left = c00 * (1 - i) + c10 * i
        right = c01 * (1 - i) + c11 * i
        grid[0, :] = top
        grid[-1, :] = bottom
        grid[:, 0] = left
        grid[:, -1] = right
        noise_scale = 0.15
        grid[0, 1:-1] += noise_scale * torch.randn(W - 2, generator=generator, dtype=dtype, device=dev)
        grid[-1, 1:-1] += noise_scale * torch.randn(W - 2, generator=generator, dtype=dtype, device=dev)
        grid[1:-1, 0] += noise_scale * torch.randn(H - 2, generator=generator, dtype=dtype, device=dev)
        grid[1:-1, -1] += noise_scale * torch.randn(H - 2, generator=generator, dtype=dtype, device=dev)
        grid[0, 0] = c00
        grid[0, -1] = c01
        grid[-1, 0] = c10
        grid[-1, -1] = c11

    else:  # fourier — low-frequency sines along edges
        n_modes = 3
        freqs = torch.arange(1, n_modes + 1, dtype=dtype, device=dev)
        j = torch.linspace(0, math.pi, W, dtype=dtype, device=dev)
        idx = torch.linspace(0, math.pi, H, dtype=dtype, device=dev)

        top = torch.zeros(W, dtype=dtype, device=dev)
        bottom = torch.zeros(W, dtype=dtype, device=dev)
        left = torch.zeros(H, dtype=dtype, device=dev)
        right = torch.zeros(H, dtype=dtype, device=dev)
        for k in range(n_modes):
            ak = torch.randn((), generator=generator, dtype=dtype, device=dev) * (1.0 / (k + 1))
            bk = torch.randn((), generator=generator, dtype=dtype, device=dev) * (1.0 / (k + 1))
            top += ak * torch.sin(freqs[k] * j)
            bottom += ak * 0.7 * torch.sin(freqs[k] * j + 0.3)
            left += bk * torch.sin(freqs[k] * idx)
            right += bk * 0.7 * torch.sin(freqs[k] * idx + 0.3)

        grid[0, :] = top
        grid[-1, :] = bottom
        grid[:, 0] = left
        grid[:, -1] = right
        # Consistent corners (average disagreeing 1D limits)
        grid[0, 0] = 0.5 * (top[0] + left[0])
        grid[0, -1] = 0.5 * (top[-1] + right[0])
        grid[-1, 0] = 0.5 * (bottom[0] + left[-1])
        grid[-1, -1] = 0.5 * (bottom[-1] + right[-1])

        m = grid.abs().max().clamp_min(1e-6)
        grid = grid / m

    return grid
