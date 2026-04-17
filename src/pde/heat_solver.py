"""2D heat equation on [0,1]^2 with explicit finite differences and Dirichlet BCs."""

from __future__ import annotations

import torch


def grid_spacing(H: int, W: int) -> float:
    """Uniform spacing for domain [0,1]^2 with H×W nodes on the closed domain."""
    return 1.0 / max(H - 1, W - 1, 1)


def enforce_dirichlet(u: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    """Copy edge values from `boundary` onto `u` (same shape [H, W])."""
    u = u.clone()
    u[0, :] = boundary[0, :]
    u[-1, :] = boundary[-1, :]
    u[:, 0] = boundary[:, 0]
    u[:, -1] = boundary[:, -1]
    return u


def construct_full_grid_from_boundary(boundary: torch.Tensor) -> torch.Tensor:
    """
    Build initial field: prescribed Dirichlet values on ∂Ω, interior zeros.

    Parameters
    ----------
    boundary : Tensor [H, W]
        Edge values; interior entries may be arbitrary (overwritten).

    Returns
    -------
    Tensor [H, W]
    """
    H, W = boundary.shape
    u = torch.zeros(H, W, dtype=boundary.dtype, device=boundary.device)
    return enforce_dirichlet(u, boundary)


def stable_dt(dx: float, kappa: float, safety: float = 0.45) -> float:
    """
    Stability for 2D explicit heat: dt <= dx^2 / (4*kappa).
    Use `safety` < 0.5 for margin.
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    return safety * dx**2 / (4.0 * kappa)


def laplacian_5pt(u: torch.Tensor, dx: float) -> torch.Tensor:
    """5-point Laplacian on interior; edges left as garbage (not used)."""
    # Central differences; interior indices 1..H-2, 1..W-2
    lap = torch.zeros_like(u)
    lap[1:-1, 1:-1] = (
        u[2:, 1:-1]
        + u[:-2, 1:-1]
        + u[1:-1, 2:]
        + u[1:-1, :-2]
        - 4.0 * u[1:-1, 1:-1]
    ) / (dx**2)
    return lap


def simulate_heat_equation(
    boundary: torch.Tensor,
    num_steps: int,
    dt: float,
    kappa: float,
) -> torch.Tensor:
    """
    Solve ∂u/∂t = κ Δu on Ω = [0,1]^2 with u|∂Ω fixed to `boundary` edges.

    Parameters
    ----------
    boundary : Tensor [H, W]
        Dirichlet values on the boundary; interior ignored for BC enforcement.
    num_steps : int
        Number of time steps (returns T = num_steps + 1 frames: u^0, ..., u^num_steps).
    dt : float
        Time step (should satisfy stability w.r.t. dx, kappa).
    kappa : float
        Thermal diffusivity κ.

    Returns
    -------
    trajectory : Tensor [T, H, W]
        T = num_steps + 1. Boundary rows/columns match `boundary` at every time.
    """
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    H, W = boundary.shape
    dx = grid_spacing(H, W)

    u = construct_full_grid_from_boundary(boundary)
    frames = [u.clone()]
    for _ in range(num_steps):
        lap = laplacian_5pt(u, dx)
        # Forward Euler on interior only
        u = u.clone()
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * kappa * lap[1:-1, 1:-1]
        u = enforce_dirichlet(u, boundary)
        frames.append(u.clone())
    return torch.stack(frames, dim=0)


def _test_heat_solver() -> None:
    """Quick sanity checks: fixed boundary and smoothing in the interior."""
    torch.manual_seed(0)
    H, W = 32, 32
    b = torch.zeros(H, W)
    b[0, :] = 1.0
    b[-1, :] = 0.0
    b[:, 0] = torch.linspace(1, 0, H)
    b[:, -1] = torch.linspace(1, 0, H)

    dx = grid_spacing(H, W)
    kappa = 0.1
    dt = stable_dt(dx, kappa)
    traj = simulate_heat_equation(b, num_steps=100, dt=dt, kappa=kappa)

    # Boundary fixed
    for t in range(traj.shape[0]):
        u = traj[t]
        assert torch.allclose(u[0, :], b[0, :])
        assert torch.allclose(u[-1, :], b[-1, :])
        assert torch.allclose(u[:, 0], b[:, 0])
        assert torch.allclose(u[:, -1], b[:, -1])

    # Interior becomes smoother (variance decreases) for this setup
    var0 = traj[0, 1:-1, 1:-1].var()
    var_last = traj[-1, 1:-1, 1:-1].var()
    assert var_last < var0, (var0.item(), var_last.item())

    print("heat_solver tests passed.")


if __name__ == "__main__":
    _test_heat_solver()
