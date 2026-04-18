"""2D heat equation on [0,1]^2 with explicit finite differences and Dirichlet BCs."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


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


def _lininterp_1d(vals: torch.Tensor, n_out: int) -> torch.Tensor:
    """Linearly resample 1D values ``vals`` of length n_in to length n_out."""
    n_in = vals.shape[0]
    if n_out == 1:
        return vals[:1].clone()
    if n_out < 1:
        raise ValueError("n_out must be positive")
    idx = torch.linspace(0, n_in - 1, n_out, device=vals.device, dtype=vals.dtype)
    i0 = idx.floor().long().clamp(0, n_in - 2)
    i1 = i0 + 1
    w = idx - i0.float()
    return vals[i0] * (1.0 - w) + vals[i1] * w


def upsample_boundary(boundary: torch.Tensor, Hf: int, Wf: int) -> torch.Tensor:
    """
    Upsample Dirichlet data defined on a coarse grid to a finer grid by 1D interpolation
    along each edge (interior of returned tensor is zero except edges).
    """
    Hc, Wc = boundary.shape
    if Hf < Hc or Wf < Wc:
        raise ValueError("upsample_boundary expects a finer target grid than input")
    out = torch.zeros(Hf, Wf, dtype=boundary.dtype, device=boundary.device)
    out[0, :] = _lininterp_1d(boundary[0, :], Wf)
    out[-1, :] = _lininterp_1d(boundary[-1, :], Wf)
    out[:, 0] = _lininterp_1d(boundary[:, 0], Hf)
    out[:, -1] = _lininterp_1d(boundary[:, -1], Hf)
    out[0, 0] = boundary[0, 0]
    out[0, -1] = boundary[0, -1]
    out[-1, 0] = boundary[-1, 0]
    out[-1, -1] = boundary[-1, -1]
    return out


def spatial_downsample(field: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
    """Bilinear downsample [H, W] or time stack [T, H, W] to (Ht, Wt)."""
    if field.dim() == 2:
        x = field.unsqueeze(0).unsqueeze(0)
        y = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
        return y[0, 0]
    if field.dim() == 3:
        x = field.unsqueeze(1)
        y = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
        return y[:, 0]
    raise ValueError("field must be [H,W] or [T,H,W]")


def simulate_heat_equation_multires(
    boundary_fine: torch.Tensor,
    num_pde_steps: int,
    kappa: float,
    store_grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Integrate on ``boundary_fine``'s grid with a stable fine timestep, but match the
    physical spacing of time used by a reference grid of size ``store_grid_size``.

    Returns a trajectory at **store** resolution with ``num_pde_steps + 1`` frames,
    covering the same physical end time as ``num_pde_steps`` coarse Euler steps would
    on the store grid (dt_coarse = stable_dt(dx_store)).

    Also returns the **final** field on the **solve** grid (last time slice of the
    fine integration), for storing a high-resolution terminal state alongside
    downsampled trajectories.

    Returns
    -------
    trajectory_store : Tensor [T, Hs, Ws]
    u_final_fine : Tensor [Hf, Wf]
        Field u(·, T_end) on the solve grid.
    info : dict with micro, dt_coarse, dt_sub, num_fine_steps
    """
    if num_pde_steps < 0:
        raise ValueError("num_pde_steps must be non-negative")
    Hf, Wf = boundary_fine.shape
    Hs = Ws = store_grid_size
    dt_coarse = stable_dt(grid_spacing(Hs, Ws), kappa)
    dt_fine_lim = stable_dt(grid_spacing(Hf, Wf), kappa)
    micro = max(1, int(math.ceil(dt_coarse / dt_fine_lim)))
    sub_dt = dt_coarse / micro
    while sub_dt > dt_fine_lim + 1e-14:
        micro += 1
        sub_dt = dt_coarse / micro

    n_fine = num_pde_steps * micro
    traj_fine = simulate_heat_equation(boundary_fine, num_steps=n_fine, dt=sub_dt, kappa=kappa)
    idx = torch.arange(0, n_fine + 1, micro, device=traj_fine.device, dtype=torch.long)
    traj_coarse_time = traj_fine[idx]
    traj_store = spatial_downsample(traj_coarse_time, Hs, Ws)
    u_final_fine = traj_fine[-1].clone()

    info = {
        "micro_steps_per_coarse": micro,
        "dt_coarse": dt_coarse,
        "dt_sub": sub_dt,
        "num_fine_steps": n_fine,
        "solve_grid_size": Hf,
    }
    return traj_store, u_final_fine, info


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

    assert torch.isfinite(traj).all()

    print("heat_solver tests passed.")


if __name__ == "__main__":
    _test_heat_solver()
