#!/usr/bin/env python3
"""Generate synthetic heat-equation trajectories; random train/test split (default 80/20)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.boundary import sample_boundary
from src.pde.heat_solver import (
    grid_spacing,
    simulate_heat_equation,
    simulate_heat_equation_multires,
    spatial_downsample,
    stable_dt,
)
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    store_h = store_w = int(cfg["grid_size"])
    solve_h = solve_w = int(cfg.get("solve_grid_size") or cfg["grid_size"])
    num_steps = int(cfg["num_pde_steps"])
    kappa = float(cfg["kappa"])
    boundary_mode = str(cfg.get("boundary_mode", "fourier"))
    n_train = int(cfg["train_samples"])
    n_test = int(cfg["test_samples"])
    total = n_train + n_test

    out_dir = ROOT / "outputs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    def one_sample(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Return (boundary_store, trajectory_store, u_final_solve, solve_info)."""
        if solve_h == store_h:
            dx = grid_spacing(store_h, store_w)
            dt = cfg.get("dt")
            if dt is None:
                dt = stable_dt(dx, kappa)
            else:
                dt = float(dt)
            traj = simulate_heat_equation(b, num_steps=num_steps, dt=dt, kappa=kappa)
            u_final = traj[-1]
            return b, traj, u_final, {"dt": dt, "solve_grid_size": solve_h}

        traj_store, u_final_fine, info = simulate_heat_equation_multires(
            b,
            num_pde_steps=num_steps,
            kappa=kappa,
            store_grid_size=store_h,
        )
        b_store = spatial_downsample(b, store_h, store_w)
        meta_extra = {**info, "dt_coarse": info["dt_coarse"]}
        return b_store, traj_store, u_final_fine, meta_extra

    g_sample = torch.Generator().manual_seed(seed)
    boundaries: list[torch.Tensor] = []
    trajs: list[torch.Tensor] = []
    finals: list[torch.Tensor] = []
    last_info: dict = {}
    for _ in tqdm(range(total), desc=f"generating ({total} samples)"):
        b = sample_boundary(solve_h, solve_w, mode=boundary_mode, generator=g_sample)
        b_st, traj, u_fin, last_info = one_sample(b)
        boundaries.append(b_st)
        trajs.append(traj)
        finals.append(u_fin)

    b_all = torch.stack(boundaries, dim=0)
    t_all = torch.stack(trajs, dim=0)
    f_all = torch.stack(finals, dim=0)

    # Random permutation for train/test (reproducible)
    g_split = torch.Generator().manual_seed(seed + 10_000)
    perm = torch.randperm(total, generator=g_split)
    i_tr = perm[:n_train]
    i_te = perm[n_train:]

    def subset(ix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return b_all[ix], t_all[ix], f_all[ix]

    train_b, train_t, train_f = subset(i_tr)
    test_b, test_t, test_f = subset(i_te)

    meta = {
        "grid_size": store_h,
        "solve_grid_size": solve_h,
        "final_grid_size": solve_h,
        "num_pde_steps": num_steps,
        "kappa": kappa,
        "boundary_mode": boundary_mode,
        "train_samples": n_train,
        "test_samples": n_test,
        "split_seed_offset": 10_000,
        "solve": last_info,
    }

    torch.save(
        {"boundaries": train_b, "trajectories": train_t, "u_final_solve": train_f, "meta": meta},
        out_dir / "train.pt",
    )
    torch.save(
        {"boundaries": test_b, "trajectories": test_t, "u_final_solve": test_f, "meta": meta},
        out_dir / "test.pt",
    )
    print(f"Wrote {out_dir / 'train.pt'} ({n_train}) and {out_dir / 'test.pt'} ({n_test})")
    print(f"  trajectories train: {train_t.shape}  test: {test_t.shape}")


if __name__ == "__main__":
    main()
