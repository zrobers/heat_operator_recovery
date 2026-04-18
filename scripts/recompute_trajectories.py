#!/usr/bin/env python3
"""
Re-run the heat solver for existing saved boundaries.

If ``solve_grid_size`` > ``grid_size``, the boundary is edge-upsampled, integrated on the
fine grid with substeps per coarse time interval, then trajectories are spatially
downsampled to ``grid_size``. The terminal field ``u_final_solve`` is kept at solve
resolution. Original ``boundaries`` tensors are unchanged.

If ``solve_grid_size`` == ``grid_size``, trajectories are recomputed at the stored
resolution; ``u_final_solve`` is the last trajectory frame (same grid).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pde.heat_solver import (
    grid_spacing,
    simulate_heat_equation,
    simulate_heat_equation_multires,
    stable_dt,
    upsample_boundary,
)
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--solve-grid-size",
        type=int,
        default=None,
        help="Override config solve_grid_size",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    store_grid = int(cfg["grid_size"])
    solve_grid = int(args.solve_grid_size or cfg.get("solve_grid_size") or store_grid)
    num_steps = int(cfg["num_pde_steps"])
    kappa = float(cfg["kappa"])
    dt_cfg = cfg.get("dt")
    if dt_cfg is not None:
        dt_cfg = float(dt_cfg)

    data_dir = ROOT / "outputs" / "data"
    # Prefer train/test split; still process legacy val.pt if present
    candidates = ("train.pt", "test.pt", "val.pt")
    for name in candidates:
        path = data_dir / name
        if not path.is_file():
            print(f"Skip missing {path}")
            continue

        pack = torch.load(path, map_location="cpu")
        boundaries = pack["boundaries"]
        n, hc, wc = boundaries.shape
        if hc != wc:
            raise ValueError("Expected square grids in boundaries")
        if hc != store_grid:
            raise ValueError(
                f"{name}: boundaries grid {hc} != config grid_size {store_grid}"
            )

        trajs: list[torch.Tensor] = []
        finals: list[torch.Tensor] = []
        desc = f"{name} ({n} samples, solve={solve_grid})"
        for i in tqdm(range(n), desc=desc):
            b = boundaries[i]
            if solve_grid == store_grid:
                dx = grid_spacing(store_grid, store_grid)
                dt = dt_cfg if dt_cfg is not None else stable_dt(dx, kappa)
                traj = simulate_heat_equation(b, num_steps=num_steps, dt=dt, kappa=kappa)
                u_fin = traj[-1]
            else:
                b_fine = upsample_boundary(b, solve_grid, solve_grid)
                traj, u_fin, _ = simulate_heat_equation_multires(
                    b_fine,
                    num_pde_steps=num_steps,
                    kappa=kappa,
                    store_grid_size=store_grid,
                )
            trajs.append(traj)
            finals.append(u_fin)

        out_meta = dict(pack.get("meta", {}))
        out_meta["solve_grid_size"] = solve_grid
        out_meta["store_grid_size"] = store_grid
        out_meta["final_grid_size"] = solve_grid
        out_meta["recomputed_trajectories"] = True

        out = {
            "boundaries": boundaries,
            "trajectories": torch.stack(trajs, dim=0),
            "u_final_solve": torch.stack(finals, dim=0),
            "meta": out_meta,
        }
        torch.save(out, path)
        print(
            f"Wrote {path} trajectories {out['trajectories'].shape} "
            f"u_final_solve {out['u_final_solve'].shape}"
        )


if __name__ == "__main__":
    main()
