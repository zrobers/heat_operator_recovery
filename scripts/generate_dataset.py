#!/usr/bin/env python3
"""Generate synthetic heat-equation trajectories and save train/val tensors."""

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
from src.pde.heat_solver import grid_spacing, simulate_heat_equation, stable_dt
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(int(cfg.get("seed", 42)))

    H = W = int(cfg["grid_size"])
    num_steps = int(cfg["num_pde_steps"])
    kappa = float(cfg["kappa"])
    dx = grid_spacing(H, W)
    dt = cfg.get("dt")
    if dt is None:
        dt = stable_dt(dx, kappa)
    else:
        dt = float(dt)

    boundary_mode = str(cfg.get("boundary_mode", "fourier"))
    n_train = int(cfg["train_samples"])
    n_val = int(cfg["val_samples"])

    out_dir = ROOT / "outputs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_split(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        boundaries = []
        trajs = []
        for _ in tqdm(range(n), desc=f"generating ({n} samples)"):
            b = sample_boundary(H, W, mode=boundary_mode, generator=gen)
            traj = simulate_heat_equation(b, num_steps=num_steps, dt=dt, kappa=kappa)
            boundaries.append(b)
            trajs.append(traj)
        b_tensor = torch.stack(boundaries, dim=0)
        t_tensor = torch.stack(trajs, dim=0)
        return b_tensor, t_tensor

    g_train = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    g_val = torch.Generator().manual_seed(int(cfg.get("seed", 42)) + 1)

    train_b, train_t = make_split(n_train, g_train)
    val_b, val_t = make_split(n_val, g_val)

    meta = {
        "grid_size": H,
        "num_pde_steps": num_steps,
        "kappa": kappa,
        "dt": dt,
        "boundary_mode": boundary_mode,
    }

    torch.save({"boundaries": train_b, "trajectories": train_t, "meta": meta}, out_dir / "train.pt")
    torch.save({"boundaries": val_b, "trajectories": val_t, "meta": meta}, out_dir / "val.pt")
    print(f"Wrote {out_dir / 'train.pt'} and {out_dir / 'val.pt'}")


if __name__ == "__main__":
    main()
