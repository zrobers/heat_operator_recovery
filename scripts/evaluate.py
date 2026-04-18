#!/usr/bin/env python3
"""Evaluate denoising quality and trajectory alignment; save diagnostic plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.diffusion import GaussianDiffusion
from src.models.unet import SmallUNet
from src.training.losses import trajectory_consistency_loss
from src.utils.config import load_config


@torch.no_grad()
def final_state_mse(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    trajectory: torch.Tensor,
    boundary: torch.Tensor,
    device: torch.device,
) -> float:
    """One-step denoising accuracy on the last PDE frame (largest time index)."""
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
def trajectory_mse_metric(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    trajectory: torch.Tensor,
    boundary: torch.Tensor,
    t_k: torch.Tensor,
    device: torch.device,
) -> float:
    """Average Tweedie error at mapped physical times (same objective as training traj loss)."""
    model.eval()
    trajectory = trajectory.to(device)
    boundary = boundary.to(device)
    t_k = t_k.to(device)
    return float(trajectory_consistency_loss(model, diffusion, trajectory, boundary, t_k).item())


def plot_grids(
    tensors: list[torch.Tensor],
    titles: list[str],
    path: Path,
    ncols: int = 4,
) -> None:
    """Each tensor [H, W] or [1,H,W]."""
    n = len(tensors)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes = axes.flatten() if n > 1 else [axes]
    for i in range(nrows * ncols):
        ax = axes[i]
        if i < n:
            u = tensors[i]
            if u.dim() == 3:
                u = u[0]
            im = ax.imshow(u.cpu().numpy(), origin="lower", cmap="viridis")
            ax.set_title(titles[i])
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument(
        "--split",
        type=str,
        choices=("test", "train", "val"),
        default="test",
        help="Which split to evaluate (default: test = held-out 20%%).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = ROOT / "outputs" / "data" / f"{args.split}.pt"
    if not data_path.is_file() and args.split == "test":
        legacy = ROOT / "outputs" / "data" / "val.pt"
        if legacy.is_file():
            print(f"Note: {data_path} missing; using legacy {legacy}")
            data_path = legacy
    pack = torch.load(data_path, map_location=device)
    boundaries = pack["boundaries"]
    trajectories = pack["trajectories"]

    ckpt_path = ROOT / "outputs" / "checkpoints" / "model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    num_diff = int(ckpt["diffusion"]["num_timesteps"])

    diffusion = GaussianDiffusion(num_timesteps=num_diff).to(device)
    model = SmallUNet(in_ch=2, base=32, time_dim=128).to(device)
    model.load_state_dict(ckpt["model"])

    k_steps = int(cfg["num_traj_loss_steps"])
    t_k = torch.linspace(0, num_diff - 1, steps=k_steps).long()

    boundary_batch = boundaries.unsqueeze(1).to(device)
    traj_batch = trajectories.to(device)
    fs_mse = final_state_mse(model, diffusion, traj_batch, boundary_batch, device)
    tr_mse = trajectory_mse_metric(model, diffusion, traj_batch, boundary_batch, t_k, device)

    out_dir = ROOT / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"final_state_mse: {fs_mse}\n")
        f.write(f"trajectory_mse: {tr_mse}\n")
    print(f"final_state_mse: {fs_mse}")
    print(f"trajectory_mse: {tr_mse}")

    # Plots for first sample in this split
    idx = 0
    boundary = boundaries[idx]
    traj = trajectories[idx]
    boundary_b = boundary.unsqueeze(0).unsqueeze(0).to(device)
    traj_d = traj.unsqueeze(0).to(device)

    # True trajectory snapshots
    t_pde = traj.shape[0]
    snaps = [0, t_pde // 4, t_pde // 2, t_pde - 1]
    true_frames = [traj[s] for s in snaps]
    plot_grids(
        [boundary] + true_frames,
        ["boundary"] + [f"true u(t), t={s}" for s in snaps],
        out_dir / "true_trajectory.png",
    )

    # Predicted "trajectory": Tweedie x̂_0 at mapped diffusion times for each physical snapshot
    model.eval()
    pred_frames: list[torch.Tensor] = []
    for s in snaps:
        tt_val = int(round(s / max(t_pde - 1, 1) * (num_diff - 1)))
        tt = torch.tensor([tt_val], device=device, dtype=torch.long)
        u_true = traj_d[:, s].unsqueeze(1)
        noise = diffusion.sample_noise(u_true)
        x_t = diffusion.add_noise(u_true, tt.expand(1), noise=noise)
        eps_pred = model(x_t, boundary_b, tt.expand(1))
        x0_hat = diffusion.predict_x0_from_noise(x_t, tt.expand(1), eps_pred)
        pred_frames.append(x0_hat[0, 0].detach().cpu())

    plot_grids(
        [boundary.cpu()] + pred_frames,
        ["boundary"] + [f"pred x̂0 @ map(t), snap s={s}" for s in snaps],
        out_dir / "pred_trajectory.png",
    )

    u_final = traj[-1]
    tt = torch.tensor([num_diff // 2], device=device, dtype=torch.long)
    uf = u_final.unsqueeze(0).unsqueeze(0).to(device)
    noise = diffusion.sample_noise(uf)
    x_t = diffusion.add_noise(uf, tt.expand(1), noise=noise)
    eps_pred = model(x_t, boundary_b, tt.expand(1))
    pred_final = diffusion.predict_x0_from_noise(x_t, tt.expand(1), eps_pred)[0, 0].cpu()

    plot_grids(
        [u_final, pred_final, (u_final - pred_final).abs()],
        ["GT final", "pred final (Tweedie)", "|error|"],
        out_dir / "final_compare.png",
    )


if __name__ == "__main__":
    main()
