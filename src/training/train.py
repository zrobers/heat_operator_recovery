"""Train conditional denoising model with trajectory-consistency loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import HeatTrajectoryDataset
from src.models.diffusion import GaussianDiffusion
from src.models.unet import SmallUNet
from src.training.losses import total_loss
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cfg_path = root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.get("seed", 42)))

    data_path = root / "outputs" / "data" / "train.pt"
    pack = torch.load(data_path, map_location="cpu")
    boundaries = pack["boundaries"]
    trajectories = pack["trajectories"]
    ds = HeatTrajectoryDataset(boundaries, trajectories)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
    )

    num_diff = int(cfg["num_diffusion_steps"])
    diffusion = GaussianDiffusion(num_timesteps=num_diff).to(device)
    model = SmallUNet(in_ch=2, base=32, time_dim=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))

    lambda_traj = float(cfg["lambda_traj"])
    k_steps = int(cfg["num_traj_loss_steps"])
    t_traj_k = torch.linspace(0, num_diff - 1, steps=k_steps).long()

    epochs = int(cfg["num_epochs"])
    ckpt_dir = root / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{epochs}")
        running = {"tot": 0.0, "base": 0.0, "traj": 0.0}
        n = 0
        for batch in pbar:
            boundary = batch["boundary"].to(device)
            trajectory = batch["trajectory"].to(device)
            bsz = trajectory.shape[0]
            t_diff = torch.randint(0, num_diff, (bsz,), device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            l_tot, l_base, l_traj = total_loss(
                model,
                diffusion,
                trajectory,
                boundary,
                t_diff,
                t_traj_k.to(device),
                lambda_traj=lambda_traj,
            )
            l_tot.backward()
            opt.step()

            running["tot"] += float(l_tot.item())
            running["base"] += float(l_base.item())
            running["traj"] += float(l_traj.item())
            n += 1
            pbar.set_postfix(
                loss=f"{running['tot']/n:.5f}",
                base=f"{running['base']/n:.5f}",
                traj=f"{running['traj']/n:.5f}",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "diffusion": {"num_timesteps": diffusion.num_timesteps},
            "config": dict(cfg),
        },
        ckpt_dir / "model.pt",
    )
    print(f"Saved checkpoint to {ckpt_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
