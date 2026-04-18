"""Standard conditional DDPM: denoise only the *final* field u(·, T) given the boundary (no trajectory loss)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import HeatTrajectoryDataset
from src.models.diffusion import GaussianDiffusion
from src.models.unet import SmallUNet
from src.training.losses import base_noise_loss
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--output",
        type=str,
        default="model_standard_final.pt",
        help="Checkpoint filename under outputs/checkpoints/",
    )
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
    u_final = pack.get("u_final_solve")
    ds = HeatTrajectoryDataset(boundaries, trajectories, u_final_solve=u_final)
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

    epochs = int(cfg["num_epochs"])
    ckpt_dir = root / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / args.output

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"[standard/final-only] epoch {epoch+1}/{epochs}")
        running = 0.0
        n = 0
        for batch in pbar:
            boundary = batch["boundary"].to(device)
            trajectory = batch["trajectory"].to(device)
            bsz = trajectory.shape[0]
            x0 = trajectory[:, -1].unsqueeze(1)
            t_diff = torch.randint(0, num_diff, (bsz,), device=device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            loss = base_noise_loss(model, diffusion, x0, boundary, t_diff)
            loss.backward()
            opt.step()

            running += float(loss.item())
            n += 1
            pbar.set_postfix(loss=f"{running/n:.5f}")

    torch.save(
        {
            "model": model.state_dict(),
            "diffusion": {"num_timesteps": diffusion.num_timesteps},
            "config": dict(cfg),
            "training_mode": "standard_final_field",
        },
        out_path,
    )
    print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
