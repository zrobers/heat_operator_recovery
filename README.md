# Mechanism-aligned diffusion (Phase I): 2D heat equation

Minimal end-to-end pipeline: synthetic **2D heat equation** data, a **conditional DDPM** that denoises interior fields given **Dirichlet boundaries**, and a **trajectory-consistency loss** that maps diffusion timesteps to physical PDE times and matches Tweedie estimates to the ground-truth trajectory.

## Layout

```
.
├── src/
│   ├── data/           # boundary sampling, PyTorch Dataset
│   ├── models/         # SmallUNet, GaussianDiffusion
│   ├── training/       # losses, train loop
│   ├── utils/          # YAML config loader
│   └── pde/            # explicit heat solver
├── scripts/            # generate_dataset.py, recompute_trajectories.py, evaluate.py
├── notebooks/          # explore_data.ipynb, Colab: colab_trajectory_aligned.ipynb, colab_standard_diffusion.ipynb
├── configs/            # default.yaml
├── outputs/            # data/, checkpoints/, eval/ (created at runtime)
├── requirements.txt
└── README.md
```

## Setup

```bash
cd /path/to/heat_operator_recovery
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Commands

1. **Generate synthetic dataset** (writes `outputs/data/train.pt` and `outputs/data/test.pt` — random **80% / 20%** train/test split by default):

```bash
python scripts/generate_dataset.py
```

   Set `grid_size` to the resolution you want on disk (e.g. 32) and `solve_grid_size` to a finer grid (e.g. 96) to integrate the heat equation at higher spatial resolution with stable substeps, then **downsample** stored **trajectories** (and boundaries) to `grid_size`. The **final** field `u(T)` is still saved at **solve** resolution in each `.pt` file as `u_final_solve` (`[N, H_solve, W_solve]`), while `trajectories` stay `[N, T, H_store, W_store]`.

2. **Recompute trajectories only** (keeps existing `boundaries`, refreshes `trajectories` using current `solve_grid_size` / κ / steps):

```bash
python scripts/recompute_trajectories.py --config configs/default.yaml
```

3. **Train** (reads `outputs/data/train.pt`, saves `outputs/checkpoints/model.pt`):

```bash
python -m src.training.train
```

4. **Evaluate** on the **held-out test** split (metrics + figures under `outputs/eval/`):

```bash
python scripts/evaluate.py --split test
```

Optional config path:

```bash
python scripts/generate_dataset.py --config configs/default.yaml
python -m src.training.train --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --split test
```

## Solver self-test

```bash
python src/pde/heat_solver.py
```

## Google Colab

Upload or sync the repo to Drive, open:

- **`notebooks/colab_trajectory_aligned.ipynb`** — trajectory-consistency training; saves `outputs/checkpoints/model_trajectory_aligned.pt`; evaluation uses **final-field MSE** on `test.pt`.
- **`notebooks/colab_standard_diffusion.ipynb`** — standard DDPM on the **terminal field** only; saves `outputs/checkpoints/model_standard_final.pt`; same **final-field** test metric.

Set the `REPO` path in each notebook after mounting Drive. Use a GPU runtime (e.g. A100).

## Configuration

Edit `configs/default.yaml` for stored `grid_size`, optional finer `solve_grid_size`, PDE steps, diffusivity κ, diffusion horizon, `lambda_traj`, batch size, learning rate, and `train_samples` / `test_samples` (e.g. 4000 + 1000 = 5000 total).

CLI training variants:

```bash
python -m src.training.train --config configs/default.yaml --output model_trajectory_aligned.pt
python -m src.training.train_standard --config configs/default.yaml --output model_standard_final.pt
```
