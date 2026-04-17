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
├── scripts/            # generate_dataset.py, evaluate.py
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

1. **Generate synthetic dataset** (writes `outputs/data/train.pt` and `outputs/data/val.pt`):

```bash
python scripts/generate_dataset.py
```

2. **Train** (reads `outputs/data/train.pt`, saves `outputs/checkpoints/model.pt`):

```bash
python -m src.training.train
```

3. **Evaluate** (metrics + figures under `outputs/eval/`):

```bash
python scripts/evaluate.py
```

Optional config path:

```bash
python scripts/generate_dataset.py --config configs/default.yaml
python -m src.training.train --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml
```

## Solver self-test

```bash
python src/pde/heat_solver.py
```

## Configuration

Edit `configs/default.yaml` for grid size, PDE steps, diffusivity κ, diffusion horizon, `lambda_traj`, batch size, learning rate, and dataset sizes.
