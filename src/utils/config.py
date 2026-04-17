"""Load YAML config with optional overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        return yaml.safe_load(f)
