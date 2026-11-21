"""Utilitarios de configuracao, resolucao de caminhos e reproducibilidade."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """Resolve caminhos relativos em relacao a raiz do projeto."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_config(path: str | Path) -> dict[str, Any]:
    """Carrega um arquivo YAML de configuracao experimental."""
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuracao invalida em {config_path}")
    return config


def set_seed(seed: int) -> None:
    """Define a semente global para reproducibilidade deterministica."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
