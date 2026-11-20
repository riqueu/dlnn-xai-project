"""Utilitarios de configuracao e reproducibilidade."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Carrega um arquivo YAML de configuracao experimental."""
    raise NotImplementedError("load_config ainda nao implementado.")


def set_seed(seed: int) -> None:
    """Define a semente global para reproducibilidade."""
    raise NotImplementedError("set_seed ainda nao implementado.")
