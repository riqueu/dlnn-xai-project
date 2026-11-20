"""Download, particionamento estratificado e dataloaders do Imagenette."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def download_imagenette(url: str, raw_dir: str | Path) -> Path:
    """Baixa e extrai o conjunto Imagenette para o diretorio especificado."""
    raise NotImplementedError("download_imagenette ainda nao implementado.")


def create_splits(
    raw_dir: str | Path,
    splits_dir: str | Path,
    val_ratio: float,
    seed: int,
) -> None:
    """Gera os arquivos de split estratificado treino, validacao e teste."""
    raise NotImplementedError("create_splits ainda nao implementado.")


def create_xai_subset(
    test_csv: str | Path,
    output_csv: str | Path,
    per_class: int,
    seed: int,
) -> None:
    """Seleciona a amostra fixa estratificada para avaliacao de XAI."""
    raise NotImplementedError("create_xai_subset ainda nao implementado.")


def build_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    """Constroi os dataloaders de treino, validacao e teste."""
    raise NotImplementedError("build_dataloaders ainda nao implementado.")
