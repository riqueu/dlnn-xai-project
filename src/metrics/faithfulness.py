"""Metricas de fidelidade via curvas Deletion e Insertion AUC."""

from __future__ import annotations

from typing import Any


def compute_deletion_insertion_auc(
    model: Any,
    image: Any,
    saliency_map: Any,
    baseline: str = "blur",
    steps: int = 50,
) -> dict[str, float]:
    """Calcula os valores de Deletion AUC e Insertion AUC para um mapa."""
    raise NotImplementedError("compute_deletion_insertion_auc ainda nao implementado.")


def generate_random_saliency(shape: tuple[int, int], seed: int) -> Any:
    """Gera um mapa de saliencia aleatorio como controle negativo."""
    raise NotImplementedError("generate_random_saliency ainda nao implementado.")
