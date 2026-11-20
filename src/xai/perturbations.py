"""Explicadores agnosticos baseados em perturbação: LIME e Partition SHAP."""

from __future__ import annotations

from typing import Any


def compute_lime(
    model: Any,
    image: Any,
    target_class: int | None = None,
    **kwargs: Any,
) -> Any:
    """Calcula o mapa de atribuicao LIME sobre superpixels SLIC."""
    raise NotImplementedError("compute_lime ainda nao implementado.")


def compute_partition_shap(
    model: Any,
    image: Any,
    target_class: int | None = None,
    **kwargs: Any,
) -> Any:
    """Calcula valores de Shapley aproximados via Partition SHAP."""
    raise NotImplementedError("compute_partition_shap ainda nao implementado.")
