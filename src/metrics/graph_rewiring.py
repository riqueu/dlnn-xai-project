"""Analise de reconfiguracao de arestas k-NN na ViG sob Deletion."""

from __future__ import annotations

from typing import Any


def measure_edge_stability(
    model: Any,
    image: Any,
    saliency_map: Any,
    baseline: str = "blur",
    steps: int = 20,
) -> list[dict[str, float]]:
    """Mede a estabilidade das arestas k-NN ao longo da oclusao progressiva."""
    raise NotImplementedError("measure_edge_stability ainda nao implementado.")
