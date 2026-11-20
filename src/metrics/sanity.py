"""Sanity checks de aleatorizacao progressiva de parametros."""

from __future__ import annotations

from typing import Any


def run_cascading_randomization(
    model: Any,
    explain_fn: Any,
    image: Any,
) -> list[dict[str, Any]]:
    """Executa aleatorizacao em cascata do cabecalho ate a entrada."""
    raise NotImplementedError("run_cascading_randomization ainda nao implementado.")


def compare_maps(map_a: Any, map_b: Any) -> dict[str, float]:
    """Calcula correlacao de postos de Spearman e SSIM entre dois mapas."""
    raise NotImplementedError("compare_maps ainda nao implementado.")
