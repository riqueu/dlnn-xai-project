"""Integrated Gradients para atribuicao baseada em gradientes de entrada."""

from __future__ import annotations

from typing import Any


def compute_integrated_gradients(
    model: Any,
    image: Any,
    target_class: int | None = None,
    n_steps: int = 50,
) -> Any:
    """Calcula atribuicoes por Integrated Gradients a partir de um baseline."""
    raise NotImplementedError("compute_integrated_gradients ainda nao implementado.")
