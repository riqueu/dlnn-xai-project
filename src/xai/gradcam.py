"""Grad-CAM em nos (ViG-Ti) e tokens espaciais (DeiT-Tiny)."""

from __future__ import annotations

from typing import Any


def compute_gradcam(
    model: Any,
    image: Any,
    target_class: int | None = None,
) -> Any:
    """Calcula o mapa de atribuicao Grad-CAM sobre a grade espacial 14x14."""
    raise NotImplementedError("compute_gradcam ainda nao implementado.")
