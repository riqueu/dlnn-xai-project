"""Attention Rollout para agregacao de pesos de self-attention em ViT."""

from __future__ import annotations

from typing import Any


def compute_attention_rollout(
    model: Any,
    image: Any,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
) -> Any:
    """Calcula o mapa de Attention Rollout entre os blocos do DeiT-Tiny."""
    raise NotImplementedError("compute_attention_rollout ainda nao implementado.")
