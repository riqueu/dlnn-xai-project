"""Metricas quantitativas de fidelidade, sanity checks e estabilidade de grafo."""

from __future__ import annotations

from src.metrics.faithfulness import (
    build_baseline_image,
    compute_deletion_insertion_auc,
    generate_random_saliency,
    rank_pixels_descending,
)

__all__ = [
    "build_baseline_image",
    "compute_deletion_insertion_auc",
    "generate_random_saliency",
    "rank_pixels_descending",
]
