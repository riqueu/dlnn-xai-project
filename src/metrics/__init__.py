"""Metricas quantitativas de fidelidade, sanity checks, estabilidade de grafo e robustez."""

from __future__ import annotations

from src.metrics.faithfulness import (
    build_baseline_image,
    compute_deletion_insertion_auc,
    generate_random_saliency,
    rank_pixels_descending,
)
from src.metrics.graph_rewiring import (
    compute_edge_metrics,
    measure_edge_stability,
)
from src.metrics.robustness import (
    add_gaussian_noise,
    rgb_to_normalized_tensor,
)
from src.metrics.sanity import (
    CASCADING_STAGES,
    compare_maps,
    randomize_stage_inplace,
    run_cascading_randomization,
)

__all__ = [
    "build_baseline_image",
    "compute_deletion_insertion_auc",
    "generate_random_saliency",
    "rank_pixels_descending",
    "CASCADING_STAGES",
    "compare_maps",
    "randomize_stage_inplace",
    "run_cascading_randomization",
    "compute_edge_metrics",
    "measure_edge_stability",
    "add_gaussian_noise",
    "rgb_to_normalized_tensor",
]
