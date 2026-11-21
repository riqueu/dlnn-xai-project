"""Metodos de explicabilidade post-hoc nativos e agnosticos."""

from __future__ import annotations

from src.xai.gradcam import compute_gradcam
from src.xai.ig import compute_integrated_gradients
from src.xai.perturbations import compute_lime, compute_partition_shap
from src.xai.rollout import compute_attention_rollout
from src.xai.utils import (
    get_predicted_class,
    load_image_pair,
    make_batch_predict_fn,
    normalize_map,
)

__all__ = [
    "compute_gradcam",
    "compute_attention_rollout",
    "compute_lime",
    "compute_partition_shap",
    "compute_integrated_gradients",
    "normalize_map",
    "load_image_pair",
    "get_predicted_class",
    "make_batch_predict_fn",
]
