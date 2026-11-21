"""Explicadores agnosticos baseados em perturbacao: LIME e Partition SHAP."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import shap
import torch
from lime import lime_image
from scipy.ndimage import uniform_filter
from skimage.segmentation import slic

from src.xai.utils import make_batch_predict_fn, normalize_map

_BLUR_PATTERN = re.compile(r"^blur\((\d+)\s*,\s*(\d+)\)$")


def compute_lime(
    model: Any,
    image: np.ndarray,
    target_class: int | None = None,
    config: dict[str, Any] | None = None,
    device: torch.device | None = None,
    num_samples: int = 1000,
    n_segments: int = 50,
    compactness: float = 10.0,
    batch_size: int = 64,
    seed: int = 42,
    return_raw: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Calcula o mapa de atribuicao LIME sobre superpixels SLIC."""
    if config is None:
        raise ValueError("config obrigatorio para compute_lime.")
    model_device = next(model.parameters()).device if device is None else device

    img_rgb = np.asarray(image, dtype=np.float32)
    predict_fn = make_batch_predict_fn(
        model=model,
        config=config,
        device=model_device,
        return_probs=True,
        chunk_size=batch_size,
    )

    target = int(predict_fn(img_rgb[None, ...]).argmax(axis=1)[0]) if target_class is None else int(target_class)

    def _segment_fn(x: np.ndarray) -> np.ndarray:
        return slic(x, n_segments=n_segments, compactness=compactness, start_label=0)

    explainer = lime_image.LimeImageExplainer(random_state=seed)
    explanation = explainer.explain_instance(
        img_rgb.astype(np.float64),
        classifier_fn=predict_fn,
        labels=(target,),
        top_labels=None,
        hide_color=0,
        num_samples=num_samples,
        batch_size=batch_size,
        segmentation_fn=_segment_fn,
        random_seed=seed,
    )

    segments = explanation.segments
    raw_map = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, weight in explanation.local_exp[target]:
        raw_map[segments == seg_id] = float(weight)

    norm_map = normalize_map(raw_map)
    if return_raw:
        return norm_map, raw_map
    return norm_map


def _build_shap_background(image: np.ndarray, masker: str) -> np.ndarray:
    match = _BLUR_PATTERN.match(masker.strip())
    if match:
        kx, ky = int(match.group(1)), int(match.group(2))
        return uniform_filter(image.astype(np.float32), size=(kx, ky, 1), mode="reflect")
    if masker.strip() == "black":
        return np.zeros_like(image, dtype=np.float32)
    raise ValueError(f"Modo de masker nao suportado para Partition SHAP: {masker}")


def compute_partition_shap(
    model: Any,
    image: np.ndarray,
    target_class: int | None = None,
    config: dict[str, Any] | None = None,
    device: torch.device | None = None,
    max_evals: int = 500,
    batch_size: int = 32,
    masker: str = "blur(32,32)",
    seed: int = 42,
    return_raw: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Calcula valores de Shapley aproximados via Partition SHAP."""
    if config is None:
        raise ValueError("config obrigatorio para compute_partition_shap.")
    model_device = next(model.parameters()).device if device is None else device

    img_rgb = np.asarray(image, dtype=np.float32)
    predict_fn = make_batch_predict_fn(
        model=model,
        config=config,
        device=model_device,
        return_probs=False,
        chunk_size=batch_size,
    )

    target = int(predict_fn(img_rgb[None, ...]).argmax(axis=1)[0]) if target_class is None else int(target_class)

    bg = _build_shap_background(img_rgb, masker)
    img_masker = shap.maskers.Image(bg, shape=img_rgb.shape)
    explainer = shap.Explainer(
        predict_fn,
        img_masker,
        algorithm="partition",
        seed=seed,
    )
    sv = explainer(
        img_rgb[None, ...],
        max_evals=max_evals,
        batch_size=batch_size,
        outputs=[target],
        silent=True,
    )

    vals = sv.values[0, :, :, :, 0] if sv.values.ndim == 5 else sv.values[0]
    raw_map = vals.sum(axis=-1).astype(np.float32)
    norm_map = normalize_map(raw_map)

    if return_raw:
        return norm_map, raw_map
    return norm_map
