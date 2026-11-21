"""Integrated Gradients para atribuicao baseada em gradientes de entrada."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from captum.attr import IntegratedGradients

from src.xai.utils import get_predicted_class, normalize_map


def compute_integrated_gradients(
    model: Any,
    image: torch.Tensor,
    target_class: int | None = None,
    n_steps: int = 50,
    baseline: str = "black",
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    internal_batch_size: int = 16,
    return_raw: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Calcula atribuicoes por Integrated Gradients a partir de um baseline."""
    device = next(model.parameters()).device
    x = image.to(device).detach().clone()
    if x.ndim == 3:
        x = x.unsqueeze(0)

    model.eval()
    model.zero_grad(set_to_none=True)

    target = get_predicted_class(model, x)[0] if target_class is None else int(target_class)

    if baseline == "black":
        mean_t = torch.tensor(imagenet_mean, device=device, dtype=x.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(imagenet_std, device=device, dtype=x.dtype).view(1, 3, 1, 1)
        baseline_tensor = (torch.zeros_like(x) - mean_t) / std_t
    elif baseline == "zero":
        baseline_tensor = torch.zeros_like(x)
    else:
        raise ValueError(f"Baseline invalido para Integrated Gradients: {baseline}")

    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        inputs=x,
        baselines=baseline_tensor,
        target=target,
        n_steps=n_steps,
        method="riemann_right",
        internal_batch_size=internal_batch_size,
    )

    raw_map = attributions[0].abs().sum(dim=0).detach().cpu().numpy().astype(np.float32)
    norm_map = normalize_map(raw_map)
    model.zero_grad(set_to_none=True)

    if return_raw:
        return norm_map, raw_map
    return norm_map
