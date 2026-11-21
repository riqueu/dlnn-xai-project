"""Grad-CAM em nos (ViG-Ti) e tokens espaciais (DeiT-Tiny)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.xai.utils import normalize_map


def compute_gradcam(
    model: Any,
    image: torch.Tensor,
    target_class: int | None = None,
    return_raw: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Calcula o mapa de atribuicao Grad-CAM sobre a grade espacial 14x14."""
    device = next(model.parameters()).device
    x = image.to(device)
    if x.ndim == 3:
        x = x.unsqueeze(0)

    model.eval()
    model.zero_grad(set_to_none=True)

    try:
        model.enable_feature_capture(True)
        with torch.enable_grad():
            logits = model(x)
            if hasattr(model, "get_node_activations"):
                activations = model.get_node_activations()
            elif hasattr(model, "get_spatial_tokens"):
                activations = model.get_spatial_tokens()
            else:
                raise AttributeError("Modelo nao expoe get_node_activations nem get_spatial_tokens.")

            activations.retain_grad()
            target = int(logits.argmax(dim=1).item()) if target_class is None else int(target_class)
            logits[0, target].backward()

        grads = activations.grad
        if grads is None:
            raise RuntimeError("Gradientes das ativacoes espaciais nao foram capturados.")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam_up = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        raw_map = cam_up.detach().cpu().numpy().astype(np.float32)
        norm_map = normalize_map(raw_map)
    finally:
        model.enable_feature_capture(False)
        model.zero_grad(set_to_none=True)

    if return_raw:
        return norm_map, raw_map
    return norm_map
