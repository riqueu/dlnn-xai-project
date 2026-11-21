"""Attention Rollout para agregacao de pesos de self-attention em ViT."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.xai.utils import normalize_map


def _fuse_attention_heads(attn: torch.Tensor, head_fusion: str) -> torch.Tensor:
    if head_fusion == "mean":
        return attn.mean(dim=0)
    if head_fusion == "max":
        return attn.max(dim=0).values
    if head_fusion == "min":
        return attn.min(dim=0).values
    raise ValueError(f"Modo head_fusion invalido: {head_fusion}")


def compute_attention_rollout(
    model: Any,
    image: torch.Tensor,
    discard_ratio: float = 0.0,
    head_fusion: str = "mean",
    return_raw: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Calcula o mapa de Attention Rollout entre os blocos do DeiT-Tiny."""
    device = next(model.parameters()).device
    x = image.to(device)
    if x.ndim == 3:
        x = x.unsqueeze(0)

    model.eval()
    try:
        model.enable_attention_capture(True)
        with torch.no_grad():
            _ = model(x)
            attentions = model.get_attention_maps()
    finally:
        model.enable_attention_capture(False)

    n_tokens = attentions[0].shape[-1]
    eye = torch.eye(n_tokens, device=attentions[0].device, dtype=torch.float32)
    result = eye.clone()

    for attn in attentions:
        attn_fused = _fuse_attention_heads(attn[0].float(), head_fusion)
        if discard_ratio > 0.0:
            flat = attn_fused[:, 1:]
            num_discard = int(flat.shape[-1] * discard_ratio)
            if num_discard > 0:
                _, idx_low = torch.topk(flat, k=num_discard, dim=-1, largest=False)
                flat = flat.clone()
                flat.scatter_(1, idx_low, 0.0)
                attn_fused = torch.cat([attn_fused[:, :1], flat], dim=-1)

        a = 0.5 * attn_fused + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        result = a @ result

    prefix = getattr(model.model, "num_prefix_tokens", 1)
    patch_mask = result[0, prefix:]
    grid_size = math.isqrt(patch_mask.numel())
    if grid_size * grid_size != patch_mask.numel():
        raise ValueError(f"Numero de tokens espaciais invalido: {patch_mask.numel()}")

    mask_2d = patch_mask.reshape(1, 1, grid_size, grid_size)
    mask_up = F.interpolate(mask_2d, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    raw_map = mask_up.detach().cpu().numpy().astype(np.float32)
    norm_map = normalize_map(raw_map)

    if return_raw:
        return norm_map, raw_map
    return norm_map
