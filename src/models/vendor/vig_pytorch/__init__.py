"""Implementacao oficial da Vision GNN (ViG) adaptada para PyTorch/timm modernos."""

from __future__ import annotations

from .vig import DeepGCN, vig_b_224_gelu, vig_s_224_gelu, vig_ti_224_gelu

__all__ = [
    "DeepGCN",
    "vig_ti_224_gelu",
    "vig_s_224_gelu",
    "vig_b_224_gelu",
]
