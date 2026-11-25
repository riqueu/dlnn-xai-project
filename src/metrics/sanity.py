"""Sanity checks de aleatorizacao progressiva de parametros."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

from src.config import set_seed

CASCADING_STAGES: list[str] = [
    "original",
    "head",
    "blocks_9_11",
    "blocks_6_11",
    "blocks_3_11",
    "blocks_0_11",
    "all_layers",
]


def compare_maps(map_a: np.ndarray, map_b: np.ndarray) -> dict[str, float | bool]:
    """Calcula correlacao de postos de Spearman e SSIM entre dois mapas."""
    a = np.asarray(map_a, dtype=np.float32)
    b = np.asarray(map_b, dtype=np.float32)

    ssim = float(structural_similarity(a, b, data_range=1.0))
    is_constant = bool(float(a.std()) < 1.0e-12 or float(b.std()) < 1.0e-12)

    if is_constant:
        rho = 0.0
    else:
        rho = float(spearmanr(a.ravel(), b.ravel()).statistic)
        if np.isnan(rho):
            rho = 0.0
            is_constant = True

    return {
        "spearman": rho,
        "abs_spearman": abs(rho),
        "ssim": ssim,
        "is_constant_map": is_constant,
    }


def _reset_module_weights(module: nn.Module) -> None:
    for sub in module.modules():
        if hasattr(sub, "reset_parameters"):
            sub.reset_parameters()


def randomize_stage_inplace(
    wrapper: nn.Module,
    model_key: str,
    stage_name: str,
    seed: int,
) -> None:
    """Reinicializa in-place os pesos correspondentes a um estagio da cascata."""
    if stage_name == "original":
        return

    set_seed(seed)
    backbone = wrapper.model

    if model_key == "vig_ti":
        if stage_name == "head":
            _reset_module_weights(backbone.prediction)
        elif stage_name == "blocks_9_11":
            _reset_module_weights(backbone.backbone[9:12])
        elif stage_name == "blocks_6_11":
            _reset_module_weights(backbone.backbone[6:9])
        elif stage_name == "blocks_3_11":
            _reset_module_weights(backbone.backbone[3:6])
        elif stage_name == "blocks_0_11":
            _reset_module_weights(backbone.backbone[0:3])
        elif stage_name == "all_layers":
            _reset_module_weights(backbone.stem)
            with torch.no_grad():
                nn.init.trunc_normal_(backbone.pos_embed, std=0.02)
        else:
            raise ValueError(f"Estagio invalido para ViG-Ti: {stage_name}")
    elif model_key == "deit_tiny":
        if stage_name == "head":
            _reset_module_weights(backbone.head)
            _reset_module_weights(backbone.norm)
        elif stage_name == "blocks_9_11":
            _reset_module_weights(backbone.blocks[9:12])
        elif stage_name == "blocks_6_11":
            _reset_module_weights(backbone.blocks[6:9])
        elif stage_name == "blocks_3_11":
            _reset_module_weights(backbone.blocks[3:6])
        elif stage_name == "blocks_0_11":
            _reset_module_weights(backbone.blocks[0:3])
        elif stage_name == "all_layers":
            _reset_module_weights(backbone.patch_embed)
            with torch.no_grad():
                nn.init.trunc_normal_(backbone.pos_embed, std=0.02)
                nn.init.trunc_normal_(backbone.cls_token, std=0.02)
        else:
            raise ValueError(f"Estagio invalido para DeiT-Tiny: {stage_name}")
    else:
        raise ValueError(f"Modelo desconhecido: {model_key}")

    wrapper.eval()


def run_cascading_randomization(
    model_loader_fn: Callable[[], nn.Module],
    model_key: str,
    explain_fn: Callable[[nn.Module], np.ndarray],
    original_map: np.ndarray,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Executa aleatorizacao em cascata do cabecalho ate a entrada em um exemplo."""
    results: list[dict[str, Any]] = [
        {
            "stage_idx": 0,
            "stage_name": "original",
            "spearman": 1.0,
            "abs_spearman": 1.0,
            "ssim": 1.0,
            "is_constant_map": False,
        }
    ]
    wrapper = model_loader_fn().eval()
    for stage_idx, stage_name in enumerate(CASCADING_STAGES[1:], start=1):
        randomize_stage_inplace(wrapper, model_key, stage_name, seed=seed + 1000 * stage_idx)
        rand_map = explain_fn(wrapper)
        metrics = compare_maps(original_map, rand_map)
        results.append(
            {
                "stage_idx": stage_idx,
                "stage_name": stage_name,
                **metrics,
            }
        )
    return results
