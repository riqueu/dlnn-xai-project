"""Analise de reconfiguracao de arestas k-NN na ViG sob Deletion."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from src.metrics.faithfulness import (
    _to_device_chw_01,
    build_baseline_image,
    rank_pixels_descending,
)


def compute_edge_metrics(
    orig_edges: torch.Tensor,
    pert_edges: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula retencao, taxa de rewiring e distancia de Jaccard por amostra."""
    orig_nn = orig_edges[0, 0] if orig_edges.ndim == 4 else orig_edges
    pert_nn = pert_edges[0] if pert_edges.ndim == 4 else pert_edges

    k = float(orig_nn.shape[-1])
    matches = (pert_nn[:, :, :, None] == orig_nn[None, :, None, :]).any(dim=-1).sum(dim=-1).float()

    retention = (matches / k).mean(dim=-1)
    rewiring = 1.0 - retention
    jaccard_dist = (1.0 - matches / (2.0 * k - matches).clamp_min(1.0)).mean(dim=-1)

    return (
        retention.detach().cpu().numpy().astype(np.float64),
        rewiring.detach().cpu().numpy().astype(np.float64),
        jaccard_dist.detach().cpu().numpy().astype(np.float64),
    )


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if float(np.std(x)) < 1.0e-12 or float(np.std(y)) < 1.0e-12:
        return 0.0, 0.0
    sp = float(spearmanr(x, y).statistic)
    pe = float(pearsonr(x, y).statistic)
    if np.isnan(sp):
        sp = 0.0
    if np.isnan(pe):
        pe = 0.0
    return sp, pe


def measure_edge_stability(
    model: Any,
    image: np.ndarray | torch.Tensor,
    saliency_map: np.ndarray,
    target_class: int | None = None,
    baseline: str = "blur",
    steps: int = 20,
    target_layer: int = -1,
    blur_kernel_size: int = 21,
    blur_sigma: float = 10.0,
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    seed: int = 42,
    capture_all_layers: bool = True,
) -> dict[str, Any]:
    """Mede a estabilidade das arestas k-NN ao longo da oclusao progressiva."""
    device = next(model.parameters()).device
    model.eval()

    img_01 = _to_device_chw_01(image, device)
    base_01 = build_baseline_image(
        img_01,
        baseline=baseline,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
    )

    _, height, width = img_01.shape
    num_pixels = height * width
    mean_t = torch.tensor(imagenet_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(imagenet_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    fractions = np.linspace(0.0, 1.0, int(steps) + 1, dtype=np.float64)
    counts = np.rint(fractions * num_pixels).astype(int)
    counts[0] = 0
    counts[-1] = num_pixels

    order_np = rank_pixels_descending(saliency_map, seed=seed)
    order_t = torch.from_numpy(order_np).to(device=device, dtype=torch.long)

    img_flat = img_01.reshape(3, num_pixels)
    base_flat = base_01.reshape(3, num_pixels)
    n_points = len(counts)

    batch_del = img_flat.unsqueeze(0).expand(n_points, -1, -1).clone()
    for b_idx, k in enumerate(counts):
        if k > 0:
            idx = order_t[: int(k)]
            batch_del[b_idx, :, idx] = base_flat[:, idx]

    batch_norm = (batch_del.reshape(n_points, 3, height, width) - mean_t) / std_t

    try:
        model.enable_graph_capture(True)
        with torch.no_grad():
            logits = model(batch_norm)
            if target_class is None:
                target = int(logits[0].argmax().item())
            else:
                target = int(target_class)
            probs = torch.softmax(logits, dim=1)[:, target].detach().cpu().numpy().astype(np.float64)

            target_edges = model.get_knn_edges(target_layer)
            orig_edges = target_edges[:, :1]
            retention, rewiring, jaccard_dist = compute_edge_metrics(orig_edges, target_edges)

            assert abs(float(retention[0]) - 1.0) < 1.0e-6, f"retention[0]={retention[0]}"
            assert abs(float(rewiring[0])) < 1.0e-6, f"rewiring[0]={rewiring[0]}"
            assert abs(float(jaccard_dist[0])) < 1.0e-6, f"jaccard[0]={jaccard_dist[0]}"

            layerwise: dict[int, dict[str, float]] = {}
            if capture_all_layers:
                step_20 = int(np.argmin(np.abs(fractions - 0.20)))
                step_50 = int(np.argmin(np.abs(fractions - 0.50)))
                n_blocks = len(model.model.backbone)
                for l_idx in range(n_blocks):
                    l_edges = model.get_knn_edges(l_idx)
                    l_orig = l_edges[:, :1]
                    l_subset = l_edges[:, [step_20, step_50]]
                    _, l_rew, l_jac = compute_edge_metrics(l_orig, l_subset)
                    layerwise[l_idx] = {
                        "rewiring_20": float(l_rew[0]),
                        "jaccard_20": float(l_jac[0]),
                        "rewiring_50": float(l_rew[1]),
                        "jaccard_50": float(l_jac[1]),
                    }
    finally:
        model.enable_graph_capture(False)

    prob_drop = probs[0] - probs
    deletion_auc = float(np.trapezoid(probs, fractions))
    rewiring_auc = float(np.trapezoid(rewiring, fractions))
    jaccard_auc = float(np.trapezoid(jaccard_dist, fractions))
    sp_corr, pe_corr = _safe_corr(rewiring, prob_drop)

    step_20 = int(np.argmin(np.abs(fractions - 0.20)))
    step_50 = int(np.argmin(np.abs(fractions - 0.50)))

    return {
        "target_class": target,
        "orig_prob": float(probs[0]),
        "final_prob": float(probs[-1]),
        "deletion_auc": deletion_auc,
        "rewiring_auc": rewiring_auc,
        "jaccard_auc": jaccard_auc,
        "rewiring_20": float(rewiring[step_20]),
        "rewiring_50": float(rewiring[step_50]),
        "jaccard_20": float(jaccard_dist[step_20]),
        "jaccard_50": float(jaccard_dist[step_50]),
        "spearman_rewiring_vs_probdrop": sp_corr,
        "pearson_rewiring_vs_probdrop": pe_corr,
        "fractions": fractions,
        "probs": probs,
        "prob_drop": prob_drop,
        "retention": retention,
        "rewiring": rewiring,
        "jaccard_dist": jaccard_dist,
        "layerwise": layerwise,
    }
