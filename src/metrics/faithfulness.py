"""Metricas de fidelidade via curvas Deletion e Insertion AUC."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF


def generate_random_saliency(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Gera um mapa de saliencia aleatorio uniforme em [0, 1] como controle negativo."""
    rng = np.random.default_rng(seed)
    return rng.random(shape, dtype=np.float32)


def build_baseline_image(
    image_01: torch.Tensor,
    baseline: str,
    blur_kernel_size: int = 21,
    blur_sigma: float = 10.0,
) -> torch.Tensor:
    """Constroi a imagem baseline (preta ou desfoque gaussiano) em [0, 1] com shape (3, H, W)."""
    if baseline == "black":
        return torch.zeros_like(image_01)
    if baseline == "blur":
        return TF.gaussian_blur(
            image_01,
            kernel_size=[int(blur_kernel_size), int(blur_kernel_size)],
            sigma=[float(blur_sigma), float(blur_sigma)],
        )
    raise ValueError(f"Baseline desconhecido para Deletion/Insertion: {baseline}")


def rank_pixels_descending(saliency_map: np.ndarray, seed: int) -> np.ndarray:
    """Ordena os indices de pixels em ordem decrescente com desempate aleatorio deterministico."""
    flat = np.asarray(saliency_map, dtype=np.float64).ravel()
    if not np.isfinite(flat).all():
        raise ValueError("Mapa de saliencia contem valores nao finitos.")
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(flat.size)
    return np.lexsort((tie_breaker, -flat))


def _to_device_chw_01(image: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Esperado array (H, W, 3), recebido {arr.shape}")
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().float()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"Esperado tensor (3, H, W), recebido {tuple(tensor.shape)}")
    else:
        raise TypeError(f"Tipo de imagem nao suportado: {type(image)}")

    return tensor.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)


def compute_deletion_insertion_auc(
    model: nn.Module,
    image: np.ndarray | torch.Tensor,
    saliency_map: np.ndarray,
    target_class: int | None = None,
    baseline: str = "blur",
    steps: int = 50,
    blur_kernel_size: int = 21,
    blur_sigma: float = 10.0,
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    batch_size: int = 64,
    seed: int = 42,
    return_curves: bool = False,
) -> dict[str, Any]:
    """Calcula os valores de Deletion AUC e Insertion AUC para um mapa."""
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

    with torch.no_grad():
        if target_class is None:
            norm_img = (img_01.unsqueeze(0) - mean_t) / std_t
            target = int(model(norm_img).argmax(dim=1).item())
        else:
            target = int(target_class)

        fractions = np.linspace(0.0, 1.0, int(steps) + 1, dtype=np.float64)
        counts = np.rint(fractions * num_pixels).astype(int)
        counts[0] = 0
        counts[-1] = num_pixels

        order_np = rank_pixels_descending(saliency_map, seed=seed)
        order_t = torch.from_numpy(order_np).to(device=device, dtype=torch.long)

        img_flat = img_01.reshape(3, num_pixels)
        base_flat = base_01.reshape(3, num_pixels)

        del_probs_list: list[float] = []
        ins_probs_list: list[float] = []

        n_points = len(counts)
        for start in range(0, n_points, batch_size):
            chunk_counts = counts[start : start + batch_size]
            bs = len(chunk_counts)

            batch_del = img_flat.unsqueeze(0).expand(bs, -1, -1).clone()
            batch_ins = base_flat.unsqueeze(0).expand(bs, -1, -1).clone()

            for b_idx, k in enumerate(chunk_counts):
                if k > 0:
                    idx = order_t[: int(k)]
                    batch_del[b_idx, :, idx] = base_flat[:, idx]
                    batch_ins[b_idx, :, idx] = img_flat[:, idx]

            both = torch.cat(
                [
                    batch_del.reshape(bs, 3, height, width),
                    batch_ins.reshape(bs, 3, height, width),
                ],
                dim=0,
            )
            both_norm = (both - mean_t) / std_t
            logits = model(both_norm)
            probs = torch.softmax(logits, dim=1)[:, target]

            del_probs_list.extend(probs[:bs].detach().cpu().tolist())
            ins_probs_list.extend(probs[bs:].detach().cpu().tolist())

    del_probs = np.asarray(del_probs_list, dtype=np.float64)
    ins_probs = np.asarray(ins_probs_list, dtype=np.float64)

    assert abs(float(del_probs[0]) - float(ins_probs[-1])) < 1.0e-4, (
        f"Inconsistencia de contorno del[0]={del_probs[0]} vs ins[-1]={ins_probs[-1]}"
    )
    assert abs(float(del_probs[-1]) - float(ins_probs[0])) < 1.0e-4, (
        f"Inconsistencia de contorno del[-1]={del_probs[-1]} vs ins[0]={ins_probs[0]}"
    )

    deletion_auc = float(np.trapezoid(del_probs, fractions))
    insertion_auc = float(np.trapezoid(ins_probs, fractions))

    result: dict[str, Any] = {
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "auc_diff": insertion_auc - deletion_auc,
        "orig_prob": float(del_probs[0]),
        "base_prob": float(del_probs[-1]),
        "target_class": target,
    }
    if return_curves:
        result["fractions"] = fractions
        result["deletion_curve"] = del_probs
        result["insertion_curve"] = ins_probs
    return result
