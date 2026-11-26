"""Metricas de robustez dos mapas de explicabilidade a ruido gaussiano de entrada (E6)."""

from __future__ import annotations

import numpy as np
import torch


def add_gaussian_noise(
    image_rgb: np.ndarray,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    """Adiciona ruido gaussiano aditivo em [0, 1] de forma deterministica."""
    arr = np.asarray(image_rgb, dtype=np.float32)
    std = float(noise_std)
    if std <= 0.0:
        return arr.copy()

    rng = np.random.default_rng(int(seed))
    noise = rng.normal(loc=0.0, scale=std, size=arr.shape).astype(np.float32)
    return np.clip(arr + noise, 0.0, 1.0).astype(np.float32)


def rgb_to_normalized_tensor(
    image_rgb: np.ndarray,
    imagenet_mean: tuple[float, float, float],
    imagenet_std: tuple[float, float, float],
) -> torch.Tensor:
    """Converte imagem RGB float32 em [0, 1] para tensor normalizado (1, 3, H, W)."""
    arr = np.ascontiguousarray(image_rgb, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean_t = torch.tensor(imagenet_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(imagenet_std, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t
