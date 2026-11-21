"""Utilitarios compartilhados para carregamento de imagens, normalizacao e inferencia em lote."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.config import resolve_path


def normalize_map(saliency: np.ndarray) -> np.ndarray:
    """Normaliza um mapa de saliencia para o intervalo [0, 1] em float32."""
    arr = np.asarray(saliency, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Mapa de saliencia contem valores nao finitos (NaN ou Inf).")
    min_val = float(arr.min())
    max_val = float(arr.max())
    span = max_val - min_val
    if span < 1.0e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - min_val) / span).astype(np.float32)


def load_image_pair(
    image_path: str | Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, torch.Tensor]:
    """Carrega a imagem RGB em [0, 1] (H, W, 3) e o tensor normalizado (1, 3, H, W)."""
    data_cfg = config["data"]
    size = int(data_cfg["image_size"])
    mean = list(data_cfg["imagenet_mean"])
    std = list(data_cfg["imagenet_std"])

    resolved = resolve_path(image_path)
    with Image.open(resolved) as img:
        rgb_img = img.convert("RGB")
        resized = transforms.Resize(
            (size, size),
            interpolation=InterpolationMode.BICUBIC,
        )(rgb_img)

    image_rgb = np.asarray(resized, dtype=np.float32) / 255.0
    tensor_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    image_tensor = tensor_tf(resized).unsqueeze(0)
    return image_rgb, image_tensor


def get_predicted_class(
    model: nn.Module,
    image_tensor: torch.Tensor,
) -> tuple[int, float]:
    """Retorna a classe prevista e sua probabilidade softmax."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    return int(pred.item()), float(conf.item())


def make_batch_predict_fn(
    model: nn.Module,
    config: dict[str, Any],
    device: torch.device,
    return_probs: bool = True,
    chunk_size: int = 64,
) -> Callable[[np.ndarray], np.ndarray]:
    """Constroi funcao de predicao em lote para explicadores baseados em perturbacao."""
    data_cfg = config["data"]
    mean_t = torch.tensor(data_cfg["imagenet_mean"], device=device, dtype=torch.float32).view(
        1, 3, 1, 1
    )
    std_t = torch.tensor(data_cfg["imagenet_std"], device=device, dtype=torch.float32).view(
        1, 3, 1, 1
    )

    model.eval()

    def _predict(images: np.ndarray) -> np.ndarray:
        arr = np.asarray(images, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.size > 0 and float(arr.max()) > 1.5:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(arr), chunk_size):
                batch_np = arr[start : start + chunk_size]
                batch_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device)
                batch_t = (batch_t - mean_t) / std_t
                logits = model(batch_t)
                out = torch.softmax(logits, dim=1) if return_probs else logits
                outputs.append(out.detach().cpu().numpy())

        return np.concatenate(outputs, axis=0)

    return _predict
