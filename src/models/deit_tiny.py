"""Wrapper do Vision Transformer (DeiT-Tiny) com hooks de tokens e atencao."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn

from src.config import resolve_path


class DeiTTinyWrapper(nn.Module):
    """Encapsula o modelo DeiT-Tiny para extracao de tokens espaciais e mapas de atencao."""

    def __init__(self, model: nn.Module) -> None:
        """Inicializa o wrapper e registra os hooks de tokens e atencao."""
        super().__init__()
        self.model = model
        self.capture_features: bool = False
        self.capture_attention: bool = False
        self._spatial_tokens: torch.Tensor | None = None
        self._attention_maps: list[torch.Tensor] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        def _token_hook(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> torch.Tensor | None:
            if not self.capture_features:
                return None

            prefix = getattr(self.model, "num_prefix_tokens", 1)
            prefix_tokens = output[:, :prefix, :]
            patch_tokens = output[:, prefix:, :]
            batch_size, num_patches, channels = patch_tokens.shape
            grid_size = math.isqrt(num_patches)
            if grid_size * grid_size != num_patches:
                raise ValueError(f"Numero de patches nao forma grade quadrada: {num_patches}")

            # Reconstroi output a partir da grade NCHW para manter os gradientes conectados ao grafo.
            spatial = patch_tokens.permute(0, 2, 1).reshape(
                batch_size,
                channels,
                grid_size,
                grid_size,
            )
            self._spatial_tokens = spatial
            reconstructed_patches = spatial.flatten(2).permute(0, 2, 1)
            return torch.cat([prefix_tokens, reconstructed_patches], dim=1)

        # Hook em blocks[-1].norm1 garante gradientes nao-nulos nos tokens espaciais antes do pooling do CLS.
        self.model.blocks[-1].norm1.register_forward_hook(_token_hook)

        def _attn_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if self.capture_attention:
                self._attention_maps.append(output.detach())

        for block in self.model.blocks:
            block.attn.attn_drop.register_forward_hook(_attn_hook)

    def enable_feature_capture(self, enabled: bool = True) -> None:
        """Ativa ou desativa a captura dos tokens espaciais da ultima camada."""
        self.capture_features = enabled

    def enable_attention_capture(self, enabled: bool = True) -> None:
        """Ativa ou desativa a captura dos mapas de self-attention por bloco."""
        self.capture_attention = enabled
        for block in self.model.blocks:
            block.attn.fused_attn = not enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem forward no modelo DeiT-Tiny."""
        self._spatial_tokens = None
        self._attention_maps.clear()
        return self.model(x)

    def get_spatial_tokens(self) -> torch.Tensor:
        """Retorna os tokens espaciais 14x14 da ultima camada em formato NCHW."""
        if self._spatial_tokens is None:
            raise RuntimeError(
                "Tokens espaciais nao capturados. Ative capture_features antes do forward."
            )
        return self._spatial_tokens

    def get_attention_maps(self) -> list[torch.Tensor]:
        """Retorna as matrizes de self-attention coletadas por bloco."""
        if not self._attention_maps:
            raise RuntimeError(
                "Mapas de atencao nao capturados. Ative enable_attention_capture antes do forward."
            )
        return list(self._attention_maps)


def load_deit_tiny(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
) -> DeiTTinyWrapper:
    """Carrega a arquitetura DeiT-Tiny com pesos pre-treinados ou ajustados."""
    deit_cfg = config["models"]["deit_tiny"]
    name = str(deit_cfg["name"])
    num_classes = int(deit_cfg["num_classes"])

    if checkpoint_path is not None:
        model = timm.create_model(name, pretrained=False, num_classes=num_classes)
        ckpt_file = resolve_path(checkpoint_path)
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
    else:
        model = timm.create_model(
            name,
            pretrained=bool(deit_cfg.get("pretrained", True)),
            num_classes=num_classes,
        )

    return DeiTTinyWrapper(model)
