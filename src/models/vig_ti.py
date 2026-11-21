"""Wrapper da Vision GNN isotropica (ViG-Ti) com hooks de nos e arestas k-NN."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.config import resolve_path
from src.models.vendor.vig_pytorch import vig_ti_224_gelu


class ViGTiWrapper(nn.Module):
    """Encapsula o modelo ViG-Ti para extracao de ativacoes e grafo k-NN."""

    def __init__(self, model: nn.Module) -> None:
        """Inicializa o wrapper e registra os hooks de captura."""
        super().__init__()
        self.model = model
        self.capture_features: bool = False
        self.capture_graph: bool = False
        self._node_activations: torch.Tensor | None = None
        self._knn_edges: dict[int, torch.Tensor] = {}

        self._register_hooks()

    def _register_hooks(self) -> None:
        def _feature_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if self.capture_features:
                self._node_activations = output

        self.model.backbone[-1].register_forward_hook(_feature_hook)

        for idx, block in enumerate(self.model.backbone):
            knn_module = block[0].graph_conv.dilated_knn_graph

            def _make_graph_hook(layer_idx: int):
                def _graph_hook(
                    _module: nn.Module,
                    _inputs: tuple[Any, ...],
                    output: torch.Tensor,
                ) -> None:
                    if self.capture_graph:
                        self._knn_edges[layer_idx] = output.detach()

                return _graph_hook

            knn_module.register_forward_hook(_make_graph_hook(idx))

    def enable_feature_capture(self, enabled: bool = True) -> None:
        """Ativa ou desativa a captura das ativacoes dos nos na ultima camada."""
        self.capture_features = enabled

    def enable_graph_capture(self, enabled: bool = True) -> None:
        """Ativa ou desativa a captura das arestas k-NN nas camadas Grapher."""
        self.capture_graph = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem forward no modelo ViG-Ti."""
        self._node_activations = None
        self._knn_edges.clear()
        return self.model(x)

    def get_node_activations(self) -> torch.Tensor:
        """Retorna as ativacoes espaciais 14x14 dos nos da ultima camada."""
        if self._node_activations is None:
            raise RuntimeError(
                "Ativacoes dos nos nao capturadas. Ative capture_features antes do forward."
            )
        return self._node_activations

    def get_knn_edges(self, layer_idx: int = -1) -> torch.Tensor:
        """Retorna os indices de arestas do grafo k-NN da camada alvo."""
        n_blocks = len(self.model.backbone)
        resolved_idx = layer_idx if layer_idx >= 0 else n_blocks + layer_idx
        if resolved_idx not in self._knn_edges:
            raise RuntimeError(
                f"Arestas k-NN da camada {layer_idx} nao capturadas. Ative capture_graph antes do forward."
            )
        return self._knn_edges[resolved_idx]


def download_vig_pretrained(url: str, dest: Path) -> Path:
    """Baixa o checkpoint oficial pre-treinado da ViG-Ti se ainda nao existir localmente."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest

    tmp_path = dest.with_suffix(dest.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(dest)
    return dest


def load_vig_ti(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
) -> ViGTiWrapper:
    """Carrega a arquitetura ViG-Ti com pesos pre-treinados ou ajustados."""
    vig_cfg = config["models"]["vig_ti"]
    num_classes = int(vig_cfg["num_classes"])
    k = int(vig_cfg["k"])

    backbone = vig_ti_224_gelu(pretrained=False, num_classes=num_classes, num_knn=k)

    if checkpoint_path is not None:
        ckpt_file = resolve_path(checkpoint_path)
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        backbone.load_state_dict(state_dict, strict=True)
    else:
        pretrained_path = resolve_path(vig_cfg["pretrained_ckpt"])
        if not pretrained_path.exists():
            download_vig_pretrained(vig_cfg["pretrained_url"], pretrained_path)

        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            raw_state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            raw_state = ckpt["model"]
        else:
            raw_state = ckpt

        cleaned_state: dict[str, Any] = {}
        head_keys = {"prediction.4.weight", "prediction.4.bias"}
        for key, value in raw_state.items():
            clean_key = key.removeprefix("module.")
            if clean_key in head_keys:
                continue
            cleaned_state[clean_key] = value

        missing_keys, unexpected_keys = backbone.load_state_dict(cleaned_state, strict=False)
        assert unexpected_keys == [], f"Chaves inesperadas no checkpoint ViG-Ti: {unexpected_keys}"
        assert set(missing_keys) == head_keys, f"Chaves ausentes inesperadas na ViG-Ti: {missing_keys}"

    return ViGTiWrapper(backbone)
