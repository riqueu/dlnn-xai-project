"""Wrapper do Vision Transformer (DeiT-Tiny) com hooks de tokens e atencao."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DeiTTinyWrapper:
    """Encapsula o modelo DeiT-Tiny para extracao de tokens espaciais e mapas de atencao."""

    def __init__(self, model: Any) -> None:
        """Inicializa o wrapper com a instancia do backbone DeiT-Tiny."""
        self.model = model

    def forward(self, x: Any) -> Any:
        """Executa a passagem forward no modelo DeiT-Tiny."""
        raise NotImplementedError("DeiTTinyWrapper.forward ainda nao implementado.")

    def get_spatial_tokens(self) -> Any:
        """Retorna os tokens espaciais 14x14 da ultima camada."""
        raise NotImplementedError("DeiTTinyWrapper.get_spatial_tokens ainda nao implementado.")

    def get_attention_maps(self) -> Any:
        """Retorna as matrizes de self-attention coletadas por bloco."""
        raise NotImplementedError("DeiTTinyWrapper.get_attention_maps ainda nao implementado.")


def load_deit_tiny(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
) -> DeiTTinyWrapper:
    """Carrega a arquitetura DeiT-Tiny com pesos pre-treinados ou ajustados."""
    raise NotImplementedError("load_deit_tiny ainda nao implementado.")
