"""Wrapper da Vision GNN isotropica (ViG-Ti) com hooks de nos e arestas k-NN."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ViGTiWrapper:
    """Encapsula o modelo ViG-Ti para extracao de ativacoes e grafo k-NN."""

    def __init__(self, model: Any) -> None:
        """Inicializa o wrapper com a instancia do backbone ViG-Ti."""
        self.model = model

    def forward(self, x: Any) -> Any:
        """Executa a passagem forward no modelo ViG-Ti."""
        raise NotImplementedError("ViGTiWrapper.forward ainda nao implementado.")

    def get_node_activations(self) -> Any:
        """Retorna as ativacoes espaciais 14x14 dos nos da ultima camada."""
        raise NotImplementedError("ViGTiWrapper.get_node_activations ainda nao implementado.")

    def get_knn_edges(self) -> Any:
        """Retorna os indices de arestas do grafo k-NN da camada alvo."""
        raise NotImplementedError("ViGTiWrapper.get_knn_edges ainda nao implementado.")


def load_vig_ti(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
) -> ViGTiWrapper:
    """Carrega a arquitetura ViG-Ti com pesos pre-treinados ou ajustados."""
    raise NotImplementedError("load_vig_ti ainda nao implementado.")
