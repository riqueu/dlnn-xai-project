"""Gera e salva os mapas de atribuicao XAI na amostra fixa de teste."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Executa os metodos de XAI sobre a amostra fixa de teste.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Caminho para o arquivo de configuracao YAML.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vig_ti", "deit_tiny", "all"],
        default="all",
        help="Modelo a ser explicado.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        help="Metodo de XAI a executar (ou all).",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada para geracao dos mapas de explicabilidade."""
    _ = parse_args()
    raise NotImplementedError("scripts/run_xai.py ainda nao implementado.")


if __name__ == "__main__":
    main()
