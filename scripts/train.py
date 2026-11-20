"""Executa o fine-tuning controlado de ViG-Ti e DeiT-Tiny no Imagenette."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Treina os modelos ViG-Ti e DeiT-Tiny no Imagenette.",
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
        help="Modelo alvo do fine-tuning.",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada para treinamento dos modelos."""
    _ = parse_args()
    raise NotImplementedError("scripts/train.py ainda nao implementado.")


if __name__ == "__main__":
    main()
