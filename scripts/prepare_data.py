"""Prepara os dados do Imagenette e gera os splits deterministicos."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Baixa o Imagenette e gera os splits estratificados.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Caminho para o arquivo de configuracao YAML.",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada para preparacao dos dados."""
    _ = parse_args()
    raise NotImplementedError("scripts/prepare_data.py ainda nao implementado.")


if __name__ == "__main__":
    main()
