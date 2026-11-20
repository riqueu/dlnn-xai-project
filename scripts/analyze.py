"""Consolida testes estatisticos, tabelas e figuras finais dos resultados."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Gera tabelas, testes pareados e figuras de analise.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Caminho para o arquivo de configuracao YAML.",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada para analise estatistica e geracao de figuras."""
    _ = parse_args()
    raise NotImplementedError("scripts/analyze.py ainda nao implementado.")


if __name__ == "__main__":
    main()
