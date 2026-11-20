"""Executa as metricas quantitativas de fidelidade, sanity checks e graph rewiring."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Avalia fidelidade (E3), sanity checks (E4) e rewiring (E5).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Caminho para o arquivo de configuracao YAML.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["e3", "e4", "e5", "all"],
        default="all",
        help="Experimento de avaliacao a executar.",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada para avaliacao quantitativa."""
    _ = parse_args()
    raise NotImplementedError("scripts/evaluate.py ainda nao implementado.")


if __name__ == "__main__":
    main()
