"""Prepara os dados do Imagenette e gera os splits deterministicos."""

from __future__ import annotations

import argparse
import json

from src.config import load_config, resolve_path, set_seed
from src.data import create_splits, create_xai_subset, download_imagenette


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
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    set_seed(seed)

    data_cfg = config["data"]
    splits_dir = resolve_path(data_cfg["splits_dir"])

    dataset_dir = download_imagenette(data_cfg["url"], data_cfg["raw_dir"])
    create_splits(
        raw_dir=dataset_dir,
        splits_dir=splits_dir,
        val_ratio=float(data_cfg["val_ratio"]),
        seed=seed,
    )
    create_xai_subset(
        test_csv=splits_dir / "test.csv",
        output_csv=splits_dir / "xai_subset.csv",
        per_class=int(data_cfg["xai_subset_per_class"]),
        seed=seed,
    )

    with (splits_dir / "split_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    print(f"Dataset preparado em: {dataset_dir}")
    print(f"Splits salvos em: {splits_dir}")
    print("Contagens (zero vazamento verificado):", json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
