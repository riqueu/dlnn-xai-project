"""Gera e salva os mapas de atribuicao XAI na amostra fixa de teste (E2)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import load_config, resolve_path, set_seed
from src.models.deit_tiny import load_deit_tiny
from src.models.vig_ti import load_vig_ti
from src.xai.gradcam import compute_gradcam
from src.xai.ig import compute_integrated_gradients
from src.xai.perturbations import compute_lime, compute_partition_shap
from src.xai.rollout import compute_attention_rollout
from src.xai.utils import get_predicted_class, load_image_pair

VALID_METHODS_BY_MODEL: dict[str, list[str]] = {
    "vig_ti": ["gradcam", "lime", "partition_shap", "integrated_gradients"],
    "deit_tiny": [
        "gradcam",
        "rollout",
        "lime",
        "partition_shap",
        "integrated_gradients",
    ],
}

ALL_METHODS = [
    "gradcam",
    "rollout",
    "lime",
    "partition_shap",
    "integrated_gradients",
]

METHOD_DISPLAY_NAMES = {
    "gradcam": "Grad-CAM",
    "rollout": "Rollout",
    "lime": "LIME",
    "partition_shap": "Partition SHAP",
    "integrated_gradients": "IG",
}


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
        help="Metodo de XAI a executar (all ou lista separada por virgula).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita a N imagens espacadas (piloto) e salva em results/maps/pilot/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve arquivos .npz ja existentes.",
    )
    return parser.parse_args()


def _resolve_methods(method_arg: str) -> list[str]:
    if method_arg.strip().lower() == "all":
        return list(ALL_METHODS)
    requested = [m.strip() for m in method_arg.split(",") if m.strip()]
    unknown = [m for m in requested if m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Metodos desconhecidos: {unknown}. Opcoes: {ALL_METHODS}")
    return requested


def _load_finetuned_model(
    model_key: str,
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    ckpt_dir = resolve_path(config["training"]["checkpoint_dir"])
    ckpt_path = ckpt_dir / f"{model_key}_best.pt"
    if model_key == "vig_ti":
        wrapper = load_vig_ti(config, checkpoint_path=ckpt_path)
    elif model_key == "deit_tiny":
        wrapper = load_deit_tiny(config, checkpoint_path=ckpt_path)
    else:
        raise ValueError(f"Modelo desconhecido: {model_key}")
    return wrapper.to(device).eval()


def _select_subset(subset_df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return subset_df.reset_index(drop=True)
    if limit <= 0 or limit > len(subset_df):
        raise ValueError(f"--limit deve estar entre 1 e {len(subset_df)}, recebido {limit}")
    indices = np.linspace(0, len(subset_df) - 1, num=limit, dtype=int)
    return subset_df.iloc[indices].reset_index(drop=True)


def _run_explainer(
    method: str,
    wrapper: torch.nn.Module,
    image_rgb: np.ndarray,
    image_tensor: torch.Tensor,
    target_class: int,
    config: dict[str, Any],
    device: torch.device,
    per_image_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    xai_cfg = config["xai"]
    data_cfg = config["data"]

    if method == "gradcam":
        return compute_gradcam(wrapper, image_tensor, target_class=target_class, return_raw=True)
    if method == "rollout":
        rollout_cfg = xai_cfg["rollout"]
        return compute_attention_rollout(
            wrapper,
            image_tensor,
            discard_ratio=float(rollout_cfg["discard_ratio"]),
            head_fusion=str(rollout_cfg["head_fusion"]),
            return_raw=True,
        )
    if method == "integrated_gradients":
        ig_cfg = xai_cfg["integrated_gradients"]
        return compute_integrated_gradients(
            wrapper,
            image_tensor,
            target_class=target_class,
            n_steps=int(ig_cfg["n_steps"]),
            baseline=str(ig_cfg["baseline"]),
            imagenet_mean=tuple(float(v) for v in data_cfg["imagenet_mean"]),
            imagenet_std=tuple(float(v) for v in data_cfg["imagenet_std"]),
            return_raw=True,
        )
    if method == "lime":
        lime_cfg = xai_cfg["lime"]
        return compute_lime(
            wrapper,
            image_rgb,
            target_class=target_class,
            config=config,
            device=device,
            num_samples=int(lime_cfg["num_samples"]),
            n_segments=int(lime_cfg["n_segments"]),
            compactness=float(lime_cfg["compactness"]),
            seed=per_image_seed,
            return_raw=True,
        )
    if method == "partition_shap":
        shap_cfg = xai_cfg["partition_shap"]
        return compute_partition_shap(
            wrapper,
            image_rgb,
            target_class=target_class,
            config=config,
            device=device,
            max_evals=int(shap_cfg["max_evals"]),
            batch_size=int(shap_cfg["batch_size"]),
            masker=str(shap_cfg["masker"]),
            seed=per_image_seed,
            return_raw=True,
        )
    raise ValueError(f"Metodo invalido: {method}")


def _verify_alignment_and_stats(
    archive_path: Path,
    model_key: str,
    method: str,
    subset_df: pd.DataFrame,
    e1_preds_df: pd.DataFrame,
) -> dict[str, Any]:
    with np.load(archive_path, allow_pickle=False) as data:
        maps = data["maps"]
        raw_maps = data["raw_maps"]
        target_classes = data["target_classes"]
        image_paths = data["image_paths"]
        elapsed_sec = data["elapsed_sec"]

    n = len(subset_df)
    assert maps.shape == (n, 224, 224), f"Shape inesperado em {archive_path}: {maps.shape}"
    assert raw_maps.shape == (n, 224, 224)
    all_finite = bool(np.isfinite(maps).all() and np.isfinite(raw_maps).all())
    assert all_finite, f"Valores nao finitos encontrados em {archive_path}"
    nonzero_maps = int((maps.reshape(n, -1).max(axis=1) > 0).sum())
    assert list(image_paths) == list(subset_df["image_path"]), f"Ordem divergente em {archive_path}"

    expected_preds = e1_preds_df.loc[subset_df["image_path"], f"{model_key}_pred"].to_numpy()
    matches = int((target_classes == expected_preds).sum())
    assert matches == n, f"Divergencia de predicoes E1 vs E2 em {model_key}/{method}: {matches}/{n}"

    return {
        "model": model_key,
        "method": method,
        "num_images": n,
        "total_time_sec": float(elapsed_sec.sum()),
        "mean_time_sec": float(elapsed_sec.mean()),
        "std_time_sec": float(elapsed_sec.std()),
        "min_time_sec": float(elapsed_sec.min()),
        "max_time_sec": float(elapsed_sec.max()),
        "all_finite": all_finite,
        "nonzero_maps": nonzero_maps,
        "mean_map_val": float(maps.mean()),
        "mean_map_std": float(maps.std()),
        "e1_pred_matches": matches,
    }


def _select_qualitative_rows(
    subset_df: pd.DataFrame,
    e1_preds_df: pd.DataFrame,
    is_pilot: bool,
) -> pd.DataFrame:
    merged = subset_df.merge(
        e1_preds_df.reset_index()[
            [
                "image_path",
                "vig_ti_pred",
                "vig_ti_conf",
                "vig_ti_correct",
                "deit_tiny_pred",
                "deit_tiny_conf",
                "deit_tiny_correct",
            ]
        ],
        on="image_path",
        how="left",
    )
    if is_pilot:
        return merged

    selected_indices: list[int] = []
    used_classes: set[str] = set()

    only_vig = merged[(merged["vig_ti_correct"] == 1) & (merged["deit_tiny_correct"] == 0)]
    only_deit = merged[(merged["vig_ti_correct"] == 0) & (merged["deit_tiny_correct"] == 1)]
    both_wrong = merged[(merged["vig_ti_correct"] == 0) & (merged["deit_tiny_correct"] == 0)]
    both_right = merged[(merged["vig_ti_correct"] == 1) & (merged["deit_tiny_correct"] == 1)].copy()
    both_right["combined_conf"] = both_right["vig_ti_conf"] + both_right["deit_tiny_conf"]
    both_right = both_right.sort_values("combined_conf", ascending=False)

    special_groups = [only_vig, only_deit, both_wrong]
    special_indices: list[int] = []
    for grp in special_groups:
        if len(grp) > 0:
            row_idx = int(grp.index[0])
            special_indices.append(row_idx)
            used_classes.add(str(merged.loc[row_idx, "class_name"]))

    for row_idx, row in both_right.iterrows():
        cls_name = str(row["class_name"])
        if cls_name in used_classes:
            continue
        selected_indices.append(int(row_idx))
        used_classes.add(cls_name)
        if len(selected_indices) == 3:
            break

    final_indices = selected_indices + special_indices
    return merged.loc[final_indices].reset_index(drop=True)


def _render_qualitative_grid(
    subset_df: pd.DataFrame,
    e1_preds_df: pd.DataFrame,
    maps_dir: Path,
    classes_df: pd.DataFrame,
    config: dict[str, Any],
    figures_dir: Path,
    is_pilot: bool,
) -> None:
    expected_pairs = [
        ("vig_ti", m) for m in VALID_METHODS_BY_MODEL["vig_ti"]
    ] + [
        ("deit_tiny", m) for m in VALID_METHODS_BY_MODEL["deit_tiny"]
    ]
    if not all((maps_dir / f"{m}_{mtd}.npz").exists() for m, mtd in expected_pairs):
        return

    label_to_name = dict(zip(classes_df["label"], classes_df["class_name"]))
    rows_df = _select_qualitative_rows(subset_df, e1_preds_df, is_pilot=is_pilot)
    path_to_idx = {p: i for i, p in enumerate(subset_df["image_path"])}

    loaded_maps: dict[tuple[str, str], np.ndarray] = {}
    for m, mtd in expected_pairs:
        with np.load(maps_dir / f"{m}_{mtd}.npz", allow_pickle=False) as d:
            loaded_maps[(m, mtd)] = d["maps"]

    col_specs: list[tuple[str, str | None, str | None]] = [
        ("Original", None, None),
        ("ViG: Grad-CAM", "vig_ti", "gradcam"),
        ("ViG: LIME", "vig_ti", "lime"),
        ("ViG: SHAP", "vig_ti", "partition_shap"),
        ("ViG: IG", "vig_ti", "integrated_gradients"),
        ("DeiT: Grad-CAM", "deit_tiny", "gradcam"),
        ("DeiT: Rollout", "deit_tiny", "rollout"),
        ("DeiT: LIME", "deit_tiny", "lime"),
        ("DeiT: SHAP", "deit_tiny", "partition_shap"),
        ("DeiT: IG", "deit_tiny", "integrated_gradients"),
    ]

    n_rows = len(rows_df)
    n_cols = len(col_specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.2), constrained_layout=True)
    if n_rows == 1:
        axes = axes[None, :]

    for r_idx, row in rows_df.iterrows():
        img_path = str(row["image_path"])
        sub_idx = path_to_idx[img_path]
        img_rgb, _ = load_image_pair(img_path, config)

        true_cls = str(row["class_name"])
        vig_pred = label_to_name[int(row["vig_ti_pred"])]
        deit_pred = label_to_name[int(row["deit_tiny_pred"])]
        vig_ok = "OK" if int(row["vig_ti_correct"]) == 1 else "ERR"
        deit_ok = "OK" if int(row["deit_tiny_correct"]) == 1 else "ERR"

        for c_idx, (col_title, m_key, mtd_key) in enumerate(col_specs):
            ax = axes[r_idx, c_idx]
            ax.imshow(img_rgb)
            if m_key is not None and mtd_key is not None:
                smap = loaded_maps[(m_key, mtd_key)][sub_idx]
                ax.imshow(smap, cmap="inferno", alpha=0.55, vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])

            if r_idx == 0:
                ax.set_title(col_title, fontsize=9, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(
                    f"Real: {true_cls}\nViG [{vig_ok}]: {vig_pred}\nDeiT [{deit_ok}]: {deit_pred}",
                    fontsize=7.5,
                    rotation=0,
                    labelpad=58,
                    va="center",
                    ha="right",
                )

    figures_dir.mkdir(parents=True, exist_ok=True)
    if is_pilot:
        fig.suptitle("Validacao Piloto E2: Mapas de Atribuicao (5 Imagens)", fontsize=11)
        fig.savefig(figures_dir / "e2_pilot_qualitative.png", dpi=200)
    else:
        fig.suptitle("Comparacao Qualitativa de Mapas XAI (E2: ViG-Ti vs. DeiT-Tiny)", fontsize=11)
        fig.savefig(figures_dir / "e2_qualitative_grid.pdf", dpi=300)
        fig.savefig(figures_dir / "e2_qualitative_grid.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Ponto de entrada para geracao dos mapas de explicabilidade."""
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = resolve_path(config["data"]["splits_dir"])
    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])
    figures_dir = resolve_path(config["evaluation"]["figures_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    is_pilot = args.limit is not None
    base_maps_dir = resolve_path(config["xai"]["output_dir"])
    maps_dir = base_maps_dir / "pilot" if is_pilot else base_maps_dir
    maps_dir.mkdir(parents=True, exist_ok=True)

    full_subset_df = pd.read_csv(splits_dir / "xai_subset.csv")
    subset_df = _select_subset(full_subset_df, args.limit)
    e1_preds_df = pd.read_csv(metrics_dir / "e1_test_predictions.csv").set_index("image_path")
    classes_df = pd.read_json(splits_dir / "classes.json")

    model_keys = ["vig_ti", "deit_tiny"] if args.model == "all" else [args.model]
    requested_methods = _resolve_methods(args.method)

    # Verifica requisicao explicita invalida (ex.: vig_ti + rollout isolado).
    if args.model == "vig_ti" and args.method.strip().lower() != "all" and "rollout" in requested_methods:
        raise ValueError("Attention Rollout nao se aplica ao modelo ViG-Ti.")

    summary_rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        valid_methods = [m for m in requested_methods if m in VALID_METHODS_BY_MODEL[model_key]]
        if not valid_methods:
            continue

        wrapper = _load_finetuned_model(model_key, config, device)

        for method in valid_methods:
            archive_path = maps_dir / f"{model_key}_{method}.npz"
            if archive_path.exists() and not args.overwrite:
                with np.load(archive_path, allow_pickle=False) as existing:
                    if len(existing["maps"]) == len(subset_df):
                        print(f"[E2] Reutilizando arquivo existente: {archive_path.name}")
                        summary_rows.append(
                            _verify_alignment_and_stats(
                                archive_path, model_key, method, subset_df, e1_preds_df
                            )
                        )
                        continue

            n = len(subset_df)
            maps_arr = np.zeros((n, 224, 224), dtype=np.float32)
            raw_maps_arr = np.zeros((n, 224, 224), dtype=np.float32)
            target_arr = np.zeros(n, dtype=np.int64)
            true_arr = subset_df["label"].to_numpy(dtype=np.int64)
            probs_arr = np.zeros(n, dtype=np.float32)
            elapsed_arr = np.zeros(n, dtype=np.float32)
            paths_list = subset_df["image_path"].tolist()

            desc = f"{model_key}/{method}"
            for idx, row in tqdm(subset_df.iterrows(), total=n, desc=desc):
                per_image_seed = seed + int(idx)
                set_seed(per_image_seed)

                image_rgb, image_tensor = load_image_pair(row["image_path"], config)
                image_tensor_dev = image_tensor.to(device)
                pred_class, pred_prob = get_predicted_class(wrapper, image_tensor_dev)

                t0 = time.perf_counter()
                norm_map, raw_map = _run_explainer(
                    method=method,
                    wrapper=wrapper,
                    image_rgb=image_rgb,
                    image_tensor=image_tensor_dev,
                    target_class=pred_class,
                    config=config,
                    device=device,
                    per_image_seed=per_image_seed,
                )
                elapsed = time.perf_counter() - t0

                assert norm_map.shape == (224, 224), f"Shape invalido: {norm_map.shape}"
                assert np.isfinite(norm_map).all() and np.isfinite(raw_map).all()

                maps_arr[idx] = norm_map
                raw_maps_arr[idx] = raw_map
                target_arr[idx] = pred_class
                probs_arr[idx] = pred_prob
                elapsed_arr[idx] = elapsed

            np.savez_compressed(
                archive_path,
                maps=maps_arr,
                raw_maps=raw_maps_arr,
                target_classes=target_arr,
                true_labels=true_arr,
                pred_probs=probs_arr,
                image_paths=np.asarray(paths_list),
                elapsed_sec=elapsed_arr,
            )

            stats = _verify_alignment_and_stats(
                archive_path, model_key, method, subset_df, e1_preds_df
            )
            summary_rows.append(stats)
            print(
                f"[E2] {model_key}/{method}: {n} imagens | "
                f"total={stats['total_time_sec']:.2f}s | "
                f"media={stats['mean_time_sec']:.3f}s/img | "
                f"e1_match={stats['e1_pred_matches']}/{n}"
            )

    summary_payload = {
        "seed": seed,
        "device": str(device),
        "num_images": len(subset_df),
        "is_pilot": is_pilot,
        "results": summary_rows,
    }

    if is_pilot:
        with (metrics_dir / "e2_pilot_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, ensure_ascii=False)
    else:
        pd.DataFrame(summary_rows).to_csv(metrics_dir / "e2_xai_summary.csv", index=False)
        with (metrics_dir / "e2_xai_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    _render_qualitative_grid(
        subset_df=subset_df,
        e1_preds_df=e1_preds_df,
        maps_dir=maps_dir,
        classes_df=classes_df,
        config=config,
        figures_dir=figures_dir,
        is_pilot=is_pilot,
    )


if __name__ == "__main__":
    main()
