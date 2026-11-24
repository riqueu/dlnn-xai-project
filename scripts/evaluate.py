"""Executa as metricas quantitativas de fidelidade, sanity checks e graph rewiring."""

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
from scipy.stats import wilcoxon
from tqdm import tqdm

from src.config import load_config, resolve_path, set_seed
from src.models.deit_tiny import load_deit_tiny
from src.models.vig_ti import load_vig_ti
from src.metrics.faithfulness import compute_deletion_insertion_auc, generate_random_saliency
from src.xai.utils import load_image_pair

METHODS_BY_MODEL: dict[str, list[str]] = {
    "vig_ti": ["gradcam", "lime", "partition_shap", "integrated_gradients"],
    "deit_tiny": [
        "gradcam",
        "rollout",
        "lime",
        "partition_shap",
        "integrated_gradients",
    ],
}

MODEL_DISPLAY = {
    "vig_ti": "ViG-Ti",
    "deit_tiny": "DeiT-Tiny",
}

METHOD_DISPLAY = {
    "gradcam": "Grad-CAM",
    "rollout": "Rollout",
    "lime": "LIME",
    "partition_shap": "Partition SHAP",
    "integrated_gradients": "IG",
    "random": "Random",
}

METHOD_COLORS = {
    "gradcam": "#1f77b4",
    "rollout": "#9467bd",
    "lime": "#ff7f0e",
    "partition_shap": "#2ca02c",
    "integrated_gradients": "#d62728",
    "random": "#7f7f7f",
}


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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Executa teste rapido nas primeiras N imagens sem sobrescrever saidas finais.",
    )
    return parser.parse_args()


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


def _paired_wilcoxon(
    df: pd.DataFrame,
    group_name: str,
    mask_a: pd.Series,
    mask_b: pd.Series,
    label_a: str,
    label_b: str,
    baseline: str,
    metric: str,
) -> dict[str, Any]:
    sub_a = df[mask_a].sort_values("image_idx")
    sub_b = df[mask_b].sort_values("image_idx")
    assert list(sub_a["image_idx"]) == list(sub_b["image_idx"]), (
        f"Desalinhamento de pareamento em {label_a} vs {label_b}"
    )

    vals_a = sub_a[metric].to_numpy(dtype=np.float64)
    vals_b = sub_b[metric].to_numpy(dtype=np.float64)
    diffs = vals_a - vals_b

    if np.allclose(diffs, 0.0):
        stat, p_val = 0.0, 1.0
    else:
        res = wilcoxon(vals_a, vals_b, zero_method="wilcox")
        stat, p_val = float(res.statistic), float(res.pvalue)

    return {
        "comparison_group": group_name,
        "label_a": label_a,
        "label_b": label_b,
        "baseline": baseline,
        "metric": metric,
        "n": int(len(vals_a)),
        "mean_a": float(vals_a.mean()),
        "mean_b": float(vals_b.mean()),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "wilcoxon_stat": stat,
        "p_value": p_val,
        "significant_005": bool(p_val < 0.05),
    }


def _compute_e3_wilcoxon(per_img_df: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["deletion_auc", "insertion_auc"]

    for baseline in baselines:
        for metric in metrics:
            # 1. Cada metodo real contra baseline aleatorio
            for model_key, methods in METHODS_BY_MODEL.items():
                for method in methods:
                    mask_a = (
                        (per_img_df["model"] == model_key)
                        & (per_img_df["method"] == method)
                        & (per_img_df["baseline"] == baseline)
                    )
                    mask_b = (
                        (per_img_df["model"] == model_key)
                        & (per_img_df["method"] == "random")
                        & (per_img_df["baseline"] == baseline)
                    )
                    rows.append(
                        _paired_wilcoxon(
                            per_img_df,
                            "method_vs_random",
                            mask_a,
                            mask_b,
                            f"{model_key}/{method}",
                            f"{model_key}/random",
                            baseline,
                            metric,
                        )
                    )

            # 2. ViG-Ti vs DeiT-Tiny nos metodos simetricos
            for method in ["gradcam", "lime", "partition_shap", "integrated_gradients"]:
                mask_a = (
                    (per_img_df["model"] == "vig_ti")
                    & (per_img_df["method"] == method)
                    & (per_img_df["baseline"] == baseline)
                )
                mask_b = (
                    (per_img_df["model"] == "deit_tiny")
                    & (per_img_df["method"] == method)
                    & (per_img_df["baseline"] == baseline)
                )
                rows.append(
                    _paired_wilcoxon(
                        per_img_df,
                        "vig_vs_deit_symmetric",
                        mask_a,
                        mask_b,
                        f"vig_ti/{method}",
                        f"deit_tiny/{method}",
                        baseline,
                        metric,
                    )
                )

            # 3. DeiT Grad-CAM vs DeiT Attention Rollout
            rows.append(
                _paired_wilcoxon(
                    per_img_df,
                    "deit_gradcam_vs_rollout",
                    (per_img_df["model"] == "deit_tiny")
                    & (per_img_df["method"] == "gradcam")
                    & (per_img_df["baseline"] == baseline),
                    (per_img_df["model"] == "deit_tiny")
                    & (per_img_df["method"] == "rollout")
                    & (per_img_df["baseline"] == baseline),
                    "deit_tiny/gradcam",
                    "deit_tiny/rollout",
                    baseline,
                    metric,
                )
            )

            # 4. ViG Grad-CAM vs DeiT Attention Rollout
            rows.append(
                _paired_wilcoxon(
                    per_img_df,
                    "vig_gradcam_vs_deit_rollout",
                    (per_img_df["model"] == "vig_ti")
                    & (per_img_df["method"] == "gradcam")
                    & (per_img_df["baseline"] == baseline),
                    (per_img_df["model"] == "deit_tiny")
                    & (per_img_df["method"] == "rollout")
                    & (per_img_df["baseline"] == baseline),
                    "vig_ti/gradcam",
                    "deit_tiny/rollout",
                    baseline,
                    metric,
                )
            )

    return pd.DataFrame(rows)


def _plot_e3_curves_for_baseline(
    curves_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    baseline: str,
    n_images: int,
    figures_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)

    panels = [
        (0, 0, "vig_ti", "deletion", "ViG-Ti: Deletion (menor e melhor)"),
        (0, 1, "deit_tiny", "deletion", "DeiT-Tiny: Deletion (menor e melhor)"),
        (1, 0, "vig_ti", "insertion", "ViG-Ti: Insertion (maior e melhor)"),
        (1, 1, "deit_tiny", "insertion", "DeiT-Tiny: Insertion (maior e melhor)"),
    ]

    sem_scale = 1.0 / np.sqrt(float(n_images))

    for row_idx, col_idx, model_key, curve_kind, title in panels:
        ax = axes[row_idx, col_idx]
        methods = METHODS_BY_MODEL[model_key] + ["random"]

        for method in methods:
            sub = curves_df[
                (curves_df["model"] == model_key)
                & (curves_df["method"] == method)
                & (curves_df["baseline"] == baseline)
            ].sort_values("step")
            if sub.empty:
                continue

            auc_row = summary_df[
                (summary_df["model"] == model_key)
                & (summary_df["method"] == method)
                & (summary_df["baseline"] == baseline)
            ].iloc[0]

            frac = sub["fraction"].to_numpy() * 100.0
            mean_vals = sub[f"{curve_kind}_mean"].to_numpy()
            sem_vals = sub[f"{curve_kind}_std"].to_numpy() * sem_scale
            auc_val = float(auc_row[f"{curve_kind}_auc_mean"])

            color = METHOD_COLORS[method]
            linestyle = "--" if method == "random" else "-"
            lw = 1.6 if method == "random" else 2.0
            label = f"{METHOD_DISPLAY[method]} (AUC={auc_val:.3f})"

            ax.plot(frac, mean_vals, color=color, linestyle=linestyle, linewidth=lw, label=label)
            ax.fill_between(
                frac,
                np.clip(mean_vals - sem_vals, 0.0, 1.0),
                np.clip(mean_vals + sem_vals, 0.0, 1.0),
                color=color,
                alpha=0.14,
            )

        ax.set_title(title)
        ax.set_xlabel("Pixels modificados (%)")
        ax.set_ylabel("Probabilidade da classe prevista")
        ax.set_xlim(0.0, 100.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle=":", alpha=0.5)
        loc = "upper right" if curve_kind == "deletion" else "lower right"
        ax.legend(loc=loc, frameon=True)

    baseline_label = "Desfoque Gaussiano (blur)" if baseline == "blur" else "Imagem Preta (black)"
    fig.suptitle(
        f"Curvas de Fidelidade Deletion/Insertion (E3: Oclusao por {baseline_label}, N={n_images})",
        fontsize=12,
    )
    fig.savefig(figures_dir / f"e3_curves_{baseline}.pdf", dpi=300)
    fig.savefig(figures_dir / f"e3_curves_{baseline}.png", dpi=300)
    plt.close(fig)


def _plot_e3_auc_comparison(
    summary_df: pd.DataFrame,
    n_images: int,
    figures_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)

    ordered_methods = [
        "gradcam",
        "rollout",
        "lime",
        "partition_shap",
        "integrated_gradients",
        "random",
    ]
    x = np.arange(len(ordered_methods), dtype=np.float64)
    width = 0.36
    sem_scale = 1.0 / np.sqrt(float(n_images))

    panels = [
        (0, 0, "blur", "deletion_auc", "Deletion AUC (blur) - menor e melhor"),
        (0, 1, "blur", "insertion_auc", "Insertion AUC (blur) - maior e melhor"),
        (1, 0, "black", "deletion_auc", "Deletion AUC (black) - menor e melhor"),
        (1, 1, "black", "insertion_auc", "Insertion AUC (black) - maior e melhor"),
    ]

    for row_idx, col_idx, baseline, metric, title in panels:
        ax = axes[row_idx, col_idx]
        vig_means, vig_sems = [], []
        deit_means, deit_sems = [], []

        for method in ordered_methods:
            sub_v = summary_df[
                (summary_df["model"] == "vig_ti")
                & (summary_df["method"] == method)
                & (summary_df["baseline"] == baseline)
            ]
            if sub_v.empty:
                vig_means.append(np.nan)
                vig_sems.append(0.0)
            else:
                vig_means.append(float(sub_v.iloc[0][f"{metric}_mean"]))
                vig_sems.append(float(sub_v.iloc[0][f"{metric}_std"]) * sem_scale)

            sub_d = summary_df[
                (summary_df["model"] == "deit_tiny")
                & (summary_df["method"] == method)
                & (summary_df["baseline"] == baseline)
            ]
            deit_means.append(float(sub_d.iloc[0][f"{metric}_mean"]))
            deit_sems.append(float(sub_d.iloc[0][f"{metric}_std"]) * sem_scale)

        bars_v = ax.bar(
            x - width / 2,
            vig_means,
            width,
            yerr=vig_sems,
            capsize=3,
            color="#1f77b4",
            alpha=0.88,
            label="ViG-Ti",
        )
        bars_d = ax.bar(
            x + width / 2,
            deit_means,
            width,
            yerr=deit_sems,
            capsize=3,
            color="#d62728",
            alpha=0.88,
            label="DeiT-Tiny",
        )

        for bar_group in (bars_v, bars_d):
            for bar in bar_group:
                val = bar.get_height()
                if np.isfinite(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        val + 0.015,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.2,
                        rotation=45,
                    )

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_DISPLAY[m] for m in ordered_methods], rotation=18, ha="right")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)

    fig.suptitle(
        f"Comparacao de Fidelidade (Deletion e Insertion AUC +/- 1 SEM, N={n_images})",
        fontsize=12,
    )
    fig.savefig(figures_dir / "e3_auc_comparison.pdf", dpi=300)
    fig.savefig(figures_dir / "e3_auc_comparison.png", dpi=300)
    plt.close(fig)


def run_e3(config: dict[str, Any], device: torch.device, limit: int | None = None) -> None:
    """Executa o experimento E3 de Deletion e Insertion AUC."""
    t_start = time.perf_counter()
    seed = int(config["seed"])
    splits_dir = resolve_path(config["data"]["splits_dir"])
    maps_dir = resolve_path(config["xai"]["output_dir"])
    eval_cfg = config["evaluation"]
    faith_cfg = eval_cfg["faithfulness"]
    metrics_dir = resolve_path(eval_cfg["metrics_dir"])
    figures_dir = resolve_path(eval_cfg["figures_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    steps = int(faith_cfg["steps"])
    baselines = [str(b) for b in faith_cfg["baselines"]]
    blur_kernel_size = int(faith_cfg["blur_kernel_size"])
    blur_sigma = float(faith_cfg["blur_sigma"])
    include_random = bool(faith_cfg.get("include_random_baseline", True))

    data_cfg = config["data"]
    imagenet_mean = tuple(float(v) for v in data_cfg["imagenet_mean"])
    imagenet_std = tuple(float(v) for v in data_cfg["imagenet_std"])
    image_size = int(data_cfg["image_size"])

    subset_df = pd.read_csv(splits_dir / "xai_subset.csv")
    if limit is not None:
        subset_df = subset_df.head(limit).reset_index(drop=True)
    n_images = len(subset_df)

    e1_preds_df = pd.read_csv(metrics_dir / "e1_test_predictions.csv").set_index("image_path")
    images_rgb = [load_image_pair(p, config)[0] for p in subset_df["image_path"]]

    per_image_rows: list[dict[str, Any]] = []
    curves_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_key in ["vig_ti", "deit_tiny"]:
        wrapper = _load_finetuned_model(model_key, config, device)
        methods = list(METHODS_BY_MODEL[model_key])
        if include_random:
            methods.append("random")

        expected_preds = e1_preds_df.loc[subset_df["image_path"], f"{model_key}_pred"].to_numpy(dtype=np.int64)
        expected_confs = e1_preds_df.loc[subset_df["image_path"], f"{model_key}_conf"].to_numpy(dtype=np.float64)
        expected_correct = e1_preds_df.loc[subset_df["image_path"], f"{model_key}_correct"].to_numpy(dtype=np.int64)
        true_labels = subset_df["label"].to_numpy(dtype=np.int64)

        for method in methods:
            if method == "random":
                saliency_maps = np.stack(
                    [generate_random_saliency((image_size, image_size), seed=seed + idx) for idx in range(n_images)],
                    axis=0,
                )
                target_classes = expected_preds
            else:
                archive_path = maps_dir / f"{model_key}_{method}.npz"
                with np.load(archive_path, allow_pickle=False) as archive:
                    saliency_maps = archive["maps"][:n_images]
                    target_classes = archive["target_classes"][:n_images]
                    archive_paths = list(archive["image_paths"][:n_images])

                assert archive_paths == list(subset_df["image_path"]), (
                    f"Caminhos desalinhados em {archive_path}"
                )
                assert (target_classes == expected_preds).all(), (
                    f"Classes previstas desalinhadas em {archive_path}"
                )

            for baseline in baselines:
                del_curves = np.zeros((n_images, steps + 1), dtype=np.float64)
                ins_curves = np.zeros((n_images, steps + 1), dtype=np.float64)
                fractions_ref: np.ndarray | None = None

                group_del_aucs: list[float] = []
                group_ins_aucs: list[float] = []
                group_diffs: list[float] = []
                group_orig_probs: list[float] = []
                group_base_probs: list[float] = []

                desc = f"E3 {model_key}/{method} ({baseline})"
                for idx in tqdm(range(n_images), desc=desc, leave=False):
                    target = int(target_classes[idx])
                    res = compute_deletion_insertion_auc(
                        model=wrapper,
                        image=images_rgb[idx],
                        saliency_map=saliency_maps[idx],
                        target_class=target,
                        baseline=baseline,
                        steps=steps,
                        blur_kernel_size=blur_kernel_size,
                        blur_sigma=blur_sigma,
                        imagenet_mean=imagenet_mean,
                        imagenet_std=imagenet_std,
                        batch_size=64,
                        seed=seed + idx,
                        return_curves=True,
                    )
                    assert abs(res["orig_prob"] - float(expected_confs[idx])) < 1.0e-3, (
                        f"Divergencia orig_prob em {model_key}/{method} idx={idx}: "
                        f"{res['orig_prob']:.5f} vs E1={expected_confs[idx]:.5f}"
                    )

                    del_curves[idx] = res["deletion_curve"]
                    ins_curves[idx] = res["insertion_curve"]
                    if fractions_ref is None:
                        fractions_ref = res["fractions"]

                    group_del_aucs.append(float(res["deletion_auc"]))
                    group_ins_aucs.append(float(res["insertion_auc"]))
                    group_diffs.append(float(res["auc_diff"]))
                    group_orig_probs.append(float(res["orig_prob"]))
                    group_base_probs.append(float(res["base_prob"]))

                    per_image_rows.append(
                        {
                            "model": model_key,
                            "method": method,
                            "baseline": baseline,
                            "image_idx": idx,
                            "image_path": str(subset_df.iloc[idx]["image_path"]),
                            "true_label": int(true_labels[idx]),
                            "target_class": target,
                            "is_correct": int(expected_correct[idx]),
                            "orig_prob": float(res["orig_prob"]),
                            "base_prob": float(res["base_prob"]),
                            "deletion_auc": float(res["deletion_auc"]),
                            "insertion_auc": float(res["insertion_auc"]),
                            "auc_diff": float(res["auc_diff"]),
                        }
                    )

                assert fractions_ref is not None
                del_mean = del_curves.mean(axis=0)
                del_std = del_curves.std(axis=0)
                ins_mean = ins_curves.mean(axis=0)
                ins_std = ins_curves.std(axis=0)

                for step_idx in range(steps + 1):
                    curves_rows.append(
                        {
                            "model": model_key,
                            "method": method,
                            "baseline": baseline,
                            "step": step_idx,
                            "fraction": float(fractions_ref[step_idx]),
                            "deletion_mean": float(del_mean[step_idx]),
                            "deletion_std": float(del_std[step_idx]),
                            "insertion_mean": float(ins_mean[step_idx]),
                            "insertion_std": float(ins_std[step_idx]),
                        }
                    )

                del_arr = np.asarray(group_del_aucs, dtype=np.float64)
                ins_arr = np.asarray(group_ins_aucs, dtype=np.float64)
                diff_arr = np.asarray(group_diffs, dtype=np.float64)
                orig_arr = np.asarray(group_orig_probs, dtype=np.float64)
                base_arr = np.asarray(group_base_probs, dtype=np.float64)

                summary_row = {
                    "model": model_key,
                    "method": method,
                    "baseline": baseline,
                    "n_images": n_images,
                    "deletion_auc_mean": float(del_arr.mean()),
                    "deletion_auc_std": float(del_arr.std()),
                    "deletion_auc_median": float(np.median(del_arr)),
                    "insertion_auc_mean": float(ins_arr.mean()),
                    "insertion_auc_std": float(ins_arr.std()),
                    "insertion_auc_median": float(np.median(ins_arr)),
                    "auc_diff_mean": float(diff_arr.mean()),
                    "auc_diff_std": float(diff_arr.std()),
                    "orig_prob_mean": float(orig_arr.mean()),
                    "base_prob_mean": float(base_arr.mean()),
                }
                summary_rows.append(summary_row)
                print(
                    f"[E3] {model_key}/{method} ({baseline}): "
                    f"Del={summary_row['deletion_auc_mean']:.4f}+/-{summary_row['deletion_auc_std']:.4f} | "
                    f"Ins={summary_row['insertion_auc_mean']:.4f}+/-{summary_row['insertion_auc_std']:.4f} | "
                    f"Diff={summary_row['auc_diff_mean']:.4f}"
                )

    elapsed_total = time.perf_counter() - t_start
    print(f"[E3] Concluido em {elapsed_total:.2f}s ({n_images} imagens)")

    if limit is not None:
        print("[E3] Modo --limit ativo: resultados completos em disco nao foram sobrescritos.")
        return

    per_img_df = pd.DataFrame(per_image_rows)
    curves_df = pd.DataFrame(curves_rows)
    summary_df = pd.DataFrame(summary_rows)
    wilcoxon_df = _compute_e3_wilcoxon(per_img_df, baselines)

    per_img_df.to_csv(metrics_dir / "e3_faithfulness_per_image.csv", index=False)
    curves_df.to_csv(metrics_dir / "e3_faithfulness_curves.csv", index=False)
    summary_df.to_csv(metrics_dir / "e3_faithfulness_summary.csv", index=False)
    wilcoxon_df.to_csv(metrics_dir / "e3_wilcoxon_tests.csv", index=False)

    summary_payload = {
        "seed": seed,
        "device": str(device),
        "n_images": n_images,
        "steps": steps,
        "baselines": baselines,
        "blur_kernel_size": blur_kernel_size,
        "blur_sigma": blur_sigma,
        "elapsed_sec": float(elapsed_total),
        "summary": summary_rows,
        "wilcoxon_tests": wilcoxon_df.to_dict(orient="records"),
    }
    with (metrics_dir / "e3_faithfulness_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    for baseline in baselines:
        _plot_e3_curves_for_baseline(curves_df, summary_df, baseline, n_images, figures_dir)
    _plot_e3_auc_comparison(summary_df, n_images, figures_dir)


def main() -> None:
    """Ponto de entrada para avaliacao quantitativa."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.experiment == "e3":
        run_e3(config, device, limit=args.limit)
        return
    if args.experiment in {"e4", "e5", "all"}:
        raise NotImplementedError(f"Experimento {args.experiment} ainda nao implementado.")


if __name__ == "__main__":
    main()
