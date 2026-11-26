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
from scipy.stats import pearsonr, spearmanr, wilcoxon
from tqdm import tqdm

from src.config import load_config, resolve_path, set_seed
from src.metrics.faithfulness import compute_deletion_insertion_auc, generate_random_saliency
from src.metrics.graph_rewiring import measure_edge_stability
from src.metrics.robustness import add_gaussian_noise, rgb_to_normalized_tensor
from src.metrics.sanity import CASCADING_STAGES, compare_maps, randomize_stage_inplace
from src.models.deit_tiny import load_deit_tiny
from src.models.vig_ti import load_vig_ti
from src.xai.gradcam import compute_gradcam
from src.xai.ig import compute_integrated_gradients
from src.xai.perturbations import compute_lime, compute_partition_shap
from src.xai.rollout import compute_attention_rollout
from src.xai.utils import get_predicted_class, load_image_pair

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
        description="Avalia fidelidade (E3), sanity checks (E4), rewiring (E5) e robustez a ruido (E6).",
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
        choices=["e3", "e4", "e5", "e6", "all"],
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


STAGE_SHORT_LABELS = {
    "original": "Orig",
    "head": "Head",
    "blocks_9_11": "Blk 9-11",
    "blocks_6_11": "Blk 6-11",
    "blocks_3_11": "Blk 3-11",
    "blocks_0_11": "Blk 0-11",
    "all_layers": "All",
}


def _select_stratified_indices(subset_df: pd.DataFrame, per_class: int) -> list[int]:
    indices: list[int] = []
    for _, grp in subset_df.groupby("label", sort=True):
        indices.extend(int(i) for i in grp.index[:per_class])
    return sorted(indices)


def _explain_for_sanity(
    method: str,
    wrapper: torch.nn.Module,
    image_rgb: np.ndarray,
    image_tensor: torch.Tensor,
    target_class: int,
    config: dict[str, Any],
    device: torch.device,
    per_image_seed: int,
) -> np.ndarray:
    xai_cfg = config["xai"]
    data_cfg = config["data"]

    if method == "gradcam":
        return compute_gradcam(wrapper, image_tensor, target_class=target_class)
    if method == "rollout":
        rollout_cfg = xai_cfg["rollout"]
        return compute_attention_rollout(
            wrapper,
            image_tensor,
            discard_ratio=float(rollout_cfg["discard_ratio"]),
            head_fusion=str(rollout_cfg["head_fusion"]),
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
        )
    raise ValueError(f"Metodo invalido para E4: {method}")


def _compute_e4_wilcoxon(per_img_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stages = ["head", "all_layers"]
    metrics = ["spearman", "ssim"]

    for stage_name in stages:
        for metric in metrics:
            for method in ["gradcam", "lime", "partition_shap", "integrated_gradients"]:
                mask_a = (
                    (per_img_df["model"] == "vig_ti")
                    & (per_img_df["method"] == method)
                    & (per_img_df["stage_name"] == stage_name)
                )
                mask_b = (
                    (per_img_df["model"] == "deit_tiny")
                    & (per_img_df["method"] == method)
                    & (per_img_df["stage_name"] == stage_name)
                )
                rows.append(
                    _paired_wilcoxon(
                        per_img_df,
                        "vig_vs_deit_sanity",
                        mask_a,
                        mask_b,
                        f"vig_ti/{method}",
                        f"deit_tiny/{method}",
                        stage_name,
                        metric,
                    )
                )

            rows.append(
                _paired_wilcoxon(
                    per_img_df,
                    "deit_gradcam_vs_rollout_sanity",
                    (per_img_df["model"] == "deit_tiny")
                    & (per_img_df["method"] == "gradcam")
                    & (per_img_df["stage_name"] == stage_name),
                    (per_img_df["model"] == "deit_tiny")
                    & (per_img_df["method"] == "rollout")
                    & (per_img_df["stage_name"] == stage_name),
                    "deit_tiny/gradcam",
                    "deit_tiny/rollout",
                    stage_name,
                    metric,
                )
            )

    return pd.DataFrame(rows)


def _plot_e4_sanity_curves(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)

    x_ticks = np.arange(len(CASCADING_STAGES), dtype=np.float64)
    x_labels = [STAGE_SHORT_LABELS[s] for s in CASCADING_STAGES]

    panels = [
        (0, 0, "vig_ti", "spearman", "ViG-Ti: Spearman vs. Aleatorizacao em Cascata"),
        (0, 1, "deit_tiny", "spearman", "DeiT-Tiny: Spearman vs. Aleatorizacao em Cascata"),
        (1, 0, "vig_ti", "ssim", "ViG-Ti: SSIM vs. Aleatorizacao em Cascata"),
        (1, 1, "deit_tiny", "ssim", "DeiT-Tiny: SSIM vs. Aleatorizacao em Cascata"),
    ]

    for row_idx, col_idx, model_key, metric, title in panels:
        ax = axes[row_idx, col_idx]
        for method in METHODS_BY_MODEL[model_key]:
            sub = summary_df[
                (summary_df["model"] == model_key) & (summary_df["method"] == method)
            ].sort_values("stage_idx")
            if sub.empty:
                continue

            means = sub[f"{metric}_mean"].to_numpy()
            stds = sub[f"{metric}_std"].to_numpy()
            ns = sub["n_images"].to_numpy(dtype=np.float64)
            sems = stds / np.sqrt(ns)
            n_val = int(ns[0])

            color = METHOD_COLORS[method]
            label = f"{METHOD_DISPLAY[method]} (n={n_val})"
            ax.plot(x_ticks, means, marker="o", markersize=4.5, linewidth=2.0, color=color, label=label)
            ax.fill_between(
                x_ticks,
                np.clip(means - sems, -1.0, 1.0),
                np.clip(means + sems, -1.0, 1.0),
                color=color,
                alpha=0.14,
            )

        if metric == "spearman":
            ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.set_ylim(-0.30, 1.08)
            ax.set_ylabel("Correlacao de Spearman")
        else:
            ax.set_ylim(-0.05, 1.08)
            ax.set_ylabel("SSIM")

        ax.set_title(title)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=15, ha="right")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)

    fig.suptitle(
        "Sanity Checks por Aleatorizacao em Cascata (E4: Adebayo et al., +/- 1 SEM)",
        fontsize=12,
    )
    fig.savefig(figures_dir / "e4_sanity_curves.pdf", dpi=300)
    fig.savefig(figures_dir / "e4_sanity_curves.png", dpi=300)
    plt.close(fig)


def run_e4(config: dict[str, Any], device: torch.device, limit: int | None = None) -> None:
    """Executa o experimento E4 de sanity checks por aleatorizacao em cascata."""
    t_start = time.perf_counter()
    seed = int(config["seed"])
    splits_dir = resolve_path(config["data"]["splits_dir"])
    maps_dir = resolve_path(config["xai"]["output_dir"])
    eval_cfg = config["evaluation"]
    sanity_cfg = eval_cfg["sanity"]
    metrics_dir = resolve_path(eval_cfg["metrics_dir"])
    figures_dir = resolve_path(eval_cfg["figures_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    subset_df = pd.read_csv(splits_dir / "xai_subset.csv")
    pert_per_class = int(sanity_cfg.get("perturbation_per_class", 2))

    if limit is not None:
        fast_indices = list(range(min(limit, len(subset_df))))
        pert_indices = list(range(min(limit, len(subset_df))))
    else:
        fast_indices = list(range(len(subset_df)))
        pert_indices = _select_stratified_indices(subset_df, pert_per_class)

    images_rgb: list[np.ndarray] = []
    images_tensor: list[torch.Tensor] = []
    for path in subset_df["image_path"]:
        rgb, ten = load_image_pair(path, config)
        images_rgb.append(rgb)
        images_tensor.append(ten)

    true_labels = subset_df["label"].to_numpy(dtype=np.int64)
    image_paths = subset_df["image_path"].tolist()
    per_image_rows: list[dict[str, Any]] = []

    for model_key in ["vig_ti", "deit_tiny"]:
        methods = list(METHODS_BY_MODEL[model_key])
        method_indices = {
            m: (pert_indices if m in {"lime", "partition_shap"} else fast_indices)
            for m in methods
        }

        orig_maps: dict[str, np.ndarray] = {}
        orig_targets: dict[str, np.ndarray] = {}
        for method in methods:
            archive_path = maps_dir / f"{model_key}_{method}.npz"
            with np.load(archive_path, allow_pickle=False) as archive:
                assert list(archive["image_paths"]) == image_paths, (
                    f"Caminhos desalinhados em {archive_path}"
                )
                orig_maps[method] = archive["maps"]
                orig_targets[method] = archive["target_classes"]

            for idx in method_indices[method]:
                per_image_rows.append(
                    {
                        "model": model_key,
                        "method": method,
                        "stage_idx": 0,
                        "stage_name": "original",
                        "image_idx": idx,
                        "image_path": image_paths[idx],
                        "true_label": int(true_labels[idx]),
                        "target_class": int(orig_targets[method][idx]),
                        "spearman": 1.0,
                        "abs_spearman": 1.0,
                        "ssim": 1.0,
                        "is_constant_map": False,
                    }
                )

        wrapper = _load_finetuned_model(model_key, config, device)
        model_seed_offset = 0 if model_key == "vig_ti" else 100

        for stage_idx, stage_name in enumerate(CASCADING_STAGES[1:], start=1):
            stage_seed = seed + 1000 * stage_idx + model_seed_offset
            randomize_stage_inplace(wrapper, model_key, stage_name, seed=stage_seed)

            for method in methods:
                idxs = method_indices[method]
                desc = f"E4 {model_key}/{method} ({stage_name})"
                stage_rhos: list[float] = []
                stage_ssims: list[float] = []

                for idx in tqdm(idxs, desc=desc, leave=False):
                    per_img_seed = seed + idx
                    set_seed(per_img_seed)
                    target = int(orig_targets[method][idx])
                    rand_map = _explain_for_sanity(
                        method=method,
                        wrapper=wrapper,
                        image_rgb=images_rgb[idx],
                        image_tensor=images_tensor[idx].to(device),
                        target_class=target,
                        config=config,
                        device=device,
                        per_image_seed=per_img_seed,
                    )
                    comp = compare_maps(orig_maps[method][idx], rand_map)
                    stage_rhos.append(float(comp["spearman"]))
                    stage_ssims.append(float(comp["ssim"]))

                    per_image_rows.append(
                        {
                            "model": model_key,
                            "method": method,
                            "stage_idx": stage_idx,
                            "stage_name": stage_name,
                            "image_idx": idx,
                            "image_path": image_paths[idx],
                            "true_label": int(true_labels[idx]),
                            "target_class": target,
                            "spearman": float(comp["spearman"]),
                            "abs_spearman": float(comp["abs_spearman"]),
                            "ssim": float(comp["ssim"]),
                            "is_constant_map": bool(comp["is_constant_map"]),
                        }
                    )

                print(
                    f"[E4] {model_key}/{method} ({stage_name}, n={len(idxs)}): "
                    f"Spearman={np.mean(stage_rhos):.4f}+/-{np.std(stage_rhos):.4f} | "
                    f"SSIM={np.mean(stage_ssims):.4f}+/-{np.std(stage_ssims):.4f}"
                )

    elapsed_total = time.perf_counter() - t_start
    print(f"[E4] Concluido em {elapsed_total:.2f}s")

    if limit is not None:
        print("[E4] Modo --limit ativo: resultados completos em disco nao foram sobrescritos.")
        return

    per_img_df = pd.DataFrame(per_image_rows)
    summary_rows: list[dict[str, Any]] = []
    for (model_key, method, stage_idx, stage_name), grp in per_img_df.groupby(
        ["model", "method", "stage_idx", "stage_name"], sort=False
    ):
        rhos = grp["spearman"].to_numpy(dtype=np.float64)
        abs_rhos = grp["abs_spearman"].to_numpy(dtype=np.float64)
        ssims = grp["ssim"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "model": model_key,
                "method": method,
                "stage_idx": int(stage_idx),
                "stage_name": stage_name,
                "n_images": int(len(grp)),
                "spearman_mean": float(rhos.mean()),
                "spearman_std": float(rhos.std()),
                "spearman_median": float(np.median(rhos)),
                "abs_spearman_mean": float(abs_rhos.mean()),
                "ssim_mean": float(ssims.mean()),
                "ssim_std": float(ssims.std()),
                "ssim_median": float(np.median(ssims)),
                "constant_maps_count": int(grp["is_constant_map"].sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    wilcoxon_df = _compute_e4_wilcoxon(per_img_df)

    per_img_df.to_csv(metrics_dir / "e4_sanity_per_image.csv", index=False)
    summary_df.to_csv(metrics_dir / "e4_sanity_summary.csv", index=False)
    wilcoxon_df.to_csv(metrics_dir / "e4_wilcoxon_tests.csv", index=False)

    summary_payload = {
        "seed": seed,
        "device": str(device),
        "n_fast": len(fast_indices),
        "n_perturbation": len(pert_indices),
        "perturbation_per_class": pert_per_class,
        "stages": CASCADING_STAGES,
        "elapsed_sec": float(elapsed_total),
        "summary": summary_rows,
        "wilcoxon_tests": wilcoxon_df.to_dict(orient="records"),
    }
    with (metrics_dir / "e4_sanity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    _plot_e4_sanity_curves(summary_df, figures_dir)


def _compute_e5_wilcoxon(per_img_df: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        for metric in ["rewiring_auc", "jaccard_auc"]:
            for method in METHODS_BY_MODEL["vig_ti"]:
                rows.append(
                    _paired_wilcoxon(
                        per_img_df,
                        "rewiring_vs_random",
                        (per_img_df["method"] == method) & (per_img_df["baseline"] == baseline),
                        (per_img_df["method"] == "random") & (per_img_df["baseline"] == baseline),
                        f"vig_ti/{method}",
                        "vig_ti/random",
                        baseline,
                        metric,
                    )
                )
    return pd.DataFrame(rows)


def _plot_e5_graph_rewiring(
    curves_df: pd.DataFrame,
    layerwise_df: pd.DataFrame,
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
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)

    vig_methods = METHODS_BY_MODEL["vig_ti"] + ["random"]
    sem_scale = 1.0 / np.sqrt(float(n_images))

    for col_idx, (baseline, b_title) in enumerate(
        [("blur", "Desfoque Gaussiano (blur)"), ("black", "Imagem Preta (black)")]
    ):
        ax = axes[0, col_idx]
        for method in vig_methods:
            sub = curves_df[
                (curves_df["method"] == method) & (curves_df["baseline"] == baseline)
            ].sort_values("step")
            frac = sub["fraction"].to_numpy() * 100.0
            mean_rew = sub["rewiring_mean"].to_numpy() * 100.0
            sem_rew = sub["rewiring_std"].to_numpy() * sem_scale * 100.0
            auc_rew = float(np.trapezoid(sub["rewiring_mean"].to_numpy(), sub["fraction"].to_numpy()))

            color = METHOD_COLORS[method]
            ls = "--" if method == "random" else "-"
            lw = 1.6 if method == "random" else 2.0
            ax.plot(
                frac,
                mean_rew,
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=f"{METHOD_DISPLAY[method]} (AUC={auc_rew:.3f})",
            )
            ax.fill_between(
                frac,
                np.clip(mean_rew - sem_rew, 0.0, 100.0),
                np.clip(mean_rew + sem_rew, 0.0, 100.0),
                color=color,
                alpha=0.14,
            )

        ax.set_title(f"Rewiring de Arestas k-NN (Bloco 11): {b_title}")
        ax.set_xlabel("Pixels removidos (%)")
        ax.set_ylabel("Arestas alteradas (%)")
        ax.set_xlim(0.0, 100.0)
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", frameon=True)

    # Bottom-left: Trajetoria Media Queda de Confianca vs Rewiring
    ax_traj = axes[1, 0]
    for method in ["gradcam", "partition_shap", "lime", "random"]:
        for baseline, ls, tag in [("blur", "-", "blur"), ("black", ":", "black")]:
            sub = curves_df[
                (curves_df["method"] == method) & (curves_df["baseline"] == baseline)
            ].sort_values("step")
            x_rew = sub["rewiring_mean"].to_numpy() * 100.0
            y_drop = sub["prob_drop_mean"].to_numpy()
            ax_traj.plot(
                x_rew,
                y_drop,
                color=METHOD_COLORS[method],
                linestyle=ls,
                linewidth=1.8,
                label=f"{METHOD_DISPLAY[method]} ({tag})",
            )

    ax_traj.set_title("Queda Media de Confianca vs. Rewiring (Bloco 11)")
    ax_traj.set_xlabel("Arestas alteradas (%)")
    ax_traj.set_ylabel("Queda de probabilidade (orig - pert)")
    ax_traj.set_xlim(0.0, 100.0)
    ax_traj.set_ylim(0.0, 0.95)
    ax_traj.grid(True, linestyle=":", alpha=0.5)
    ax_traj.legend(loc="upper left", ncol=2, fontsize=7.2, frameon=True)

    # Bottom-right: Perfil por camada (Blocos 0..11) em 20% de Deletion
    ax_layer = axes[1, 1]
    layers = np.arange(12, dtype=np.int64)
    for method in ["gradcam", "partition_shap", "random"]:
        for baseline, ls, marker, tag in [
            ("blur", "-", "o", "blur"),
            ("black", "--", "s", "black"),
        ]:
            sub = layerwise_df[
                (layerwise_df["method"] == method) & (layerwise_df["baseline"] == baseline)
            ].sort_values("layer_idx")
            vals = sub["rewiring_20_mean"].to_numpy() * 100.0
            ax_layer.plot(
                layers,
                vals,
                color=METHOD_COLORS[method],
                linestyle=ls,
                marker=marker,
                markersize=3.8,
                linewidth=1.8,
                label=f"{METHOD_DISPLAY[method]} ({tag})",
            )

    ax_layer.set_title("Rewiring por Camada Grapher (20% Deletion)")
    ax_layer.set_xlabel("Bloco Grapher (0 a 11)")
    ax_layer.set_ylabel("Arestas alteradas (%)")
    ax_layer.set_xticks(layers)
    ax_layer.set_ylim(0.0, 100.0)
    ax_layer.grid(True, linestyle=":", alpha=0.5)
    ax_layer.legend(loc="upper left", ncol=2, fontsize=7.2, frameon=True)

    fig.suptitle(
        f"Analise Estrutural de Graph Rewiring k-NN na ViG-Ti (E5, N={n_images})",
        fontsize=12,
    )
    fig.savefig(figures_dir / "e5_graph_rewiring.pdf", dpi=300)
    fig.savefig(figures_dir / "e5_graph_rewiring.png", dpi=300)
    plt.close(fig)


def run_e5(config: dict[str, Any], device: torch.device, limit: int | None = None) -> None:
    """Executa o experimento E5 de estabilidade e rewiring das arestas k-NN na ViG-Ti."""
    t_start = time.perf_counter()
    seed = int(config["seed"])
    splits_dir = resolve_path(config["data"]["splits_dir"])
    maps_dir = resolve_path(config["xai"]["output_dir"])
    eval_cfg = config["evaluation"]
    rew_cfg = eval_cfg["graph_rewiring"]
    faith_cfg = eval_cfg["faithfulness"]
    metrics_dir = resolve_path(eval_cfg["metrics_dir"])
    figures_dir = resolve_path(eval_cfg["figures_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    steps = int(rew_cfg["steps"])
    target_layer = int(rew_cfg["target_layer"])
    baselines = [str(b) for b in rew_cfg["baselines"]]
    blur_kernel_size = int(faith_cfg["blur_kernel_size"])
    blur_sigma = float(faith_cfg["blur_sigma"])

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

    wrapper = _load_finetuned_model("vig_ti", config, device)
    expected_preds = e1_preds_df.loc[subset_df["image_path"], "vig_ti_pred"].to_numpy(dtype=np.int64)
    true_labels = subset_df["label"].to_numpy(dtype=np.int64)
    image_paths = subset_df["image_path"].tolist()

    vig_methods = list(METHODS_BY_MODEL["vig_ti"]) + ["random"]
    per_image_rows: list[dict[str, Any]] = []
    curves_rows: list[dict[str, Any]] = []
    layerwise_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for method in vig_methods:
        if method == "random":
            saliency_maps = np.stack(
                [generate_random_saliency((image_size, image_size), seed=seed + idx) for idx in range(n_images)],
                axis=0,
            )
            target_classes = expected_preds
        else:
            archive_path = maps_dir / f"vig_ti_{method}.npz"
            with np.load(archive_path, allow_pickle=False) as archive:
                saliency_maps = archive["maps"][:n_images]
                target_classes = archive["target_classes"][:n_images]
                archive_paths = list(archive["image_paths"][:n_images])
            assert archive_paths == image_paths
            assert (target_classes == expected_preds).all()

        for baseline in baselines:
            probs_mat = np.zeros((n_images, steps + 1), dtype=np.float64)
            drop_mat = np.zeros((n_images, steps + 1), dtype=np.float64)
            ret_mat = np.zeros((n_images, steps + 1), dtype=np.float64)
            rew_mat = np.zeros((n_images, steps + 1), dtype=np.float64)
            jac_mat = np.zeros((n_images, steps + 1), dtype=np.float64)
            fractions_ref: np.ndarray | None = None

            layer_rew20: dict[int, list[float]] = {l_idx: [] for l_idx in range(12)}
            layer_jac20: dict[int, list[float]] = {l_idx: [] for l_idx in range(12)}
            layer_rew50: dict[int, list[float]] = {l_idx: [] for l_idx in range(12)}
            layer_jac50: dict[int, list[float]] = {l_idx: [] for l_idx in range(12)}

            desc = f"E5 vig_ti/{method} ({baseline})"
            for idx in tqdm(range(n_images), desc=desc, leave=False):
                res = measure_edge_stability(
                    model=wrapper,
                    image=images_rgb[idx],
                    saliency_map=saliency_maps[idx],
                    target_class=int(target_classes[idx]),
                    baseline=baseline,
                    steps=steps,
                    target_layer=target_layer,
                    blur_kernel_size=blur_kernel_size,
                    blur_sigma=blur_sigma,
                    imagenet_mean=imagenet_mean,
                    imagenet_std=imagenet_std,
                    seed=seed + idx,
                    capture_all_layers=True,
                )

                probs_mat[idx] = res["probs"]
                drop_mat[idx] = res["prob_drop"]
                ret_mat[idx] = res["retention"]
                rew_mat[idx] = res["rewiring"]
                jac_mat[idx] = res["jaccard_dist"]
                if fractions_ref is None:
                    fractions_ref = res["fractions"]

                for l_idx, l_vals in res["layerwise"].items():
                    layer_rew20[l_idx].append(l_vals["rewiring_20"])
                    layer_jac20[l_idx].append(l_vals["jaccard_20"])
                    layer_rew50[l_idx].append(l_vals["rewiring_50"])
                    layer_jac50[l_idx].append(l_vals["jaccard_50"])

                per_image_rows.append(
                    {
                        "model": "vig_ti",
                        "method": method,
                        "baseline": baseline,
                        "image_idx": idx,
                        "image_path": image_paths[idx],
                        "true_label": int(true_labels[idx]),
                        "target_class": int(target_classes[idx]),
                        "orig_prob": float(res["orig_prob"]),
                        "final_prob": float(res["final_prob"]),
                        "deletion_auc": float(res["deletion_auc"]),
                        "rewiring_auc": float(res["rewiring_auc"]),
                        "jaccard_auc": float(res["jaccard_auc"]),
                        "rewiring_20": float(res["rewiring_20"]),
                        "rewiring_50": float(res["rewiring_50"]),
                        "jaccard_20": float(res["jaccard_20"]),
                        "jaccard_50": float(res["jaccard_50"]),
                        "spearman_rewiring_vs_probdrop": float(res["spearman_rewiring_vs_probdrop"]),
                        "pearson_rewiring_vs_probdrop": float(res["pearson_rewiring_vs_probdrop"]),
                    }
                )

            assert fractions_ref is not None
            mean_prob = probs_mat.mean(axis=0)
            std_prob = probs_mat.std(axis=0)
            mean_drop = drop_mat.mean(axis=0)
            mean_ret = ret_mat.mean(axis=0)
            mean_rew = rew_mat.mean(axis=0)
            std_rew = rew_mat.std(axis=0)
            mean_jac = jac_mat.mean(axis=0)
            std_jac = jac_mat.std(axis=0)

            for s_idx in range(steps + 1):
                curves_rows.append(
                    {
                        "method": method,
                        "baseline": baseline,
                        "step": s_idx,
                        "fraction": float(fractions_ref[s_idx]),
                        "prob_mean": float(mean_prob[s_idx]),
                        "prob_std": float(std_prob[s_idx]),
                        "prob_drop_mean": float(mean_drop[s_idx]),
                        "retention_mean": float(mean_ret[s_idx]),
                        "rewiring_mean": float(mean_rew[s_idx]),
                        "rewiring_std": float(std_rew[s_idx]),
                        "jaccard_mean": float(mean_jac[s_idx]),
                        "jaccard_std": float(std_jac[s_idx]),
                    }
                )

            for l_idx in range(12):
                layerwise_rows.append(
                    {
                        "method": method,
                        "baseline": baseline,
                        "layer_idx": l_idx,
                        "rewiring_20_mean": float(np.mean(layer_rew20[l_idx])),
                        "rewiring_20_std": float(np.std(layer_rew20[l_idx])),
                        "jaccard_20_mean": float(np.mean(layer_jac20[l_idx])),
                        "rewiring_50_mean": float(np.mean(layer_rew50[l_idx])),
                        "rewiring_50_std": float(np.std(layer_rew50[l_idx])),
                        "jaccard_50_mean": float(np.mean(layer_jac50[l_idx])),
                    }
                )

            grp_df = pd.DataFrame(
                [r for r in per_image_rows if r["method"] == method and r["baseline"] == baseline]
            )
            global_sp = float(spearmanr(mean_rew, mean_drop).statistic)
            global_pe = float(pearsonr(mean_rew, mean_drop).statistic)

            summary_row = {
                "model": "vig_ti",
                "method": method,
                "baseline": baseline,
                "n_images": n_images,
                "deletion_auc_mean": float(grp_df["deletion_auc"].mean()),
                "rewiring_auc_mean": float(grp_df["rewiring_auc"].mean()),
                "rewiring_auc_std": float(grp_df["rewiring_auc"].std()),
                "jaccard_auc_mean": float(grp_df["jaccard_auc"].mean()),
                "jaccard_auc_std": float(grp_df["jaccard_auc"].std()),
                "rewiring_20_mean": float(grp_df["rewiring_20"].mean()),
                "rewiring_50_mean": float(grp_df["rewiring_50"].mean()),
                "spearman_img_mean": float(grp_df["spearman_rewiring_vs_probdrop"].mean()),
                "pearson_img_mean": float(grp_df["pearson_rewiring_vs_probdrop"].mean()),
                "global_curve_spearman": global_sp,
                "global_curve_pearson": global_pe,
            }
            summary_rows.append(summary_row)
            print(
                f"[E5] vig_ti/{method} ({baseline}): "
                f"RewAUC={summary_row['rewiring_auc_mean']:.4f}+/-{summary_row['rewiring_auc_std']:.4f} | "
                f"Rew20={summary_row['rewiring_20_mean']:.3f} | "
                f"Rew50={summary_row['rewiring_50_mean']:.3f} | "
                f"Corr(img)={summary_row['spearman_img_mean']:.3f}"
            )

    elapsed_total = time.perf_counter() - t_start
    print(f"[E5] Concluido em {elapsed_total:.2f}s ({n_images} imagens)")

    if limit is not None:
        print("[E5] Modo --limit ativo: resultados completos em disco nao foram sobrescritos.")
        return

    per_img_df = pd.DataFrame(per_image_rows)
    curves_df = pd.DataFrame(curves_rows)
    layerwise_df = pd.DataFrame(layerwise_rows)
    summary_df = pd.DataFrame(summary_rows)
    wilcoxon_df = _compute_e5_wilcoxon(per_img_df, baselines)

    per_img_df.to_csv(metrics_dir / "e5_rewiring_per_image.csv", index=False)
    curves_df.to_csv(metrics_dir / "e5_rewiring_curves.csv", index=False)
    layerwise_df.to_csv(metrics_dir / "e5_rewiring_layerwise.csv", index=False)
    summary_df.to_csv(metrics_dir / "e5_rewiring_summary.csv", index=False)
    wilcoxon_df.to_csv(metrics_dir / "e5_wilcoxon_tests.csv", index=False)

    summary_payload = {
        "seed": seed,
        "device": str(device),
        "n_images": n_images,
        "steps": steps,
        "target_layer": target_layer,
        "baselines": baselines,
        "elapsed_sec": float(elapsed_total),
        "summary": summary_rows,
        "wilcoxon_tests": wilcoxon_df.to_dict(orient="records"),
    }
    with (metrics_dir / "e5_rewiring_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    _plot_e5_graph_rewiring(curves_df, layerwise_df, n_images, figures_dir)


def _compute_e6_wilcoxon(per_img_df: pd.DataFrame, noise_stds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for noise_std in noise_stds:
        std_val = float(noise_std)
        if std_val <= 0.0:
            continue
        std_tag = f"sigma_{std_val:.2f}"
        for metric in ["spearman", "ssim"]:
            for method in ["gradcam", "lime", "partition_shap", "integrated_gradients"]:
                mask_a = (
                    (per_img_df["model"] == "vig_ti")
                    & (per_img_df["method"] == method)
                    & (np.isclose(per_img_df["noise_std"], std_val))
                )
                mask_b = (
                    (per_img_df["model"] == "deit_tiny")
                    & (per_img_df["method"] == method)
                    & (np.isclose(per_img_df["noise_std"], std_val))
                )
                rows.append(
                    _paired_wilcoxon(
                        per_img_df,
                        "vig_vs_deit_robustness",
                        mask_a,
                        mask_b,
                        f"vig_ti/{method}",
                        f"deit_tiny/{method}",
                        std_tag,
                        metric,
                    )
                )

            rows.append(
                _paired_wilcoxon(
                    per_img_df,
                    "deit_gradcam_vs_rollout_robustness",
                    (per_img_df["model"] == "deit_tiny")
                    & (per_img_df["method"] == "gradcam")
                    & (np.isclose(per_img_df["noise_std"], std_val)),
                    (per_img_df["model"] == "deit_tiny")
                    & (per_img_df["method"] == "rollout")
                    & (np.isclose(per_img_df["noise_std"], std_val)),
                    "deit_tiny/gradcam",
                    "deit_tiny/rollout",
                    std_tag,
                    metric,
                )
            )

    return pd.DataFrame(rows)


def _plot_e6_noise_robustness(
    summary_df: pd.DataFrame,
    pred_stability: dict[str, dict[str, dict[str, float]]],
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
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)

    noise_vals = sorted(summary_df["noise_std"].unique().tolist())
    max_sigma_key = f"{max(noise_vals):.2f}"

    panels = [
        (0, 0, "vig_ti", "spearman", "ViG-Ti: Spearman vs. Ruido Gaussiano"),
        (0, 1, "deit_tiny", "spearman", "DeiT-Tiny: Spearman vs. Ruido Gaussiano"),
        (1, 0, "vig_ti", "ssim", "ViG-Ti: SSIM vs. Ruido Gaussiano"),
        (1, 1, "deit_tiny", "ssim", "DeiT-Tiny: SSIM vs. Ruido Gaussiano"),
    ]

    for row_idx, col_idx, model_key, metric, title in panels:
        ax = axes[row_idx, col_idx]
        for method in METHODS_BY_MODEL[model_key]:
            sub = summary_df[
                (summary_df["model"] == model_key) & (summary_df["method"] == method)
            ].sort_values("noise_std")
            if sub.empty:
                continue

            xs = sub["noise_std"].to_numpy(dtype=np.float64)
            means = sub[f"{metric}_mean"].to_numpy(dtype=np.float64)
            stds = sub[f"{metric}_std"].to_numpy(dtype=np.float64)
            ns = sub["n_images"].to_numpy(dtype=np.float64)
            sems = stds / np.sqrt(ns)
            n_val = int(ns[0])

            color = METHOD_COLORS[method]
            label = f"{METHOD_DISPLAY[method]} (n={n_val})"
            ax.plot(xs, means, marker="o", markersize=4.8, linewidth=2.0, color=color, label=label)
            ax.fill_between(
                xs,
                np.clip(means - sems, 0.0, 1.0),
                np.clip(means + sems, 0.0, 1.0),
                color=color,
                alpha=0.14,
            )

        if row_idx == 0:
            flip_rate = pred_stability[model_key][max_sigma_key]["pred_changed_rate"] * 100.0
            ax.text(
                0.02,
                0.04,
                f"Troca de predicao em sigma={max_sigma_key}: {flip_rate:.1f}%",
                transform=ax.transAxes,
                fontsize=8.2,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            )

        ax.set_title(title)
        ax.set_xlabel("Desvio-padrao do ruido gaussiano (sigma)")
        ax.set_ylabel("Correlacao de Spearman" if metric == "spearman" else "SSIM")
        ax.set_xticks(noise_vals)
        ax.set_xticklabels([f"{v:.2f}" for v in noise_vals])
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="lower left" if row_idx == 1 else "upper right", frameon=True)

    fig.suptitle(
        "Robustez dos Explicadores a Ruido Gaussiano de Baixa Magnitude (E6, +/- 1 SEM)",
        fontsize=12,
    )
    fig.savefig(figures_dir / "e6_noise_robustness.pdf", dpi=300)
    fig.savefig(figures_dir / "e6_noise_robustness.png", dpi=300)
    plt.close(fig)


def run_e6(config: dict[str, Any], device: torch.device, limit: int | None = None) -> None:
    """Executa o experimento E6 de robustez a ruido gaussiano na entrada."""
    t_start = time.perf_counter()
    seed = int(config["seed"])
    splits_dir = resolve_path(config["data"]["splits_dir"])
    maps_dir = resolve_path(config["xai"]["output_dir"])
    eval_cfg = config["evaluation"]
    noise_cfg = eval_cfg["noise_robustness"]
    metrics_dir = resolve_path(eval_cfg["metrics_dir"])
    figures_dir = resolve_path(eval_cfg["figures_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    noise_stds = [float(s) for s in noise_cfg["noise_stds"]]
    pert_per_class = int(noise_cfg.get("perturbation_per_class", 2))

    data_cfg = config["data"]
    imagenet_mean = tuple(float(v) for v in data_cfg["imagenet_mean"])
    imagenet_std = tuple(float(v) for v in data_cfg["imagenet_std"])

    subset_df = pd.read_csv(splits_dir / "xai_subset.csv")
    if limit is not None:
        fast_indices = list(range(min(limit, len(subset_df))))
        pert_indices = list(range(min(limit, len(subset_df))))
    else:
        fast_indices = list(range(len(subset_df)))
        pert_indices = _select_stratified_indices(subset_df, pert_per_class)

    e1_preds_df = pd.read_csv(metrics_dir / "e1_test_predictions.csv").set_index("image_path")
    images_rgb = [load_image_pair(p, config)[0] for p in subset_df["image_path"]]
    true_labels = subset_df["label"].to_numpy(dtype=np.int64)
    image_paths = subset_df["image_path"].tolist()

    # Pre-computa entradas ruidosas compartilhadas entre os dois modelos
    noisy_inputs: dict[tuple[float, int], tuple[np.ndarray, torch.Tensor]] = {}
    for std_idx, sigma in enumerate(noise_stds):
        for idx in fast_indices:
            noise_seed = seed + 10000 * (std_idx + 1) + idx
            noisy_rgb = add_gaussian_noise(images_rgb[idx], noise_std=sigma, seed=noise_seed)
            noisy_tensor = rgb_to_normalized_tensor(noisy_rgb, imagenet_mean, imagenet_std)
            noisy_inputs[(sigma, idx)] = (noisy_rgb, noisy_tensor)

    per_image_rows: list[dict[str, Any]] = []
    pred_stability: dict[str, dict[str, dict[str, float]]] = {}

    for model_key in ["vig_ti", "deit_tiny"]:
        methods = list(METHODS_BY_MODEL[model_key])
        method_indices = {
            m: (pert_indices if m in {"lime", "partition_shap"} else fast_indices)
            for m in methods
        }

        expected_preds = e1_preds_df.loc[image_paths, f"{model_key}_pred"].to_numpy(dtype=np.int64)
        expected_confs = e1_preds_df.loc[image_paths, f"{model_key}_conf"].to_numpy(dtype=np.float64)

        orig_maps: dict[str, np.ndarray] = {}
        orig_targets: dict[str, np.ndarray] = {}
        for method in methods:
            archive_path = maps_dir / f"{model_key}_{method}.npz"
            with np.load(archive_path, allow_pickle=False) as archive:
                assert list(archive["image_paths"]) == image_paths, (
                    f"Caminhos desalinhados em {archive_path}"
                )
                orig_maps[method] = archive["maps"]
                orig_targets[method] = archive["target_classes"]
                assert (orig_targets[method] == expected_preds).all(), (
                    f"Classes alvo desalinhadas em {archive_path}"
                )

            for idx in method_indices[method]:
                per_image_rows.append(
                    {
                        "model": model_key,
                        "method": method,
                        "noise_std": 0.0,
                        "image_idx": idx,
                        "image_path": image_paths[idx],
                        "true_label": int(true_labels[idx]),
                        "target_class": int(orig_targets[method][idx]),
                        "noisy_pred": int(orig_targets[method][idx]),
                        "pred_changed": 0,
                        "noisy_target_prob": float(expected_confs[idx]),
                        "noisy_max_conf": float(expected_confs[idx]),
                        "spearman": 1.0,
                        "abs_spearman": 1.0,
                        "ssim": 1.0,
                        "is_constant_map": False,
                    }
                )

        wrapper = _load_finetuned_model(model_key, config, device)
        pred_stability[model_key] = {}

        for sigma in noise_stds:
            sigma_key = f"{sigma:.2f}"
            pred_cache: dict[int, tuple[int, int, float, float]] = {}

            with torch.no_grad():
                for idx in fast_indices:
                    noisy_tensor = noisy_inputs[(sigma, idx)][1].to(device)
                    logits = wrapper(noisy_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    noisy_pred = int(probs.argmax().item())
                    noisy_max_conf = float(probs[noisy_pred].item())
                    orig_target = int(expected_preds[idx])
                    noisy_target_prob = float(probs[orig_target].item())
                    pred_changed = int(noisy_pred != orig_target)
                    pred_cache[idx] = (noisy_pred, pred_changed, noisy_target_prob, noisy_max_conf)

            changed_arr = np.asarray([pred_cache[i][1] for i in fast_indices], dtype=np.float64)
            tprob_arr = np.asarray([pred_cache[i][2] for i in fast_indices], dtype=np.float64)
            pred_stability[model_key][sigma_key] = {
                "n_images": int(len(fast_indices)),
                "pred_changed_count": int(changed_arr.sum()),
                "pred_changed_rate": float(changed_arr.mean()),
                "target_prob_mean": float(tprob_arr.mean()),
                "target_prob_std": float(tprob_arr.std()),
            }
            print(
                f"[E6] {model_key} predicoes (sigma={sigma:.2f}, n={len(fast_indices)}): "
                f"flip={int(changed_arr.sum())}/{len(fast_indices)} ({changed_arr.mean() * 100:.1f}%) | "
                f"P(target)={tprob_arr.mean():.4f}"
            )

            for method in methods:
                idxs = method_indices[method]
                desc = f"E6 {model_key}/{method} (sigma={sigma:.2f})"
                stage_rhos: list[float] = []
                stage_ssims: list[float] = []

                for idx in tqdm(idxs, desc=desc, leave=False):
                    per_img_seed = seed + idx
                    set_seed(per_img_seed)
                    orig_target = int(orig_targets[method][idx])
                    noisy_rgb, noisy_tensor = noisy_inputs[(sigma, idx)]

                    noisy_map = _explain_for_sanity(
                        method=method,
                        wrapper=wrapper,
                        image_rgb=noisy_rgb,
                        image_tensor=noisy_tensor.to(device),
                        target_class=orig_target,
                        config=config,
                        device=device,
                        per_image_seed=per_img_seed,
                    )
                    comp = compare_maps(orig_maps[method][idx], noisy_map)
                    stage_rhos.append(float(comp["spearman"]))
                    stage_ssims.append(float(comp["ssim"]))

                    noisy_pred, pred_changed, noisy_target_prob, noisy_max_conf = pred_cache[idx]
                    per_image_rows.append(
                        {
                            "model": model_key,
                            "method": method,
                            "noise_std": float(sigma),
                            "image_idx": idx,
                            "image_path": image_paths[idx],
                            "true_label": int(true_labels[idx]),
                            "target_class": orig_target,
                            "noisy_pred": noisy_pred,
                            "pred_changed": pred_changed,
                            "noisy_target_prob": noisy_target_prob,
                            "noisy_max_conf": noisy_max_conf,
                            "spearman": float(comp["spearman"]),
                            "abs_spearman": float(comp["abs_spearman"]),
                            "ssim": float(comp["ssim"]),
                            "is_constant_map": bool(comp["is_constant_map"]),
                        }
                    )

                print(
                    f"[E6] {model_key}/{method} (sigma={sigma:.2f}, n={len(idxs)}): "
                    f"Spearman={np.mean(stage_rhos):.4f}+/-{np.std(stage_rhos):.4f} | "
                    f"SSIM={np.mean(stage_ssims):.4f}+/-{np.std(stage_ssims):.4f}"
                )

    elapsed_total = time.perf_counter() - t_start
    print(f"[E6] Concluido em {elapsed_total:.2f}s")

    if limit is not None:
        print("[E6] Modo --limit ativo: resultados completos em disco nao foram sobrescritos.")
        return

    per_img_df = pd.DataFrame(per_image_rows)
    summary_rows: list[dict[str, Any]] = []
    for (model_key, method, noise_std), grp in per_img_df.groupby(
        ["model", "method", "noise_std"], sort=False
    ):
        rhos = grp["spearman"].to_numpy(dtype=np.float64)
        abs_rhos = grp["abs_spearman"].to_numpy(dtype=np.float64)
        ssims = grp["ssim"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "model": model_key,
                "method": method,
                "noise_std": float(noise_std),
                "n_images": int(len(grp)),
                "spearman_mean": float(rhos.mean()),
                "spearman_std": float(rhos.std()),
                "spearman_median": float(np.median(rhos)),
                "abs_spearman_mean": float(abs_rhos.mean()),
                "ssim_mean": float(ssims.mean()),
                "ssim_std": float(ssims.std()),
                "ssim_median": float(np.median(ssims)),
                "pred_changed_rate": float(grp["pred_changed"].mean()),
                "target_prob_mean": float(grp["noisy_target_prob"].mean()),
                "constant_maps_count": int(grp["is_constant_map"].sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    wilcoxon_df = _compute_e6_wilcoxon(per_img_df, noise_stds)

    per_img_df.to_csv(metrics_dir / "e6_robustness_per_image.csv", index=False)
    summary_df.to_csv(metrics_dir / "e6_robustness_summary.csv", index=False)
    wilcoxon_df.to_csv(metrics_dir / "e6_wilcoxon_tests.csv", index=False)

    summary_payload = {
        "seed": seed,
        "device": str(device),
        "n_fast": len(fast_indices),
        "n_perturbation": len(pert_indices),
        "perturbation_per_class": pert_per_class,
        "noise_stds": noise_stds,
        "elapsed_sec": float(elapsed_total),
        "prediction_stability": pred_stability,
        "summary": summary_rows,
        "wilcoxon_tests": wilcoxon_df.to_dict(orient="records"),
    }
    with (metrics_dir / "e6_robustness_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    _plot_e6_noise_robustness(summary_df, pred_stability, figures_dir)


def main() -> None:
    """Ponto de entrada para avaliacao quantitativa."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.experiment in {"e3", "all"}:
        run_e3(config, device, limit=args.limit)
    if args.experiment in {"e4", "all"}:
        run_e4(config, device, limit=args.limit)
    if args.experiment in {"e5", "all"}:
        run_e5(config, device, limit=args.limit)
    if args.experiment == "e6" or (
        args.experiment == "all" and bool(config["evaluation"]["noise_robustness"].get("enabled", False))
    ):
        run_e6(config, device, limit=args.limit)


if __name__ == "__main__":
    main()
