"""Executa ablacoes de hiperparametros, consolida tabela mestra e gera figuras qualitativas."""

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
from src.metrics.faithfulness import compute_deletion_insertion_auc
from src.metrics.sanity import compare_maps
from src.models.deit_tiny import load_deit_tiny
from src.models.vig_ti import load_vig_ti
from src.xai.ig import compute_integrated_gradients
from src.xai.perturbations import compute_lime
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

MODEL_COLORS = {
    "vig_ti": "#1f77b4",
    "deit_tiny": "#d62728",
}


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


def _select_stratified_indices(subset_df: pd.DataFrame, per_class: int) -> list[int]:
    indices: list[int] = []
    for _, grp in subset_df.groupby("label", sort=True):
        indices.extend(int(i) for i in grp.index[:per_class])
    return sorted(indices)

ERROR_CASE_INDICES = [33, 99, 5, 44]
DIVERGENCE_CASE_INDICES = [50, 75, 80]

QUAL_PANEL_SPECS: list[tuple[str, str | None, str | None]] = [
    ("Imagem Original", None, None),
    ("ViG-Ti Grad-CAM", "vig_ti", "gradcam"),
    ("ViG-Ti SHAP", "vig_ti", "partition_shap"),
    ("ViG-Ti IG", "vig_ti", "integrated_gradients"),
    ("DeiT-Tiny Grad-CAM", "deit_tiny", "gradcam"),
    ("DeiT-Tiny Rollout", "deit_tiny", "rollout"),
    ("DeiT-Tiny SHAP", "deit_tiny", "partition_shap"),
]


def parse_args() -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Executa ablacoes (LIME/IG), consolida tabela mestra E1-E6 e gera figuras qualitativas.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Caminho para o arquivo de configuracao YAML.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ablations", "summary", "figures", "all"],
        default="all",
        help="Etapa de analise a executar.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Executa teste rapido nas primeiras N imagens sem sobrescrever saidas finais.",
    )
    return parser.parse_args()


def _overlay_heatmap(image_rgb: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    """Sobrepoe mapa de saliencia normalizado com colormap inferno (padrao E2)."""
    cmap = plt.get_cmap("inferno")
    heat_rgb = cmap(np.clip(saliency, 0.0, 1.0))[..., :3]
    blended = image_rgb * 0.55 + heat_rgb * 0.45
    return np.clip(blended, 0.0, 1.0)


def _plot_ablation_sensitivity(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)

    configs = [
        (0, "lime", "n_segments", "LIME: numero de superpixels (n_segments)"),
        (1, "integrated_gradients", "n_steps", "Integrated Gradients: passos de interpolacao (n_steps)"),
    ]

    for row_idx, method, param_name, method_title in configs:
        blur_sub = summary_df[
            (summary_df["method"] == method) & (summary_df["baseline"] == "blur")
        ].sort_values("param_value")
        param_vals = sorted(blur_sub["param_value"].unique().tolist())
        x_pos = np.arange(len(param_vals), dtype=np.float64)

        # Coluna 0: Deletion e Insertion AUC (blur)
        ax_auc = axes[row_idx, 0]
        for model_key in ["vig_ti", "deit_tiny"]:
            m_sub = blur_sub[blur_sub["model"] == model_key].sort_values("param_value")
            ns = m_sub["n_images"].to_numpy(dtype=np.float64)
            ins_m = m_sub["insertion_auc_mean"].to_numpy(dtype=np.float64)
            ins_sem = m_sub["insertion_auc_std"].to_numpy(dtype=np.float64) / np.sqrt(ns)
            del_m = m_sub["deletion_auc_mean"].to_numpy(dtype=np.float64)
            del_sem = m_sub["deletion_auc_std"].to_numpy(dtype=np.float64) / np.sqrt(ns)
            color = MODEL_COLORS[model_key]
            m_name = MODEL_DISPLAY[model_key]

            ax_auc.errorbar(
                x_pos,
                ins_m,
                yerr=ins_sem,
                marker="o",
                capsize=3,
                linewidth=2.0,
                linestyle="-",
                color=color,
                label=f"{m_name} Insertion",
            )
            ax_auc.errorbar(
                x_pos,
                del_m,
                yerr=del_sem,
                marker="s",
                capsize=3,
                linewidth=2.0,
                linestyle="--",
                color=color,
                alpha=0.82,
                label=f"{m_name} Deletion",
            )

        n_val = int(blur_sub["n_images"].iloc[0])
        ax_auc.set_title(f"{method_title} (Fidelidade blur, n={n_val})")
        ax_auc.set_xlabel(param_name)
        ax_auc.set_ylabel("AUC (+/- 1 SEM)")
        ax_auc.set_xticks(x_pos)
        ax_auc.set_xticklabels([str(int(v)) for v in param_vals])
        ax_auc.set_ylim(0.0, 1.05)
        ax_auc.grid(True, linestyle=":", alpha=0.5)
        ax_auc.legend(loc="best", ncol=2, fontsize=7.6, frameon=True)

        # Coluna 1: Tempo medio por imagem (barras) e Spearman vs padrao (eixo direito)
        ax_time = axes[row_idx, 1]
        ax_rho = ax_time.twinx()
        bar_w = 0.34
        max_t = float(blur_sub["runtime_sec_mean"].max())

        for m_idx, model_key in enumerate(["vig_ti", "deit_tiny"]):
            m_sub = blur_sub[blur_sub["model"] == model_key].sort_values("param_value")
            t_mean = m_sub["runtime_sec_mean"].to_numpy(dtype=np.float64)
            rho_mean = m_sub["spearman_vs_default_mean"].to_numpy(dtype=np.float64)
            color = MODEL_COLORS[model_key]
            m_name = MODEL_DISPLAY[model_key]
            offset = (-bar_w / 2.0) if m_idx == 0 else (bar_w / 2.0)

            bars = ax_time.bar(
                x_pos + offset,
                t_mean,
                width=bar_w,
                color=color,
                alpha=0.35,
                edgecolor=color,
                linewidth=1.0,
                label=f"{m_name} Tempo (s)",
            )
            for bar in bars:
                h = bar.get_height()
                if max_t * 0.20 <= h < max_t * 0.90:
                    ax_time.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        h * 0.92,
                        f"{h:.2f}s",
                        ha="center",
                        va="top",
                        fontsize=7.3,
                        fontweight="bold",
                    )
                else:
                    ax_time.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        h + max_t * 0.015,
                        f"{h:.2f}s",
                        ha="center",
                        va="bottom",
                        fontsize=7.3,
                        fontweight="bold" if h >= max_t * 0.90 else "normal",
                    )

            ax_rho.plot(
                x_pos,
                rho_mean,
                marker="o",
                markersize=5.0,
                linewidth=2.0,
                color=color,
                label=f"{m_name} Spearman",
            )

        ax_time.set_title(f"{method_title} (Custo e Estabilidade)")
        ax_time.set_xlabel(param_name)
        ax_time.set_ylabel("Tempo medio por imagem (s)")
        ax_rho.set_ylabel("Spearman vs. configuracao padrao")
        ax_time.set_xticks(x_pos)
        ax_time.set_xticklabels([str(int(v)) for v in param_vals])
        ax_time.set_ylim(0.0, max_t * 1.25)
        ax_rho.set_ylim(0.0, 1.08)
        ax_time.grid(True, axis="y", linestyle=":", alpha=0.5)

        h1, l1 = ax_time.get_legend_handles_labels()
        h2, l2 = ax_rho.get_legend_handles_labels()
        ax_time.legend(h1 + h2, l1 + l2, loc="lower left", ncol=2, fontsize=7.4, frameon=True)

    fig.suptitle(
        "Ablacao de Hiperparametros dos Explicadores (LIME n_segments e IG n_steps)",
        fontsize=12,
    )
    fig.savefig(figures_dir / "ablation_sensitivity.pdf", dpi=300)
    fig.savefig(figures_dir / "ablation_sensitivity.png", dpi=300)
    plt.close(fig)


def run_ablations(config: dict[str, Any], device: torch.device, limit: int | None = None) -> None:
    """Executa ablacoes de n_segments (LIME) e n_steps (IG) em subconjunto estratificado."""
    t_start = time.perf_counter()
    seed = int(config["seed"])
    splits_dir = resolve_path(config["data"]["splits_dir"])
    maps_dir = resolve_path(config["xai"]["output_dir"])
    eval_cfg = config["evaluation"]
    metrics_dir = resolve_path(eval_cfg["metrics_dir"])
    figures_dir = resolve_path(eval_cfg["figures_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ablation_cfg = eval_cfg["ablation"]
    per_class = int(ablation_cfg.get("per_class", 2))
    baselines = [str(b) for b in ablation_cfg.get("baselines", ["black", "blur"])]

    faith_cfg = eval_cfg["faithfulness"]
    steps = int(faith_cfg["steps"])
    blur_kernel_size = int(faith_cfg["blur_kernel_size"])
    blur_sigma = float(faith_cfg["blur_sigma"])

    data_cfg = config["data"]
    imagenet_mean = tuple(float(v) for v in data_cfg["imagenet_mean"])
    imagenet_std = tuple(float(v) for v in data_cfg["imagenet_std"])

    lime_cfg = config["xai"]["lime"]
    default_segments = int(lime_cfg["n_segments"])
    ablation_segments = [int(v) for v in lime_cfg["ablation_segments"]]
    lime_num_samples = int(lime_cfg["num_samples"])
    lime_compactness = float(lime_cfg["compactness"])

    ig_cfg = config["xai"]["integrated_gradients"]
    default_ig_steps = int(ig_cfg["n_steps"])
    ablation_ig_steps = [int(v) for v in ig_cfg["ablation_steps"]]
    ig_baseline = str(ig_cfg["baseline"])

    subset_df = pd.read_csv(splits_dir / "xai_subset.csv")
    if limit is not None:
        sel_indices = list(range(min(limit, len(subset_df))))
    else:
        sel_indices = _select_stratified_indices(subset_df, per_class)

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
        wrapper = _load_finetuned_model(model_key, config, device)

        # Carrega mapas padrao do E2 para referencia
        with np.load(maps_dir / f"{model_key}_lime.npz", allow_pickle=False) as arc:
            lime_orig_maps = arc["maps"]
            lime_targets = arc["target_classes"]
            lime_elapsed = arc["elapsed_sec"] if "elapsed_sec" in arc else None
            assert list(arc["image_paths"]) == image_paths

        with np.load(maps_dir / f"{model_key}_integrated_gradients.npz", allow_pickle=False) as arc:
            ig_orig_maps = arc["maps"]
            ig_targets = arc["target_classes"]
            assert list(arc["image_paths"]) == image_paths

        # 1. Ablacao LIME (n_segments)
        for n_seg in ablation_segments:
            desc = f"Ablacao {model_key}/lime (n_segments={n_seg})"
            for idx in tqdm(sel_indices, desc=desc, leave=False):
                per_img_seed = seed + idx
                target = int(lime_targets[idx])
                orig_map = lime_orig_maps[idx]

                if n_seg == default_segments and lime_elapsed is not None:
                    sal_map = orig_map
                    elapsed = float(lime_elapsed[idx])
                else:
                    set_seed(per_img_seed)
                    t0 = time.perf_counter()
                    sal_map = compute_lime(
                        wrapper,
                        images_rgb[idx],
                        target_class=target,
                        config=config,
                        device=device,
                        num_samples=lime_num_samples,
                        n_segments=n_seg,
                        compactness=lime_compactness,
                        seed=per_img_seed,
                    )
                    elapsed = time.perf_counter() - t0

                comp = compare_maps(orig_map, sal_map)
                for baseline in baselines:
                    res = compute_deletion_insertion_auc(
                        model=wrapper,
                        image=images_rgb[idx],
                        saliency_map=sal_map,
                        target_class=target,
                        baseline=baseline,
                        steps=steps,
                        blur_kernel_size=blur_kernel_size,
                        blur_sigma=blur_sigma,
                        imagenet_mean=imagenet_mean,
                        imagenet_std=imagenet_std,
                        batch_size=64,
                        seed=per_img_seed,
                        return_curves=False,
                    )
                    per_image_rows.append(
                        {
                            "model": model_key,
                            "method": "lime",
                            "param_name": "n_segments",
                            "param_value": int(n_seg),
                            "is_default": int(n_seg == default_segments),
                            "baseline": baseline,
                            "image_idx": idx,
                            "image_path": image_paths[idx],
                            "true_label": int(true_labels[idx]),
                            "target_class": target,
                            "deletion_auc": float(res["deletion_auc"]),
                            "insertion_auc": float(res["insertion_auc"]),
                            "auc_diff": float(res["auc_diff"]),
                            "spearman_vs_default": float(comp["spearman"]),
                            "ssim_vs_default": float(comp["ssim"]),
                            "runtime_sec": float(elapsed),
                        }
                    )

        # 2. Ablacao Integrated Gradients (n_steps)
        for n_steps in ablation_ig_steps:
            desc = f"Ablacao {model_key}/ig (n_steps={n_steps})"
            for idx in tqdm(sel_indices, desc=desc, leave=False):
                per_img_seed = seed + idx
                target = int(ig_targets[idx])
                orig_map = ig_orig_maps[idx]

                set_seed(per_img_seed)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                sal_map = compute_integrated_gradients(
                    wrapper,
                    images_tensor[idx].to(device),
                    target_class=target,
                    n_steps=n_steps,
                    baseline=ig_baseline,
                    imagenet_mean=imagenet_mean,
                    imagenet_std=imagenet_std,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                comp = compare_maps(orig_map, sal_map)
                for baseline in baselines:
                    res = compute_deletion_insertion_auc(
                        model=wrapper,
                        image=images_rgb[idx],
                        saliency_map=sal_map,
                        target_class=target,
                        baseline=baseline,
                        steps=steps,
                        blur_kernel_size=blur_kernel_size,
                        blur_sigma=blur_sigma,
                        imagenet_mean=imagenet_mean,
                        imagenet_std=imagenet_std,
                        batch_size=64,
                        seed=per_img_seed,
                        return_curves=False,
                    )
                    per_image_rows.append(
                        {
                            "model": model_key,
                            "method": "integrated_gradients",
                            "param_name": "n_steps",
                            "param_value": int(n_steps),
                            "is_default": int(n_steps == default_ig_steps),
                            "baseline": baseline,
                            "image_idx": idx,
                            "image_path": image_paths[idx],
                            "true_label": int(true_labels[idx]),
                            "target_class": target,
                            "deletion_auc": float(res["deletion_auc"]),
                            "insertion_auc": float(res["insertion_auc"]),
                            "auc_diff": float(res["auc_diff"]),
                            "spearman_vs_default": float(comp["spearman"]),
                            "ssim_vs_default": float(comp["ssim"]),
                            "runtime_sec": float(elapsed),
                        }
                    )

    per_img_df = pd.DataFrame(per_image_rows)
    summary_rows: list[dict[str, Any]] = []
    for (model_key, method, param_name, param_value, baseline), grp in per_img_df.groupby(
        ["model", "method", "param_name", "param_value", "baseline"], sort=False
    ):
        row = {
            "model": model_key,
            "method": method,
            "param_name": param_name,
            "param_value": int(param_value),
            "is_default": int(grp["is_default"].iloc[0]),
            "baseline": baseline,
            "n_images": int(len(grp)),
            "deletion_auc_mean": float(grp["deletion_auc"].mean()),
            "deletion_auc_std": float(grp["deletion_auc"].std()),
            "insertion_auc_mean": float(grp["insertion_auc"].mean()),
            "insertion_auc_std": float(grp["insertion_auc"].std()),
            "auc_diff_mean": float(grp["auc_diff"].mean()),
            "spearman_vs_default_mean": float(grp["spearman_vs_default"].mean()),
            "spearman_vs_default_std": float(grp["spearman_vs_default"].std()),
            "ssim_vs_default_mean": float(grp["ssim_vs_default"].mean()),
            "ssim_vs_default_std": float(grp["ssim_vs_default"].std()),
            "runtime_sec_mean": float(grp["runtime_sec"].mean()),
            "runtime_sec_std": float(grp["runtime_sec"].std()),
        }
        summary_rows.append(row)
        if baseline == "blur":
            print(
                f"[Ablacao] {model_key}/{method} {param_name}={param_value} (blur, n={len(grp)}): "
                f"Del={row['deletion_auc_mean']:.4f} | Ins={row['insertion_auc_mean']:.4f} | "
                f"Spearman={row['spearman_vs_default_mean']:.4f} | Tempo={row['runtime_sec_mean']:.3f}s"
            )

    elapsed_total = time.perf_counter() - t_start
    print(f"[Ablacao] Concluido em {elapsed_total:.2f}s (n={len(sel_indices)} imagens)")

    if limit is not None:
        print("[Ablacao] Modo --limit ativo: resultados completos em disco nao foram sobrescritos.")
        return

    summary_df = pd.DataFrame(summary_rows)
    per_img_df.to_csv(metrics_dir / "ablation_per_image.csv", index=False)
    summary_df.to_csv(metrics_dir / "ablation_summary.csv", index=False)

    payload = {
        "seed": seed,
        "device": str(device),
        "n_images": len(sel_indices),
        "per_class": per_class,
        "selected_indices": sel_indices,
        "baselines": baselines,
        "elapsed_sec": float(elapsed_total),
        "summary": summary_rows,
    }
    with (metrics_dir / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    _plot_ablation_sensitivity(summary_df, figures_dir)


def build_master_summary(config: dict[str, Any]) -> None:
    """Consolida metricas de E1 a E6 em tabela mestra e comparacao acertos vs erros."""
    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])

    with (metrics_dir / "e1_test_summary.json").open("r", encoding="utf-8") as handle:
        e1_json = json.load(handle)
    e2_df = pd.read_csv(metrics_dir / "e2_xai_summary.csv")
    e3_df = pd.read_csv(metrics_dir / "e3_faithfulness_summary.csv")
    e3_per_df = pd.read_csv(metrics_dir / "e3_faithfulness_per_image.csv")
    e4_df = pd.read_csv(metrics_dir / "e4_sanity_summary.csv")
    e5_df = pd.read_csv(metrics_dir / "e5_rewiring_summary.csv")
    e6_df = pd.read_csv(metrics_dir / "e6_robustness_summary.csv")
    with (metrics_dir / "e6_robustness_summary.json").open("r", encoding="utf-8") as handle:
        e6_json = json.load(handle)

    e2_idx = e2_df.set_index(["model", "method"])
    e3_idx = e3_df.set_index(["model", "method", "baseline"])
    e4_idx = e4_df.set_index(["model", "method", "stage_name"])
    e5_idx = e5_df.set_index(["model", "method", "baseline"])
    e6_df = e6_df.copy()
    e6_df["std_tag"] = e6_df["noise_std"].map(lambda v: f"{float(v):.2f}")
    e6_idx = e6_df.set_index(["model", "method", "std_tag"])

    master_rows: list[dict[str, Any]] = []
    for model_key in ["vig_ti", "deit_tiny"]:
        ordered_methods = list(METHODS_BY_MODEL[model_key]) + ["random"]
        for method in ordered_methods:
            e2_time = (
                float(e2_idx.loc[(model_key, method), "mean_time_sec"])
                if (model_key, method) in e2_idx.index
                else np.nan
            )

            blur_row = e3_idx.loc[(model_key, method, "blur")]
            black_row = e3_idx.loc[(model_key, method, "black")]

            if (model_key, method, "head") in e4_idx.index:
                e4_head = e4_idx.loc[(model_key, method, "head")]
                e4_all = e4_idx.loc[(model_key, method, "all_layers")]
                e4_head_sp = float(e4_head["spearman_mean"])
                e4_head_ss = float(e4_head["ssim_mean"])
                e4_all_sp = float(e4_all["spearman_mean"])
                e4_all_ss = float(e4_all["ssim_mean"])
            else:
                e4_head_sp = np.nan
                e4_head_ss = np.nan
                e4_all_sp = np.nan
                e4_all_ss = np.nan

            if (model_key, method, "blur") in e5_idx.index:
                e5_blur_rew = float(e5_idx.loc[(model_key, method, "blur"), "rewiring_auc_mean"])
                e5_black_rew = float(e5_idx.loc[(model_key, method, "black"), "rewiring_auc_mean"])
            else:
                e5_blur_rew = np.nan
                e5_black_rew = np.nan

            if (model_key, method, "0.02") in e6_idx.index:
                e6_002_sp = float(e6_idx.loc[(model_key, method, "0.02"), "spearman_mean"])
                e6_005_sp = float(e6_idx.loc[(model_key, method, "0.05"), "spearman_mean"])
                e6_010_sp = float(e6_idx.loc[(model_key, method, "0.10"), "spearman_mean"])
                e6_005_ss = float(e6_idx.loc[(model_key, method, "0.05"), "ssim_mean"])
            else:
                e6_002_sp = np.nan
                e6_005_sp = np.nan
                e6_010_sp = np.nan
                e6_005_ss = np.nan

            master_rows.append(
                {
                    "model": model_key,
                    "method": method,
                    "e2_mean_time_sec": e2_time,
                    "e3_blur_del_mean": float(blur_row["deletion_auc_mean"]),
                    "e3_blur_del_std": float(blur_row["deletion_auc_std"]),
                    "e3_blur_ins_mean": float(blur_row["insertion_auc_mean"]),
                    "e3_blur_ins_std": float(blur_row["insertion_auc_std"]),
                    "e3_black_del_mean": float(black_row["deletion_auc_mean"]),
                    "e3_black_del_std": float(black_row["deletion_auc_std"]),
                    "e3_black_ins_mean": float(black_row["insertion_auc_mean"]),
                    "e3_black_ins_std": float(black_row["insertion_auc_std"]),
                    "e4_head_spearman": e4_head_sp,
                    "e4_head_ssim": e4_head_ss,
                    "e4_all_spearman": e4_all_sp,
                    "e4_all_ssim": e4_all_ss,
                    "e5_blur_rewiring_auc": e5_blur_rew,
                    "e5_black_rewiring_auc": e5_black_rew,
                    "e6_std002_spearman": e6_002_sp,
                    "e6_std005_spearman": e6_005_sp,
                    "e6_std010_spearman": e6_010_sp,
                    "e6_std005_ssim": e6_005_ss,
                }
            )

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(metrics_dir / "summary_master_table.csv", index=False)

    # Analise acertos vs erros (E3 per image agrupado por is_correct)
    err_rows: list[dict[str, Any]] = []
    for (model_key, method, baseline, is_correct), grp in e3_per_df.groupby(
        ["model", "method", "baseline", "is_correct"], sort=False
    ):
        err_rows.append(
            {
                "model": model_key,
                "method": method,
                "baseline": baseline,
                "is_correct": int(is_correct),
                "n_images": int(len(grp)),
                "orig_prob_mean": float(grp["orig_prob"].mean()),
                "deletion_auc_mean": float(grp["deletion_auc"].mean()),
                "deletion_auc_std": float(grp["deletion_auc"].std()),
                "insertion_auc_mean": float(grp["insertion_auc"].mean()),
                "insertion_auc_std": float(grp["insertion_auc"].std()),
            }
        )
    err_df = pd.DataFrame(err_rows)
    err_df.to_csv(metrics_dir / "analysis_error_vs_correct.csv", index=False)

    master_payload = {
        "e1_models": e1_json.get("models", {}),
        "e6_prediction_stability": e6_json.get("prediction_stability", {}),
        "master_table": master_rows,
        "error_vs_correct": err_rows,
    }
    with (metrics_dir / "summary_master_table.json").open("w", encoding="utf-8") as handle:
        json.dump(master_payload, handle, indent=2, ensure_ascii=False)

    print("[Summary] Tabela mestra e analise acertos vs erros gravadas em results/metrics/.")


def generate_qualitative_figures(config: dict[str, Any]) -> None:
    """Gera figuras qualitativas de acertos vs erros e plausibilidade vs fidelidade."""
    splits_dir = resolve_path(config["data"]["splits_dir"])
    maps_dir = resolve_path(config["xai"]["output_dir"])
    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])
    figures_dir = resolve_path(config["evaluation"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    subset_df = pd.read_csv(splits_dir / "xai_subset.csv")
    e1_preds_df = pd.read_csv(metrics_dir / "e1_test_predictions.csv").set_index("image_path")
    e3_per_df = pd.read_csv(metrics_dir / "e3_faithfulness_per_image.csv")
    e4_per_df = pd.read_csv(metrics_dir / "e4_sanity_per_image.csv")
    with (splits_dir / "classes.json").open("r", encoding="utf-8") as handle:
        classes_list = json.load(handle)
    label_to_name = {int(c["label"]): str(c["class_name"]) for c in classes_list}

    e3_blur = e3_per_df[e3_per_df["baseline"] == "blur"].set_index(
        ["image_idx", "model", "method"]
    )
    e4_idx = e4_per_df.set_index(["image_idx", "model", "method", "stage_name"])

    loaded_maps: dict[tuple[str, str], np.ndarray] = {}
    for _, model_key, method in QUAL_PANEL_SPECS:
        if model_key is None or method is None:
            continue
        with np.load(maps_dir / f"{model_key}_{method}.npz", allow_pickle=False) as arc:
            loaded_maps[(model_key, method)] = arc["maps"]

    plt.rcParams.update(
        {
            "font.size": 8.6,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.0,
        }
    )

    # Figura 1: Acertos vs. Erros (4 casos de discordancia/erro)
    fig1, axes1 = plt.subplots(
        len(ERROR_CASE_INDICES),
        len(QUAL_PANEL_SPECS),
        figsize=(16.5, 10.4),
        constrained_layout=True,
    )
    for row_i, img_idx in enumerate(ERROR_CASE_INDICES):
        row_meta = subset_df.iloc[img_idx]
        img_path = str(row_meta["image_path"])
        true_name = str(row_meta["class_name"])
        rgb, _ = load_image_pair(img_path, config)
        pred_row = e1_preds_df.loc[img_path]

        vig_p = label_to_name[int(pred_row["vig_ti_pred"])]
        vig_c = float(pred_row["vig_ti_conf"])
        vig_tag = "OK" if int(pred_row["vig_ti_correct"]) == 1 else "ERR"

        deit_p = label_to_name[int(pred_row["deit_tiny_pred"])]
        deit_c = float(pred_row["deit_tiny_conf"])
        deit_tag = "OK" if int(pred_row["deit_tiny_correct"]) == 1 else "ERR"

        for col_i, (col_title, model_key, method) in enumerate(QUAL_PANEL_SPECS):
            ax = axes1[row_i, col_i]
            if model_key is None or method is None:
                ax.imshow(rgb)
                ax.set_ylabel(f"idx={img_idx}\n{true_name}", fontsize=8.8, fontweight="bold")
                ax.set_xlabel(
                    f"ViG: {vig_p} ({vig_c:.2f}, {vig_tag})\nDeiT: {deit_p} ({deit_c:.2f}, {deit_tag})",
                    fontsize=7.6,
                )
            else:
                sal = loaded_maps[(model_key, method)][img_idx]
                ax.imshow(_overlay_heatmap(rgb, sal))
                m_row = e3_blur.loc[(img_idx, model_key, method)]
                del_v = float(m_row["deletion_auc"])
                ins_v = float(m_row["insertion_auc"])
                ax.set_xlabel(f"Del={del_v:.3f} | Ins={ins_v:.3f}", fontsize=7.8)

            if row_i == 0:
                ax.set_title(col_title, fontsize=9.5, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig1.suptitle(
        "Inspecao Qualitativa: Acertos vs. Erros e Discordancias entre ViG-Ti e DeiT-Tiny (blur AUC)",
        fontsize=12.2,
    )
    fig1.savefig(figures_dir / "analysis_errors_vs_correct.pdf", dpi=300)
    fig1.savefig(figures_dir / "analysis_errors_vs_correct.png", dpi=300)
    plt.close(fig1)

    # Figura 2: Plausibilidade Visual vs. Fidelidade Quantitativa / Sanity Checks
    fig2, axes2 = plt.subplots(
        len(DIVERGENCE_CASE_INDICES),
        len(QUAL_PANEL_SPECS),
        figsize=(16.5, 8.4),
        constrained_layout=True,
    )
    for row_i, img_idx in enumerate(DIVERGENCE_CASE_INDICES):
        row_meta = subset_df.iloc[img_idx]
        img_path = str(row_meta["image_path"])
        true_name = str(row_meta["class_name"])
        rgb, _ = load_image_pair(img_path, config)
        pred_row = e1_preds_df.loc[img_path]
        vig_c = float(pred_row["vig_ti_conf"])
        deit_c = float(pred_row["deit_tiny_conf"])

        for col_i, (col_title, model_key, method) in enumerate(QUAL_PANEL_SPECS):
            ax = axes2[row_i, col_i]
            if model_key is None or method is None:
                ax.imshow(rgb)
                ax.set_ylabel(f"idx={img_idx}\n{true_name}", fontsize=8.8, fontweight="bold")
                ax.set_xlabel(
                    f"Ambos OK (ViG={vig_c:.2f}, DeiT={deit_c:.2f})",
                    fontsize=7.8,
                )
            else:
                sal = loaded_maps[(model_key, method)][img_idx]
                ax.imshow(_overlay_heatmap(rgb, sal))
                m_row = e3_blur.loc[(img_idx, model_key, method)]
                del_v = float(m_row["deletion_auc"])
                ins_v = float(m_row["insertion_auc"])
                label_txt = f"Del={del_v:.3f} | Ins={ins_v:.3f}"

                if method in {"gradcam", "rollout"}:
                    rho_head = float(e4_idx.loc[(img_idx, model_key, method, "head"), "spearman"])
                    label_txt += f"\nE4 head rho={rho_head:.2f}"
                elif method == "integrated_gradients":
                    rho_all = float(e4_idx.loc[(img_idx, model_key, method, "all_layers"), "spearman"])
                    label_txt += f"\nE4 all rho={rho_all:.2f}"

                ax.set_xlabel(label_txt, fontsize=7.6)

            if row_i == 0:
                ax.set_title(col_title, fontsize=9.5, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig2.suptitle(
        "Plausibilidade Visual vs. Fidelidade Quantitativa (E3 blur AUC) e Sanity Checks (E4)",
        fontsize=12.2,
    )
    fig2.savefig(figures_dir / "analysis_plausibility_vs_faithfulness.pdf", dpi=300)
    fig2.savefig(figures_dir / "analysis_plausibility_vs_faithfulness.png", dpi=300)
    plt.close(fig2)

    print("[Figures] Figuras qualitativas geradas em results/figures/.")


def main() -> None:
    """Ponto de entrada para ablacoes, consolidacao e figuras qualitativas."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in {"ablations", "all"}:
        run_ablations(config, device, limit=args.limit)
    if args.limit is not None:
        if args.mode in {"summary", "figures"}:
            print("[Analyze] Modo --limit se aplica apenas a ablacoes.")
        return
    if args.mode in {"summary", "all"}:
        build_master_summary(config)
    if args.mode in {"figures", "all"}:
        ablation_csv = resolve_path(config["evaluation"]["metrics_dir"]) / "ablation_summary.csv"
        if args.mode == "figures" and ablation_csv.exists():
            _plot_ablation_sensitivity(
                pd.read_csv(ablation_csv),
                resolve_path(config["evaluation"]["figures_dir"]),
            )
        generate_qualitative_figures(config)


if __name__ == "__main__":
    main()
