"""Executa o fine-tuning controlado de ViG-Ti e DeiT-Tiny no Imagenette (E1)."""

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
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.config import load_config, resolve_path, set_seed
from src.data import build_dataloaders
from src.models.deit_tiny import load_deit_tiny
from src.models.vig_ti import load_vig_ti


MODEL_DISPLAY_NAMES = {
    "vig_ti": "ViG-Ti",
    "deit_tiny": "DeiT-Tiny",
}


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


def _load_model_for_training(model_key: str, config: dict[str, Any]) -> nn.Module:
    if model_key == "vig_ti":
        return load_vig_ti(config, checkpoint_path=None)
    if model_key == "deit_tiny":
        return load_deit_tiny(config, checkpoint_path=None)
    raise ValueError(f"Modelo desconhecido: {model_key}")


def _load_model_from_checkpoint(
    model_key: str,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> nn.Module:
    if model_key == "vig_ti":
        return load_vig_ti(config, checkpoint_path=checkpoint_path)
    if model_key == "deit_tiny":
        return load_deit_tiny(config, checkpoint_path=checkpoint_path)
    raise ValueError(f"Modelo desconhecido: {model_key}")


def _build_scheduler(
    optimizer: AdamW,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=1.0e-6,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1.0e-6)


def _run_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    desc: str = "",
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss nao-finito detectado ({loss.item()}) durante {desc}.")

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_single_model(
    model_key: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Executa o fine-tuning controlado de um modelo e retorna metadados de treino."""
    seed = int(config["seed"])
    set_seed(seed)
    dataloaders = build_dataloaders(config)

    train_cfg = config["training"]
    epochs = int(train_cfg["epochs"])
    warmup_epochs = int(train_cfg["warmup_epochs"])
    patience = int(train_cfg["early_stopping_patience"])
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])

    ckpt_dir = resolve_path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"{model_key}_best.pt"

    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = metrics_dir / f"e1_{model_key}_history.csv"

    wrapper = _load_model_for_training(model_key, config).to(device)
    total_params = sum(p.numel() for p in wrapper.parameters())
    trainable_params = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)

    optimizer = AdamW(wrapper.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _build_scheduler(optimizer, epochs=epochs, warmup_epochs=warmup_epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_es_loss = float("inf")
    no_improve = 0

    history_rows: list[dict[str, Any]] = []
    train_start = time.perf_counter()

    print(
        f"\n[E1] Iniciando fine-tuning de {MODEL_DISPLAY_NAMES[model_key]} "
        f"({total_params:,} params) em {device}"
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_loss, train_acc = _run_epoch(
            wrapper,
            dataloaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            desc=f"{model_key} E{epoch:02d}/{epochs} train",
        )
        val_loss, val_acc = _run_epoch(
            wrapper,
            dataloaders["val"],
            criterion,
            device,
            optimizer=None,
            desc=f"{model_key} E{epoch:02d}/{epochs} val",
        )
        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start

        history_rows.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
            }
        )

        is_better = val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": wrapper.model.state_dict(),
                    "model_name": model_key,
                    "best_epoch": best_epoch,
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                    "seed": seed,
                    "config": config,
                },
                best_ckpt_path,
            )

        if val_loss < best_es_loss - 1.0e-4:
            best_es_loss = val_loss
            no_improve = 0
        elif epoch > warmup_epochs:
            no_improve += 1

        print(
            f"Epoch {epoch:02d}/{epochs} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        if no_improve >= patience:
            print(f"[E1] Early stopping acionado na epoca {epoch} para {model_key}.")
            break

    total_train_time = time.perf_counter() - train_start
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    return {
        "model": model_key,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "best_epoch": int(best_epoch),
        "epochs_trained": len(history_rows),
        "total_train_time_sec": float(total_train_time),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "checkpoint_path": str(best_ckpt_path.relative_to(resolve_path("."))),
    }


def _predict_split(
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[np.ndarray] = []
    all_confs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels_dev)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            all_preds.append(preds.cpu().numpy())
            all_confs.append(confs.cpu().numpy())
            all_labels.append(labels.numpy())

    return (
        total_loss / total_samples,
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_confs),
    )


def evaluate_models(
    model_keys: list[str],
    train_infos: dict[str, dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> None:
    """Avalia os melhores checkpoints salvos no conjunto de teste e no subconjunto XAI."""
    dataloaders = build_dataloaders(config)
    splits_dir = resolve_path(config["data"]["splits_dir"])
    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])
    figures_dir = resolve_path(config["evaluation"]["figures_dir"])
    ckpt_dir = resolve_path(config["training"]["checkpoint_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(splits_dir / "test.csv")
    xai_df = pd.read_csv(splits_dir / "xai_subset.csv")
    xai_paths = set(xai_df["image_path"])

    models_summary: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    pred_store: dict[str, dict[str, np.ndarray]] = {}

    for model_key in model_keys:
        ckpt_path = ckpt_dir / f"{model_key}_best.pt"
        wrapper = _load_model_from_checkpoint(model_key, config, ckpt_path).to(device)
        test_loss, y_true, y_pred, y_conf = _predict_split(wrapper, dataloaders["test"], device)

        test_acc = float((y_pred == y_true).mean())
        test_macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

        in_xai = test_df["image_path"].isin(xai_paths).to_numpy()
        xai_correct = int((y_pred[in_xai] == y_true[in_xai]).sum())
        xai_total = int(in_xai.sum())
        xai_errors = int(xai_total - xai_correct)
        xai_acc = float(xai_correct / xai_total)

        info = dict(train_infos[model_key])
        info.update(
            {
                "test_loss": float(test_loss),
                "test_acc": test_acc,
                "test_macro_f1": test_macro_f1,
                "xai_subset_acc": xai_acc,
                "xai_subset_correct": xai_correct,
                "xai_subset_errors": xai_errors,
            }
        )
        models_summary[model_key] = info
        summary_rows.append(info)
        pred_store[model_key] = {
            "true": y_true,
            "pred": y_pred,
            "conf": y_conf,
            "correct": (y_pred == y_true).astype(int),
        }

    pd.DataFrame(summary_rows).to_csv(metrics_dir / "e1_test_summary.csv", index=False)

    summary_payload: dict[str, Any] = {
        "seed": int(config["seed"]),
        "device": str(device),
        "models": models_summary,
    }

    if "vig_ti" in pred_store and "deit_tiny" in pred_store:
        pred_df = pd.DataFrame(
            {
                "image_path": test_df["image_path"],
                "synset": test_df["synset"],
                "class_name": test_df["class_name"],
                "true_label": test_df["label"],
                "vig_ti_pred": pred_store["vig_ti"]["pred"],
                "vig_ti_conf": pred_store["vig_ti"]["conf"],
                "vig_ti_correct": pred_store["vig_ti"]["correct"],
                "deit_tiny_pred": pred_store["deit_tiny"]["pred"],
                "deit_tiny_conf": pred_store["deit_tiny"]["conf"],
                "deit_tiny_correct": pred_store["deit_tiny"]["correct"],
                "in_xai_subset": test_df["image_path"].isin(xai_paths).astype(int),
            }
        )
        pred_df.to_csv(metrics_dir / "e1_test_predictions.csv", index=False)

        def _agreement_stats(df_sub: pd.DataFrame) -> dict[str, int]:
            v = df_sub["vig_ti_correct"] == 1
            d = df_sub["deit_tiny_correct"] == 1
            return {
                "total": int(len(df_sub)),
                "both_correct": int((v & d).sum()),
                "only_vig_ti_correct": int((v & ~d).sum()),
                "only_deit_tiny_correct": int((~v & d).sum()),
                "both_wrong": int((~v & ~d).sum()),
            }

        summary_payload["agreement"] = {
            "test": _agreement_stats(pred_df),
            "xai_subset": _agreement_stats(pred_df[pred_df["in_xai_subset"] == 1]),
        }

        per_class_rows: list[dict[str, Any]] = []
        for label, group in pred_df.groupby("true_label", sort=True):
            per_class_rows.append(
                {
                    "label": int(label),
                    "synset": str(group["synset"].iloc[0]),
                    "class_name": str(group["class_name"].iloc[0]),
                    "test_samples": int(len(group)),
                    "vig_ti_acc": float(group["vig_ti_correct"].mean()),
                    "deit_tiny_acc": float(group["deit_tiny_correct"].mean()),
                    "vig_ti_correct": int(group["vig_ti_correct"].sum()),
                    "deit_tiny_correct": int(group["deit_tiny_correct"].sum()),
                }
            )
        pd.DataFrame(per_class_rows).to_csv(
            metrics_dir / "e1_per_class_accuracy.csv",
            index=False,
        )

    with (metrics_dir / "e1_test_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    _plot_training_curves(model_keys, metrics_dir, figures_dir)


def _plot_training_curves(
    model_keys: list[str],
    metrics_dir: Path,
    figures_dir: Path,
) -> None:
    colors = {
        "vig_ti": "#1f77b4",
        "deit_tiny": "#d62728",
    }

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    for model_key in model_keys:
        hist_path = metrics_dir / f"e1_{model_key}_history.csv"
        if not hist_path.exists():
            continue
        df = pd.read_csv(hist_path)
        label = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        color = colors.get(model_key, "#333333")

        axes[0].plot(
            df["epoch"],
            df["train_loss"],
            linestyle="--",
            color=color,
            alpha=0.75,
            label=f"{label} (Train)",
        )
        axes[0].plot(
            df["epoch"],
            df["val_loss"],
            linestyle="-",
            marker="o",
            markersize=3.5,
            color=color,
            label=f"{label} (Val)",
        )

        axes[1].plot(
            df["epoch"],
            df["train_acc"] * 100.0,
            linestyle="--",
            color=color,
            alpha=0.75,
            label=f"{label} (Train)",
        )
        axes[1].plot(
            df["epoch"],
            df["val_acc"] * 100.0,
            linestyle="-",
            marker="o",
            markersize=3.5,
            color=color,
            label=f"{label} (Val)",
        )

    axes[0].set_title("Loss por epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(frameon=True)

    axes[1].set_title("Acuracia por epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Acuracia (%)")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(frameon=True)

    fig.suptitle("Curvas de convergencia no Imagenette (E1: ViG-Ti vs. DeiT-Tiny)", fontsize=12)
    fig.savefig(figures_dir / "e1_training_curves.pdf", dpi=300)
    fig.savefig(figures_dir / "e1_training_curves.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Ponto de entrada para treinamento dos modelos."""
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_keys = ["vig_ti", "deit_tiny"] if args.model == "all" else [args.model]
    train_infos: dict[str, dict[str, Any]] = {}

    for model_key in model_keys:
        train_infos[model_key] = train_single_model(model_key, config, device)

    evaluate_models(model_keys, train_infos, config, device)

    metrics_dir = resolve_path(config["evaluation"]["metrics_dir"])
    with (metrics_dir / "e1_test_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    print("\n[E1] Resumo final no conjunto de teste:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
