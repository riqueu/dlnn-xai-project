"""Download, particionamento estratificado e dataloaders do Imagenette."""

from __future__ import annotations

import json
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.config import PROJECT_ROOT, resolve_path

IMAGENETTE_CLASSES: dict[str, str] = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def _resolve_dataset_dir(raw_dir: str | Path) -> Path:
    resolved = resolve_path(raw_dir)
    if (resolved / "train").is_dir() and (resolved / "val").is_dir():
        return resolved
    return resolved / "imagenette2-160"


def download_imagenette(url: str, raw_dir: str | Path) -> Path:
    """Baixa e extrai o conjunto Imagenette para o diretorio especificado."""
    resolved_raw = resolve_path(raw_dir)
    resolved_raw.mkdir(parents=True, exist_ok=True)
    dataset_dir = resolved_raw / "imagenette2-160"

    if (dataset_dir / "train").is_dir() and (dataset_dir / "val").is_dir():
        return dataset_dir

    archive_path = resolved_raw / "imagenette2-160.tgz"
    if not archive_path.exists():
        tmp_path = resolved_raw / "imagenette2-160.tgz.part"
        if tmp_path.exists():
            tmp_path.unlink()
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=resolved_raw, filter="data")

    if not ((dataset_dir / "train").is_dir() and (dataset_dir / "val").is_dir()):
        raise RuntimeError(f"Estrutura inesperada apos extracao em {dataset_dir}")
    return dataset_dir


def _collect_split_rows(split_dir: Path, synset_to_label: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for synset in sorted(synset_to_label.keys()):
        class_dir = split_dir / synset
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Diretorio de classe ausente: {class_dir}")
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for img_path in files:
            rel_path = img_path.resolve().relative_to(PROJECT_ROOT).as_posix()
            rows.append(
                {
                    "image_path": rel_path,
                    "synset": synset,
                    "class_name": IMAGENETTE_CLASSES[synset],
                    "label": synset_to_label[synset],
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["label", "image_path"]).reset_index(drop=True)


def create_splits(
    raw_dir: str | Path,
    splits_dir: str | Path,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Gera os arquivos de split estratificado treino, validacao e teste."""
    dataset_dir = _resolve_dataset_dir(raw_dir)
    out_dir = resolve_path(splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_synsets = sorted(IMAGENETTE_CLASSES.keys())
    synset_to_label = {synset: idx for idx, synset in enumerate(sorted_synsets)}

    train_df = _collect_split_rows(dataset_dir / "train", synset_to_label)
    official_val_df = _collect_split_rows(dataset_dir / "val", synset_to_label)

    test_size = round(1.0 - float(val_ratio), 6)
    val_df, test_df = train_test_split(
        official_val_df,
        test_size=test_size,
        stratify=official_val_df["label"],
        random_state=seed,
    )
    val_df = val_df.sort_values(["label", "image_path"]).reset_index(drop=True)
    test_df = test_df.sort_values(["label", "image_path"]).reset_index(drop=True)

    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])
    assert train_paths.isdisjoint(val_paths), "Sobreposicao entre train e val"
    assert train_paths.isdisjoint(test_paths), "Sobreposicao entre train e test"
    assert val_paths.isdisjoint(test_paths), "Sobreposicao entre val e test"

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    classes_payload = [
        {
            "label": idx,
            "synset": synset,
            "class_name": IMAGENETTE_CLASSES[synset],
        }
        for idx, synset in enumerate(sorted_synsets)
    ]
    with (out_dir / "classes.json").open("w", encoding="utf-8") as handle:
        json.dump(classes_payload, handle, indent=2, ensure_ascii=False)

    summary = {
        "seed": seed,
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_size),
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "per_class": {
            IMAGENETTE_CLASSES[synset]: {
                "train": int((train_df["synset"] == synset).sum()),
                "val": int((val_df["synset"] == synset).sum()),
                "test": int((test_df["synset"] == synset).sum()),
            }
            for synset in sorted_synsets
        },
    }
    with (out_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def create_xai_subset(
    test_csv: str | Path,
    output_csv: str | Path,
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Seleciona a amostra fixa estratificada para avaliacao de XAI."""
    test_path = resolve_path(test_csv)
    out_path = resolve_path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_path).sort_values("image_path").reset_index(drop=True)
    subset_parts: list[pd.DataFrame] = []
    for _, group in test_df.groupby("label", sort=True):
        if len(group) < per_class:
            raise ValueError(
                f"Classe possui apenas {len(group)} amostras no teste, menor que {per_class}"
            )
        subset_parts.append(group.sample(n=per_class, random_state=seed))

    subset_df = pd.concat(subset_parts, ignore_index=True)
    subset_df = subset_df.sort_values(["label", "image_path"]).reset_index(drop=True)

    assert len(subset_df) == len(IMAGENETTE_CLASSES) * per_class
    assert set(subset_df["image_path"]).issubset(set(test_df["image_path"]))

    subset_df.to_csv(out_path, index=False)

    summary_path = out_path.parent / "split_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary["counts"]["xai_subset"] = int(len(subset_df))
        summary["xai_subset_per_class"] = int(per_class)
        for synset, name in IMAGENETTE_CLASSES.items():
            if name in summary.get("per_class", {}):
                summary["per_class"][name]["xai_subset"] = int(
                    (subset_df["synset"] == synset).sum()
                )
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    return subset_df


class ImagenetteDataset(Dataset):
    """Dataset baseado em manifesto CSV para imagens RGB do Imagenette."""

    def __init__(self, csv_path: str | Path, transform: Any = None) -> None:
        self.csv_path = resolve_path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = resolve_path(row["image_path"])
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
        if self.transform is not None:
            tensor = self.transform(rgb_img)
        else:
            tensor = transforms.ToTensor()(rgb_img)
        return tensor, int(row["label"])


def build_transforms(config: dict[str, Any]) -> tuple[transforms.Compose, transforms.Compose]:
    """Constroi as transformacoes de treino e avaliacao padronizadas pelo ImageNet."""
    data_cfg = config["data"]
    size = int(data_cfg["image_size"])
    mean = list(data_cfg["imagenet_mean"])
    std = list(data_cfg["imagenet_std"])

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Constroi os dataloaders deterministicos de treino, validacao, teste e XAI."""
    splits_dir = resolve_path(config["data"]["splits_dir"])
    train_cfg = config["training"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    seed = int(config["seed"])
    pin_memory = torch.cuda.is_available()

    train_tf, eval_tf = build_transforms(config)

    train_ds = ImagenetteDataset(splits_dir / "train.csv", transform=train_tf)
    val_ds = ImagenetteDataset(splits_dir / "val.csv", transform=eval_tf)
    test_ds = ImagenetteDataset(splits_dir / "test.csv", transform=eval_tf)
    xai_ds = ImagenetteDataset(splits_dir / "xai_subset.csv", transform=eval_tf)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=generator,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "xai": DataLoader(
            xai_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
