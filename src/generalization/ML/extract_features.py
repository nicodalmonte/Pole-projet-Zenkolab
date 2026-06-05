"""Utilities to extract embedding features into pandas dataframes.

The module provides two layers of functionality:
- reusable helpers that turn one dataset or many datasets into a dataframe;
- a CLI that loads the full dataset set used by dataset clustering and writes
  the combined embeddings to a CSV file.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.datasets import (
    ACRIMADataset,
    AIROGSLightDataset,
    FundusTrainValDataset,
    HarvardGlaucomaDataset,
    JRAIGSDataset,
    LAGDataset,
    ORIGADataset,
)
from src.datasets.RIMONE import RIMONEDataset


def _log(message: str) -> None:
    """Print a timestamped progress line that flushes immediately."""

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def build_eval_transform(backbone_name: str, img_size: int = 224):
    """Build the default evaluation transform for a timm backbone."""

    import timm

    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    data_cfg["input_size"] = (3, img_size, img_size)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    if hasattr(transform, "transforms") and not any(isinstance(item, T.Normalize) for item in transform.transforms):
        transform = T.Compose([
            *transform.transforms,
            T.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
        ])

    return transform


def load_backbone(backbone_name: str, device: torch.device) -> torch.nn.Module:
    """Load a frozen timm backbone for feature extraction."""

    import timm

    model = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _log(f"Backbone: {backbone_name}  -  feature dim: {model.num_features}")
    return model


def _extract_embedding_output(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return a 2D tensor of embeddings for a batch of images."""

    if hasattr(model, "backbone"):
        output = model.backbone(images)  # type: ignore[attr-defined]
    elif hasattr(model, "features"):
        output = model.features(images)  # type: ignore[attr-defined]
    else:
        output = model(images)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if output.dim() == 4:
        output = output.mean(dim=(2, 3))

    if output.dim() != 2:
        raise ValueError(f"Expected a 2D embedding tensor, got shape {tuple(output.shape)}")

    return output


def _batch_to_device(batch: Any, device: torch.device) -> dict[str, Any]:
    if not isinstance(batch, Mapping):
        raise TypeError("Expected each batch to be a mapping with an 'image' key")

    moved: dict[str, Any] = dict(batch)
    moved["image"] = batch["image"].to(device, non_blocking=True)
    return moved


def _rows_to_dataframe(rows: list[dict[str, Any]], embeddings: np.ndarray) -> pd.DataFrame:
    embedding_columns = [f"embedding_{index}" for index in range(embeddings.shape[1])]
    embedding_frame = pd.DataFrame(embeddings, columns=embedding_columns)
    row_frame = pd.DataFrame(rows).reset_index(drop=True)
    return pd.concat([row_frame, embedding_frame], axis=1)


@torch.inference_mode()
def extract_dataset_embeddings(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    source: str | None = None,
) -> pd.DataFrame:
    """Extract embeddings for one dataset and return them as a dataframe."""

    try:
        first_parameter = next(model.parameters())
        default_device = first_parameter.device
    except StopIteration:
        default_device = torch.device("cpu")

    resolved_device = torch.device(device) if device is not None else default_device
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": resolved_device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(dataset, **loader_kwargs)
    was_training = model.training
    model.eval()

    total_batches = len(loader)
    _log(f"Extracting {source or 'dataset'}: {len(dataset)} samples in {total_batches} batches")

    row_batches: list[dict[str, Any]] = []
    embedding_batches: list[np.ndarray] = []

    try:
        for batch_index, batch in enumerate(loader, start=1):
            moved_batch = _batch_to_device(batch, resolved_device)
            embeddings = _extract_embedding_output(model, moved_batch["image"]).detach().cpu().float().numpy()

            batch_size_current = embeddings.shape[0]
            for row_index in range(batch_size_current):
                row: dict[str, Any] = {}
                for key, value in batch.items():
                    if key == "image":
                        continue

                    if isinstance(value, torch.Tensor):
                        item = value[row_index].detach().cpu()
                        row[key] = item.item() if item.ndim == 0 else item.tolist()
                    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                        row[key] = value[row_index]
                    else:
                        row[key] = value

                if source is not None:
                    row["source"] = source

                row_batches.append(row)

            embedding_batches.append(embeddings)

            if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
                _log(f"  {source or 'dataset'}: processed batch {batch_index}/{total_batches}")
    finally:
        if was_training:
            model.train()

    if not embedding_batches:
        empty_columns = ["source"] if source is not None else []
        return pd.DataFrame(columns=empty_columns)

    embeddings = np.concatenate(embedding_batches, axis=0)
    return _rows_to_dataframe(row_batches, embeddings)


def extract_multiple_datasets_embeddings(
    model: torch.nn.Module,
    datasets: Mapping[str, Dataset] | Sequence[tuple[str, Dataset]],
    *,
    device: torch.device | str | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Extract embeddings for multiple datasets and merge them into one dataframe."""

    if isinstance(datasets, Mapping):
        dataset_items = list(datasets.items())
    else:
        dataset_items = list(datasets)

    frames: list[pd.DataFrame] = []
    for source_name, dataset in dataset_items:
        frame = extract_dataset_embeddings(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            source=source_name,
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["source"])

    return pd.concat(frames, ignore_index=True)


def load_feature_dataset_factories(data_dir: str | Path, transforms) -> dict[str, Callable[[], Dataset]]:
    """Return the dataset factory list used for generalization feature export."""

    return {
        "JRAIGS": lambda: JRAIGSDataset(data_dir=data_dir, transforms=transforms),
        "ACRIMA": lambda: ACRIMADataset(data_dir=data_dir, transforms=transforms),
        "ORIGA": lambda: ORIGADataset(data_dir=data_dir, transforms=transforms),
        "LAG": lambda: LAGDataset(data_dir=data_dir, split="train", transforms=transforms),
        "Harvard": lambda: HarvardGlaucomaDataset(data_dir=data_dir, transforms=transforms),
        "RIMONE(train)": lambda: RIMONEDataset(data_dir=data_dir, split="train", transforms=transforms),
        "RIMONE(test)": lambda: RIMONEDataset(data_dir=data_dir, split="test", transforms=transforms),
        "AIRROGS": lambda: AIROGSLightDataset(data_dir=data_dir, transforms=transforms),
        "Fundus(train)": lambda: FundusTrainValDataset(data_dir=data_dir, split="train", transforms=transforms),
        "Fundus(val)": lambda: FundusTrainValDataset(data_dir=data_dir, split="validation", transforms=transforms),
    }


def extract_all_datasets_to_csv(
    *,
    data_dir: str | Path,
    out_csv: str | Path,
    backbone: str,
    device: torch.device | str | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Load all datasets, extract embeddings, and write the combined dataframe."""

    resolved_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    transforms = build_eval_transform(backbone)
    model = load_backbone(backbone, resolved_device)

    factories = load_feature_dataset_factories(data_dir, transforms)
    loaded_datasets: dict[str, Dataset] = {}
    _log(f"Scanning datasets under {Path(data_dir)}")
    for name, factory in factories.items():
        try:
            dataset = factory()
            loaded_datasets[name] = dataset
            _log(f"  {name:<20} {len(dataset):>6} samples")
        except Exception as exc:
            _log(f"  {name:<20} SKIPPED  ({exc})")

    _log(f"Starting extraction for {len(loaded_datasets)} datasets")

    frame = extract_multiple_datasets_embeddings(
        model,
        loaded_datasets,
        device=resolved_device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Writing CSV to {out_path}")
    frame.to_csv(out_path, index=False)
    _log(f"Saved embeddings CSV: {out_path}")
    _log(f"Rows: {len(frame)}")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract dataset embeddings to a CSV file.")
    parser.add_argument("--data_dir", default="data/datasets")
    parser.add_argument("--out_csv", default="data/ML/all_embeddings.csv")
    parser.add_argument("--backbone", default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_all_datasets_to_csv(
        data_dir=args.data_dir,
        out_csv=args.out_csv,
        backbone=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()