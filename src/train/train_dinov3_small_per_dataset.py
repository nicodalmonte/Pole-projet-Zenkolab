"""Train DINOv3-small on ACRIMA, ORIGA, and RIM-ONE independently.

Each dataset is split independently into train/val/test (70/15/15 by default).
Single training phase with the backbone fully unfrozen from epoch 0.
RIM-ONE: training_set + test_set are merged then re-split randomly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import lightning as L
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

from src.datasets import ACRIMADataset, ORIGADataset
from src.datasets.RIMONE import RIMONEDataset
from src.datasets.augmentations import AUGMENTATION_TRANSFORMS
from src.models.dino_v3_1 import DinoV3_1
from src.train.paper.common_data import split_train_val_test_indices


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(backbone_name: str, image_size: int):
    """Return (train_tf, eval_tf) derived from the timm model config."""
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    data_cfg["input_size"] = (3, image_size, image_size)
    timm_train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    timm_eval_tf = timm.data.create_transform(**data_cfg, is_training=False)
    train_tf = transforms.Compose([AUGMENTATION_TRANSFORMS, timm_train_tf])
    return train_tf, timm_eval_tf


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class SplitTransformDataset(Dataset)
    """Apply a transform to the 'image' field of each sample."""

    def __init__(self, dataset: Dataset, image_transform) -> None:
        self.dataset = dataset
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = dict(self.dataset[index])
        image = sample["image"]
        if not isinstance(image, torch.Tensor):
            if hasattr(image, "shape"):
                image = Image.fromarray(image)
            sample["image"] = self.image_transform(image)
        return sample


class RIMONEAllDataset(Dataset):
    """RIM-ONE with training_set + test_set merged for independent re-splitting."""

    def __init__(
        self,
        data_dir: str | Path = "data/datasets",
        transforms=None,
    ) -> None:
        self.transforms = transforms
        train_ds = RIMONEDataset(data_dir=data_dir, split="train", transforms=None)
        test_ds = RIMONEDataset(data_dir=data_dir, split="test", transforms=None)
        self.image_paths = train_ds.image_paths + test_ds.image_paths
        self.labels = train_ds.labels + test_ds.labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            import numpy as np
            image = np.array(image)
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(image_path),
        }


DATASET_MAP: dict[str, type] = {
    "acrima": ACRIMADataset,
    "origa": ORIGADataset,
    "rimone": RIMONEAllDataset,
}


# ---------------------------------------------------------------------------
# Label extraction and class weights
# ---------------------------------------------------------------------------

def _get_all_labels(ds: Dataset) -> list[int]:
    if isinstance(ds, ACRIMADataset):
        return [1 if "_g_" in p.name else 0 for p in ds.image_paths]
    if isinstance(ds, ORIGADataset):
        return [label for _, label in ds.samples]
    if isinstance(ds, RIMONEAllDataset):
        return list(ds.labels)
    raise TypeError(f"Unsupported dataset type: {type(ds).__name__}")


def _compute_class_weights(labels: list[int], indices: list[int]) -> list[float]:
    split_labels = [labels[i] for i in indices]
    total = len(split_labels)
    pos = sum(split_labels)
    neg = total - pos
    if pos == 0 or neg == 0:
        return [1.0, 1.0]
    return [total / (2.0 * neg), total / (2.0 * pos)]


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def _make_dataloader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_dataloaders(
    dataset_name: str,
    data_dir: str,
    backbone_name: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list[float]]:
    factory = DATASET_MAP[dataset_name]
    train_tf, eval_tf = build_transforms(backbone_name, image_size)

    ds_raw = factory(data_dir=data_dir, transforms=None)
    all_labels = _get_all_labels(ds_raw)
    n = len(ds_raw)

    train_idx, val_idx, test_idx = split_train_val_test_indices(
        n, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    class_weights = _compute_class_weights(all_labels, train_idx)

    ds_for_train = factory(data_dir=data_dir, transforms=None)
    ds_for_eval = factory(data_dir=data_dir, transforms=None)

    train_dl = _make_dataloader(
        SplitTransformDataset(Subset(ds_for_train, train_idx), train_tf),
        batch_size, num_workers, shuffle=True,
    )
    val_dl = _make_dataloader(
        SplitTransformDataset(Subset(ds_for_eval, val_idx), eval_tf),
        batch_size, num_workers, shuffle=False,
    )
    test_dl = _make_dataloader(
        SplitTransformDataset(Subset(ds_for_eval, test_idx), eval_tf),
        batch_size, num_workers, shuffle=False,
    )

    _print_split_info(dataset_name, all_labels, train_idx, val_idx, test_idx)
    return train_dl, val_dl, test_dl, class_weights


def _print_split_info(
    name: str,
    labels: list[int],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {name.upper()}  —  {len(labels)} samples total")
    print(f"{'─' * width}")
    for split_name, idx in [("TRAIN", train_idx), ("VAL", val_idx), ("TEST", test_idx)]:
        sl = [labels[i] for i in idx]
        pos = sum(sl)
        neg = len(sl) - pos
        print(f"  {split_name:<6}  {len(sl):>4} samples  glaucoma={pos}  normal={neg}")
    print(f"{'─' * width}")


# ---------------------------------------------------------------------------
# Train one dataset
# ---------------------------------------------------------------------------

def train_on_dataset(dataset_name: str, args: argparse.Namespace) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Training on: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    train_dl, val_dl, test_dl, class_weights = build_dataloaders(
        dataset_name=dataset_name,
        data_dir=args.data_dir,
        backbone_name=args.backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"\n  Class weights — neg: {class_weights[0]:.4f}  pos: {class_weights[1]:.4f}")

    model = DinoV3_1(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.image_size,
        class_weights=class_weights,
        unfreeze_backbone_epoch=0,  # single phase: fully unfrozen
    )

    log_name = f"dinov3_small_{dataset_name}"
    logger = CSVLogger(save_dir="lightning_logs", name=log_name)
    version_number = logger.version

    ckpt_dir = f"{args.checkpoint_dir}/{log_name}/version_{version_number}"
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{log_name}_v{version_number}" + "-{epoch:02d}-{val_auc:.4f}",
            monitor="val_auc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            min_delta=1e-3,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(leave=True),
        RichModelSummary(max_depth=2),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        precision=args.precision,
        devices=args.devices,
        log_every_n_steps=10,
        deterministic=False,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    trainer.fit(model, train_dl, val_dl)

    print(f"\n  Testing [{dataset_name.upper()}] on held-out set…")
    trainer.test(model, test_dl, ckpt_path="best")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DINOv3-small independently on ACRIMA, ORIGA, and RIM-ONE"
    )
    parser.add_argument("--data_dir", default="data/datasets")
    parser.add_argument(
        "--backbone",
        default="vit_small_patch16_dinov3.lvd1689m",
        help="timm model ID for the backbone (default: DINOv3-small patch16)",
    )
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument(
        "--devices",
        default="auto",
        help="Lightning devices (e.g. 'auto', '1', '[0,1]')",
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Input resolution; must be a multiple of the patch size (16 for dinov3 patch16)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Fraction of each dataset assigned to training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of each dataset assigned to validation (remainder = test)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["acrima", "origa", "rimone"],
        choices=["acrima", "origa", "rimone"],
        help="Datasets to train on (default: all three sequentially)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    for dataset_name in args.datasets:
        train_on_dataset(dataset_name, args)


if __name__ == "__main__":
    main()
