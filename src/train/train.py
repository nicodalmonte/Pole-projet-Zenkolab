"""Training script for DinoV3_1 glaucoma classifier.

Dataset split strategy
----------------------
A single dataset is loaded (selectable via --dataset) and split into
train / val / test using stratified sampling to preserve class balance.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import numpy as np
from PIL import Image as PILImage

import timm
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split

from src.datasets import (
    ACRIMADataset,
    FundusTrainValDataset,
    LAGDataset,
    ORIGADataset,
    REFUGE2Dataset,
    JRAIGSDataset,
    G1020Dataset,
)
from src.models.dino_v3_1 import DinoV3_1
from src.datasets.augmentations import AUGMENTATION_TRANSFORMS

SEED = 42

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_MAP: dict[str, type] = {
    "JRAIGS": JRAIGSDataset,
    "ACRIMA": ACRIMADataset,
    "LAG": LAGDataset,
    "ORIGA": ORIGADataset,
    "REFUGE2": REFUGE2Dataset,
    "Fundus": FundusTrainValDataset,
    "G1020": G1020Dataset,
}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(backbone_name: str, image_size: int = 896):
    """Return (train_transform, eval_transform) derived from the timm model config.

    Training transforms include data augmentations (rotations, affine transforms, etc.)
    defined in src.augmentations.AUGMENTATION_TRANSFORMS.

    Evaluation transforms do not include augmentations.
    """
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    # Override the crop size so the full image_size is passed to the backbone.
    # DINOv3 uses patch_size=16, so any multiple of 16 is valid.
    data_cfg["input_size"] = (3, image_size, image_size)

    # Build the timm transforms
    timm_train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    timm_eval_tf = timm.data.create_transform(**data_cfg, is_training=False)

    # Compose timm transforms with augmentations for training
    train_tf = transforms.Compose([
        AUGMENTATION_TRANSFORMS,
        timm_train_tf,
    ])

    return train_tf, timm_eval_tf


# ---------------------------------------------------------------------------
# TransformedSubset
# ---------------------------------------------------------------------------

class TransformedSubset(Dataset):
    """Wraps a Subset and applies a transform, allowing train/val/test to have
    different transforms even when split from the same Dataset instance."""

    def __init__(self, subset: Subset, transform) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.subset[idx]
        if self.transform is not None:
            img = sample["image"]
            if isinstance(img, np.ndarray):
                img = PILImage.fromarray(img)
            sample["image"] = self.transform(img)
        return sample


# ---------------------------------------------------------------------------
# Label extraction (generic, works with all dataset classes)
# ---------------------------------------------------------------------------

def extract_labels(ds: Dataset) -> list[int]:
    """Return a flat list of integer labels for every sample in *ds*.

    Supports Subset, all project Dataset classes, and falls back to iterating
    __getitem__ for unknown types (slow).
    """
    if isinstance(ds, Subset):
        base = ds.dataset
        if hasattr(base, "samples"):
            return [base.samples[i][1] for i in ds.indices]
        # Fall through to base-dataset logic on the Subset's underlying dataset
        base_labels = extract_labels(base)
        return [base_labels[i] for i in ds.indices]

    if isinstance(ds, (JRAIGSDataset, ORIGADataset, G1020Dataset)):
        return [lbl for _, lbl in ds.samples]
    if isinstance(ds, ACRIMADataset):
        return [1 if "_g_" in p.name else 0 for p in ds.image_paths]
    if isinstance(ds, LAGDataset):
        return [1 if p.name.startswith("g.") else 0 for p in ds.image_paths]
    if isinstance(ds, REFUGE2Dataset):
        return [1 if p.name.startswith("g") else 0 for p in ds.image_paths]
    if isinstance(ds, FundusTrainValDataset):
        return list(ds.labels)

    # Generic fallback — loads every image (slow)
    return [int(ds[i]["label"].item()) for i in range(len(ds))]


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    dataset: Dataset,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = SEED,
) -> tuple[Subset, Subset, Subset]:
    """Split *dataset* into (train, val, test) subsets using stratified sampling.

    Args:
        dataset: Any Dataset with consistent labels (use extract_labels internally).
        ratios: (train, val, test) proportions, must sum to 1.
        seed: Random seed for reproducibility.

    Returns:
        Three Subset objects: train_subset, val_subset, test_subset.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    train_ratio, val_ratio, test_ratio = ratios

    labels = extract_labels(dataset)
    indices = list(range(len(dataset)))

    # First split: train vs (val + test)
    train_idx, rest_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test (relative proportion within rest)
    rest_labels = [labels[i] for i in rest_idx]
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=relative_test,
        stratify=rest_labels,
        random_state=seed,
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ---------------------------------------------------------------------------
# WeightedRandomSampler
# ---------------------------------------------------------------------------

def build_weighted_sampler(subset: Subset, seed: int = SEED) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that upsamples the minority class.

    Each sample is assigned a weight equal to 1 / class_count so that every
    class is drawn with equal expected frequency per epoch.
    """
    labels = extract_labels(subset)
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[lbl] for lbl in labels]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=g,
    )


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _count_glaucoma(ds: Dataset) -> tuple[int, int]:
    """Return (n_glaucoma, total). No images opened."""
    labels = extract_labels(ds)
    return sum(labels), len(labels)


def print_split_info(split_name: str, ds: Dataset) -> None:
    """Pretty-print size and class balance for a split."""
    sub_datasets = ds.datasets if hasattr(ds, "datasets") else [ds]
    rows: list[tuple[str, int, int, int]] = []
    total_g = total_ng = 0
    for sub in sub_datasets:
        g, tot = _count_glaucoma(sub)
        ng = tot - g
        total_g += g
        total_ng += ng
        rows.append((type(sub).__name__, tot, g, ng))
    total = total_g + total_ng
    glaucoma_pct = 100.0 * total_g / total if total else 0.0

    W = 64
    print(f"\n{'─' * W}")
    print(f"  {split_name}  —  {total} samples total")
    print(f"{'─' * W}")
    print(f"  {'Dataset':<32}  {'Total':>6}  {'Glaucoma':>9}  {'Non-Glauc.':>10}")
    print(f"  {'─' * 60}")
    for ds_name, tot, g, ng in rows:
        print(f"  {ds_name:<32}  {tot:>6}  {g:>9}  {ng:>10}")
    if len(rows) > 1:
        print(f"  {'─' * 60}")
        print(f"  {'TOTAL':<32}  {total:>6}  {total_g:>9}  {total_ng:>10}")
    print(
        f"  Class balance: {glaucoma_pct:.1f}% glaucoma / "
        f"{100.0 - glaucoma_pct:.1f}% non-glaucoma"
    )
    print(f"{'─' * W}")


def compute_class_weights(train_ds: Dataset) -> list[float]:
    """Return inverse-frequency class weights [w_neg, w_pos]."""
    sub_datasets = train_ds.datasets if hasattr(train_ds, "datasets") else [train_ds]
    total_g = total_ng = 0
    for sub in sub_datasets:
        g, tot = _count_glaucoma(sub)
        total_g += g
        total_ng += tot - g
    total = total_g + total_ng
    if total_ng == 0 or total_g == 0:
        return [1.0, 1.0]
    w_neg = total / (2.0 * total_ng)
    w_pos = total / (2.0 * total_g)
    return [w_neg, w_pos]


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str,
    dataset_name: str,
    backbone_name: str,
    batch_size: int,
    num_workers: int,
    image_size: int = 896,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms(backbone_name, image_size=image_size)

    dataset_cls = DATASET_MAP[dataset_name]

    # Load full dataset without transforms (applied per-split below)
    full_ds = dataset_cls(data_dir=data_dir, transforms=None)

    train_subset, val_subset, test_subset = stratified_split(
        full_ds, ratios=split_ratios, seed=SEED
    )

    # Apply transforms per split
    train_ds = TransformedSubset(train_subset, train_tf)
    val_ds   = TransformedSubset(val_subset,   eval_tf)
    test_ds  = TransformedSubset(test_subset,  eval_tf)

    print_split_info("TRAIN", train_ds)
    print_split_info("VAL  ", val_ds)
    print_split_info("TEST ", test_ds)

    pin = torch.cuda.is_available()
    sampler = build_weighted_sampler(train_subset, seed=SEED)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )

    print(f"\nTrain samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    print(f"Test  samples : {len(test_ds)}")

    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DinoV3_1 glaucoma classifier")
    p.add_argument("--data_dir", default="data/datasets")
    p.add_argument(
        "--dataset",
        default="JRAIGS",
        choices=list(DATASET_MAP.keys()),
        help="Dataset to use for training. The dataset is split into train/val/test "
             "using stratified sampling.",
    )
    p.add_argument(
        "--split_ratios",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratios (must sum to 1). Default: 0.8 0.1 0.1",
    )
    p.add_argument("--backbone", default="vit_large_patch16_dinov3.lvd1689m")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--unfreeze_backbone_epoch",
        type=int,
        default=3,
        help="If > 0, keep the backbone frozen until this epoch, then unfreeze it.",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--devices", default="auto")
    p.add_argument("--precision", default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument(
        "--image_size",
        type=int,
        default=896,
        help="Input image size fed to the backbone. Must be a multiple of 16 "
             "(patch size, DINOv3). Default 896 = 4×224.",
    )
    p.add_argument(
        "--class_weights",
        type=float,
        nargs=2,
        default=None,
        metavar=("W_NEG", "W_POS"),
        help="Manual class weights for [non-glaucoma, glaucoma]. "
             "If omitted, weights are computed automatically from the training "
             "set using inverse-frequency balancing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    L.seed_everything(SEED, workers=True)

    train_dl, val_dl, test_dl = build_dataloaders(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        split_ratios=tuple(args.split_ratios),
    )

    class_weights = (
        args.class_weights
        if args.class_weights is not None
        else compute_class_weights(train_dl.dataset)
    )
    print(f"\nClass weights — non-glaucoma: {class_weights[0]:.4f}  glaucoma: {class_weights[1]:.4f}")

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
        unfreeze_backbone_epoch=args.unfreeze_backbone_epoch,
    )

    logger = CSVLogger(save_dir="lightning_logs", name="dinov3_1")
    version_number = logger.version
    print(f"Logging to: lightning_logs/dinov3_1/version_{version_number}")

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.checkpoint_dir}/version_{version_number}",
            filename=f"dinov3_1_v{version_number}-" + "{epoch:02d}-{val_auc:.4f}",
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
        RichModelSummary(max_depth=3),
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
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, train_dl, val_dl, ckpt_path=args.resume)

    print("\n" + "=" * 60)
    print(f"Testing on held-out test split ({args.dataset})")
    print("=" * 60)
    trainer.test(model, test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()
