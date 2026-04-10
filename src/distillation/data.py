"""Dataloaders for the distillation pipeline.

Split strategy (inspired by train_split.py)
-------------------------------------------
Pool  : JRAIGS + ACRIMA + LAG  → random 80/20 → Train / Val
Test  : ORIGA + G1020 + REFUGE2  ← NEVER used during training
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bisect import bisect_right

import timm
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler, random_split
from torchvision import transforms

from src.datasets import (
    ACRIMADataset,
    G1020Dataset,
    JRAIGSDataset,
    LAGDataset,
    ORIGADataset,
    REFUGE2Dataset,
)
from src.datasets.augmentations import AUGMENTATION_TRANSFORMS


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(backbone_name: str, image_size: int = 448):
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    data_cfg["input_size"] = (3, image_size, image_size)
    timm_train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    timm_eval_tf  = timm.data.create_transform(**data_cfg, is_training=False)
    train_tf = transforms.Compose([AUGMENTATION_TRANSFORMS, timm_train_tf])
    return train_tf, timm_eval_tf


# ---------------------------------------------------------------------------
# Label helpers (for class-weight computation)
# ---------------------------------------------------------------------------

def _resolve(ds: Dataset, idx: int) -> tuple[Dataset, int]:
    if isinstance(ds, Subset):
        return _resolve(ds.dataset, ds.indices[idx])
    if isinstance(ds, ConcatDataset):
        d_idx = bisect_right(ds.cumulative_sizes, idx)
        prev  = 0 if d_idx == 0 else ds.cumulative_sizes[d_idx - 1]
        return _resolve(ds.datasets[d_idx], idx - prev)
    return ds, idx


def _label_at(ds: Dataset, idx: int) -> int:
    base, i = _resolve(ds, idx)
    if isinstance(base, ACRIMADataset):
        return 1 if "_g_" in base.image_paths[i].name else 0
    if isinstance(base, JRAIGSDataset):
        return int(base.samples[i][1])
    if isinstance(base, LAGDataset):
        return 1 if base.image_paths[i].name.startswith("g.") else 0
    raise TypeError(f"Unsupported dataset: {type(base).__name__}")


def compute_class_weights(train_subset: Subset) -> list[float]:
    labels = [_label_at(train_subset.dataset, i) for i in train_subset.indices]
    g  = sum(labels)
    ng = len(labels) - g
    if g == 0 or ng == 0:
        return [1.0, 1.0]
    total = g + ng
    return [total / (2.0 * ng), total / (2.0 * g)]


# ---------------------------------------------------------------------------
# Wrapper to apply per-split transforms on a raw (no-transform) Subset
# ---------------------------------------------------------------------------

class _TransformWrapper(Dataset):
    def __init__(self, dataset: Dataset, transform) -> None:
        self.dataset   = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = dict(self.dataset[idx])
        img = sample["image"]
        if not isinstance(img, torch.Tensor):
            if hasattr(img, "shape"):
                img = Image.fromarray(img)
            sample["image"] = self.transform(img)
        return sample


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str = "data/datasets",
    backbone_name: str = "vit_small_patch16_dinov3.lvd1689m",
    image_size: int = 448,
    batch_size: int = 16,
    num_workers: int = 8,
    train_ratio: float = 0.8,
) -> tuple[DataLoader, DataLoader, DataLoader, list[float]]:
    """Return (train_dl, val_dl, test_dl, class_weights).

    Pool (JRAIGS+ACRIMA+LAG) is randomly split 80/20 into train/val.
    Test set is ORIGA + G1020 + REFUGE2, held-out for all experiments.
    """
    train_tf, eval_tf = build_transforms(backbone_name, image_size=image_size)

    # ── JRAIGS: all glaucoma + random non-glaucoma up to 25 000 total ────────
    jraigs_full = JRAIGSDataset(data_dir=data_dir, transforms=None)
    glaucoma_idx     = [i for i, (_, lbl) in enumerate(jraigs_full.samples) if lbl == 1]
    non_glaucoma_idx = [i for i, (_, lbl) in enumerate(jraigs_full.samples) if lbl == 0]
    target_total = 25_000
    remaining = max(target_total - len(glaucoma_idx), 0)
    if remaining >= len(non_glaucoma_idx):
        selected_ng = non_glaucoma_idx
    else:
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(non_glaucoma_idx), generator=g)[:remaining].tolist()
        selected_ng = [non_glaucoma_idx[i] for i in perm]
    jraigs_subset = Subset(jraigs_full, glaucoma_idx + selected_ng)
    print(f"JRAIGS subset: {len(glaucoma_idx)} glaucoma + {len(selected_ng)} non-glaucoma = {len(jraigs_subset)} total")

    # ── Build raw pool (no transforms yet) ──────────────────────────────────
    pool = ConcatDataset([
        jraigs_subset,
        ACRIMADataset(data_dir=data_dir, split="train", transforms=None),
        LAGDataset(data_dir=data_dir,    split="train",      transforms=None),
        LAGDataset(data_dir=data_dir,    split="validation", transforms=None),
        LAGDataset(data_dir=data_dir,    split="test",       transforms=None),
    ])

    n_train = int(len(pool) * train_ratio)
    n_val   = len(pool) - n_train
    g = torch.Generator().manual_seed(42)
    train_raw, val_raw = random_split(pool, [n_train, n_val], generator=g)

    class_weights = compute_class_weights(train_raw)

    train_ds = _TransformWrapper(train_raw, train_tf)
    val_ds   = _TransformWrapper(val_raw,   eval_tf)

    # ── Held-out test set ────────────────────────────────────────────────────
    test_parts = [
        ORIGADataset(data_dir=data_dir, transforms=eval_tf),
        REFUGE2Dataset(data_dir=data_dir, split="train", transforms=eval_tf),
    ]
    try:
        test_parts.append(G1020Dataset(data_dir=data_dir, split="test", transforms=eval_tf))
    except FileNotFoundError:
        print("Warning: G1020 not found, skipping from test set.")
    test_ds = ConcatDataset(test_parts)

    # ── WeightedRandomSampler — ensures ~50/50 glaucoma/non-glaucoma per batch ─
    sample_weights = [
        class_weights[_label_at(train_raw.dataset, i)] for i in train_raw.indices
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(42),
    )

    pin = torch.cuda.is_available()
    # persistent_workers=False avoids fork/NFS deadlocks on HPC clusters
    kw  = dict(pin_memory=pin, num_workers=num_workers, persistent_workers=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **kw)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   **kw)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,   **kw)

    print(f"Pool (JRAIGS+ACRIMA+LAG)     : {len(pool):>6} samples")
    print(f"  Train (80%)                : {len(train_ds):>6} samples  [WeightedRandomSampler]")
    print(f"  Val   (20%)                : {len(val_ds):>6} samples")
    print(f"Test  (ORIGA+G1020+REFUGE2)  : {len(test_ds):>6} samples  ← held-out")
    print(f"Class weights                : non-glau={class_weights[0]:.3f}  glau={class_weights[1]:.3f}")

    return train_dl, val_dl, test_dl, class_weights
