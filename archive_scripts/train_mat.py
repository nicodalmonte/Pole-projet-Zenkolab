"""Training script for glaucoma classifiers.

Supports multiple model architectures via --model argument:
  - dinov3_1        : DINOv3 ViT-H (default)
  - retfound_dinov2 : RETFound DINOv2 (ophthalmic pretrained)
  - vit_generalist  : ViT-L pretrained on ImageNet-21k

Dataset split strategy
----------------------
Train : JRAIGS (balanced subset, ~8000 samples)
Val   : ACRIMA + ORIGA + LAG
Test  : REFUGE2 (held-out, evaluated after training)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import timm
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import CSVLogger

from src.datasets import (
    ACRIMADataset,
    FundusTrainValDataset,
    LAGDataset,
    ORIGADataset,
    REFUGE2Dataset,
    JRAIGSDataset,
)
from src.models.dino_v3_1 import DinoV3_1
from src.models.eva02_large import EVA02Large
from src.models.retfound_dinov2 import RETFoundDinoV2
from src.models.vit_generalist import ViTGeneralist
from src.datasets.augmentations import AUGMENTATION_TRANSFORMS


# ---------------------------------------------------------------------------
# Model registry — add new architectures here
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "dinov3_1": {
        "class": DinoV3_1,
        "default_backbone": "vit_huge_plus_patch16_dinov3.lvd1689m",
        "default_image_size": 896,
    },
    "eva02_large": {
        "class": EVA02Large,
        "default_backbone": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "default_image_size": 448,
    },
    "retfound_dinov2": {
        "class": RETFoundDinoV2,
        "default_backbone": "hf_hub:YukunZhou/RETFound_dinov2_meh",
        "default_image_size": 448,
    },
    "vit_generalist": {
        "class": ViTGeneralist,
        "default_backbone": "vit_large_patch16_224.augreg_in21k_ft_in1k",
        "default_image_size": 448,
    },
}

# Mapping from HuggingFace Hub model IDs to their timm equivalents
# (used only for resolving data transforms, not for loading weights)
_HF_HUB_TIMM_EQUIVALENT: dict[str, str] = {
    "hf_hub:YukunZhou/RETFound_dinov2_meh": "vit_large_patch16_224",
}


def _timm_name_for_transforms(backbone_name: str) -> str:
    """Return a valid timm model name to use when building transforms.

    HuggingFace Hub backbones (hf_hub:...) are not natively understood by
    timm.data.resolve_model_data_config, so we map them to their timm equivalent.
    """
    if backbone_name.startswith("hf_hub:"):
        return _HF_HUB_TIMM_EQUIVALENT.get(backbone_name, "vit_large_patch16_224")
    return backbone_name


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(backbone_name: str, image_size: int = 896):
    """Return (train_transform, eval_transform) derived from the timm model config.

    Training transforms include data augmentations (rotations, affine transforms, etc.)
    defined in src.augmentations.AUGMENTATION_TRANSFORMS.
    Evaluation transforms do not include augmentations.
    """
    timm_name = _timm_name_for_transforms(backbone_name)
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(timm_name, pretrained=False, num_classes=0)
    )
    # Override the crop size so the full image_size is passed to the backbone.
    data_cfg["input_size"] = (3, image_size, image_size)

    timm_train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    timm_eval_tf  = timm.data.create_transform(**data_cfg, is_training=False)

    train_tf = transforms.Compose([
        AUGMENTATION_TRANSFORMS,
        timm_train_tf,
    ])

    return train_tf, timm_eval_tf


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _count_glaucoma(ds) -> tuple[int, int]:
    """Return (n_glaucoma, total) by inspecting stored paths/labels (no I/O)."""
    if isinstance(ds, Subset):
        base_ds = ds.dataset
        if hasattr(base_ds, "samples"):
            g = sum(base_ds.samples[i][1] for i in ds.indices)
            return g, len(ds)
        return 0, len(ds)

    if isinstance(ds, ACRIMADataset):
        g = sum(1 for p in ds.image_paths if "_g_" in p.name)
    elif isinstance(ds, LAGDataset):
        g = sum(1 for p in ds.image_paths if p.name.startswith("g."))
    elif isinstance(ds, ORIGADataset):
        g = sum(lbl for _, lbl in ds.samples)
    elif isinstance(ds, REFUGE2Dataset):
        g = sum(1 for p in ds.image_paths if p.name.startswith("g"))
    elif isinstance(ds, FundusTrainValDataset):
        g = sum(ds.labels)
    elif isinstance(ds, JRAIGSDataset):
        g = sum(lbl for _, lbl in ds.samples)
    else:
        return 0, len(ds)
    return g, len(ds)


def print_split_info(split_name: str, ds) -> None:
    """Pretty-print size and class balance for a split (ConcatDataset or Dataset)."""
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


def compute_class_weights(train_ds) -> list[float]:
    """Return inverse-frequency class weights [w_neg, w_pos] from the training set."""
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


def build_dataloaders(
    data_dir: str,
    backbone_name: str,
    batch_size: int,
    num_workers: int,
    image_size: int = 896,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms(backbone_name, image_size=image_size)
    target_train_size = 32_000

    # --- Train: JRAIGS balanced subset (all glaucoma + random non-glaucoma) ---
    jraigs_train = JRAIGSDataset(data_dir=data_dir, transforms=train_tf)
    glaucoma_indices     = [i for i, (_, lbl) in enumerate(jraigs_train.samples) if lbl == 1]
    non_glaucoma_indices = [i for i, (_, lbl) in enumerate(jraigs_train.samples) if lbl == 0]

    remaining_slots = max(target_train_size - len(glaucoma_indices), 0)
    if remaining_slots >= len(non_glaucoma_indices):
        selected_non_glaucoma = non_glaucoma_indices
    else:
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(non_glaucoma_indices), generator=g)[:remaining_slots].tolist()
        selected_non_glaucoma = [non_glaucoma_indices[i] for i in perm]

    train_ds = ConcatDataset([Subset(jraigs_train, glaucoma_indices + selected_non_glaucoma)])

    # --- Val: ACRIMA + ORIGA + LAG ---
    val_ds = ConcatDataset([
        ACRIMADataset(data_dir=data_dir, split="train", transforms=eval_tf),
        ORIGADataset(data_dir=data_dir,  split="train", transforms=eval_tf),
        LAGDataset(data_dir=data_dir,    split="train", transforms=eval_tf),
    ])

    # --- Test: REFUGE2 (held-out) ---
    test_ds = ConcatDataset([
        REFUGE2Dataset(data_dir=data_dir, split="train", transforms=eval_tf),
    ])

    print_split_info("TRAIN", train_ds)
    print_split_info("VAL  ", val_ds)
    print_split_info("TEST (REFUGE2)", test_ds)

    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
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

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    print(f"Test  samples : {len(test_ds)} (REFUGE2)")

    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a glaucoma classifier")

    # --- Model selection ---
    p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="dinov3_1",
        help="Architecture to train. Each model has sensible defaults for backbone "
             "and image_size that can be overridden with --backbone / --image_size.",
    )
    p.add_argument(
        "--backbone",
        default=None,
        help="Backbone name (timm or hf_hub:...). Defaults to the model's registry default.",
    )
    p.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Input image size (must be a multiple of 16). "
             "Defaults to the model's registry default (896 for dinov3_1, 448 for others).",
    )

    # --- Training hyperparameters ---
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--hidden_dim",   type=int,   default=256)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--unfreeze_backbone_epoch", type=int, default=3,
        help="Freeze backbone for this many epochs, then unfreeze. 0 = always unfrozen.",
    )
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--max_epochs",  type=int, default=50)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--class_weights", type=float, nargs=2, default=None,
        metavar=("W_NEG", "W_POS"),
        help="Manual class weights [non-glaucoma, glaucoma]. "
             "Computed automatically from training set if omitted.",
    )

    # --- Infrastructure ---
    p.add_argument("--data_dir",       default="data/datasets")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume",         default=None, help="Checkpoint path to resume from")
    p.add_argument("--devices",        default="auto")
    p.add_argument("--precision",      default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve model-specific defaults
    registry   = MODEL_REGISTRY[args.model]
    backbone   = args.backbone   or registry["default_backbone"]
    image_size = args.image_size or registry["default_image_size"]
    ModelClass = registry["class"]

    print(f"\n{'=' * 60}")
    print(f"  Model      : {args.model}")
    print(f"  Backbone   : {backbone}")
    print(f"  Image size : {image_size}")
    print(f"{'=' * 60}")

    L.seed_everything(42, workers=True)

    train_dl, val_dl, test_dl = build_dataloaders(
        data_dir=args.data_dir,
        backbone_name=backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
    )

    class_weights = (
        args.class_weights
        if args.class_weights is not None
        else compute_class_weights(train_dl.dataset)
    )
    print(f"\nClass weights — non-glaucoma: {class_weights[0]:.4f}  glaucoma: {class_weights[1]:.4f}")

    model = ModelClass(
        backbone_name=backbone,
        pretrained=args.pretrained,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=image_size,
        class_weights=class_weights,
        unfreeze_backbone_epoch=args.unfreeze_backbone_epoch,
    )

    logger = CSVLogger(save_dir="lightning_logs", name=args.model)
    version_number = logger.version
    print(f"Logging to: lightning_logs/{args.model}/version_{version_number}")

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.checkpoint_dir}/{args.model}/version_{version_number}",
            filename=f"{args.model}_v{version_number}-" + "{epoch:02d}-{val_auc:.4f}",
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
    print("Testing on REFUGE2 (held-out test set)")
    print("=" * 60)
    trainer.test(model, test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()