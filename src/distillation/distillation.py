"""Training script for distilled student glaucoma classifier.

Dataset split strategy
----------------------
Train  : ACRIMA (all) + Fundus (train) + LAG (train) + ORIGA (all)
Val    : Fundus (validation) + LAG (validation)
Test   : REFUGE2 (train, held-out, evaluated after training)

Notes
-----
- The teacher is loaded from a trained checkpoint and kept frozen in eval mode.
- Only the student is trained.
- The student backbone is NOT frozen.
- The student is created with pretrained=True.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import timm
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from torchvision import transforms

from src.datasets import (
    ACRIMADataset,
    FundusTrainValDataset,
    LAGDataset,
    ORIGADataset,
    REFUGE2Dataset,
    JRAIGSDataset,
)
from src.models.dino_v3_1 import DinoV3_1
from src.models.student import StudentGlaucomaDistilled

from src.datasets.augmentations import AUGMENTATION_TRANSFORMS


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(backbone_name: str, image_size: int = 896):
    """Return (train_transform, eval_transform) derived from the timm model config."""
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    data_cfg["input_size"] = (3, image_size, image_size)

    timm_train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    timm_eval_tf = timm.data.create_transform(**data_cfg, is_training=False)

    train_tf = transforms.Compose([
        AUGMENTATION_TRANSFORMS,
        timm_train_tf,
    ])

    return train_tf, timm_eval_tf


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _count_glaucoma(ds) -> tuple[int, int]:
    """Return (n_glaucoma, total) by inspecting stored paths/labels.

    No images are opened — only filename metadata / label lists are read.
    """
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


def compute_class_weights(train_ds) -> list[float]:
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


def build_dataloaders(
    data_dir: str,
    backbone_name: str,
    batch_size: int,
    num_workers: int,
    image_size: int = 896,
    dataset: str = "jraigs",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms(backbone_name, image_size=image_size)

    if dataset == "lag_origa":
        train_ds = ConcatDataset([LAGDataset(data_dir=data_dir, split="train", transforms=train_tf)])
        val_ds = ConcatDataset([LAGDataset(data_dir=data_dir, split="validation", transforms=eval_tf)])
        test_ds = ConcatDataset([ORIGADataset(data_dir=data_dir, transforms=eval_tf)])
    else:  # jraigs (default)
        target_train_size = 8_000

        jraigs_train = JRAIGSDataset(data_dir=data_dir, transforms=train_tf)
        glaucoma_indices = [i for i, (_, lbl) in enumerate(jraigs_train.samples) if lbl == 1]
        non_glaucoma_indices = [i for i, (_, lbl) in enumerate(jraigs_train.samples) if lbl == 0]

        remaining_slots = max(target_train_size - len(glaucoma_indices), 0)
        if remaining_slots >= len(non_glaucoma_indices):
            selected_non_glaucoma = non_glaucoma_indices
        else:
            g = torch.Generator().manual_seed(42)
            perm = torch.randperm(len(non_glaucoma_indices), generator=g)[:remaining_slots].tolist()
            selected_non_glaucoma = [non_glaucoma_indices[i] for i in perm]

        selected_indices = glaucoma_indices + selected_non_glaucoma
        train_ds = ConcatDataset([Subset(jraigs_train, selected_indices)])

        val_ds = ConcatDataset([
            ACRIMADataset(data_dir=data_dir, split="train", transforms=eval_tf),
            ORIGADataset(data_dir=data_dir, split="train", transforms=eval_tf),
            LAGDataset(data_dir=data_dir, split="train", transforms=eval_tf),
        ])
        test_ds = ConcatDataset([REFUGE2Dataset(data_dir=data_dir, split="train", transforms=eval_tf)])

    print_split_info("TRAIN", train_ds)
    print_split_info("VAL  ", val_ds)
    print_split_info("TEST ", test_ds)

    pin = torch.cuda.is_available()

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    print(f"Test  samples : {len(test_ds)}")

    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train distilled student glaucoma classifier")

    p.add_argument("--data_dir", default="data/datasets")

    # Teacher
    p.add_argument(
        "--teacher_ckpt",
        type=str,
        required=True,
        help="Path to trained DinoV3_1 checkpoint.",
    )
    p.add_argument(
        "--teacher_backbone",
        default="vit_huge_plus_patch16_dinov3.lvd1689m",
        help="Teacher backbone name used when the checkpoint was trained.",
    )

    # Student
    p.add_argument(
        "--student_backbone",
        default="fastvit_t8.apple_dist_in1k",
        help="Student timm backbone.",
    )
    p.add_argument(
        "--student_pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights for the student. Default is False.",
    )
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)

    # Optimization
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=16)

    # Distillation
    p.add_argument("--kd_alpha", type=float, default=0.7)
    p.add_argument("--kd_temperature", type=float, default=4.0)

    # Misc
    p.add_argument("--checkpoint_dir", default="checkpoints/student")
    p.add_argument("--resume", default=None, help="Path to student checkpoint to resume from")
    p.add_argument("--devices", default="auto")
    p.add_argument("--dataset", default="jraigs", choices=["jraigs", "lag_origa"],
                   help="Dataset config: 'jraigs' (default) or 'lag_origa' (LAG train, ORIGA test).")
    p.add_argument("--precision", default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument(
        "--image_size",
        type=int,
        default=896,
        help="Input image size. Must be compatible with the student backbone.",
    )
    p.add_argument(
        "--class_weights",
        type=float,
        nargs=2,
        default=None,
        metavar=("W_NEG", "W_POS"),
        help="Manual class weights for [non-glaucoma, glaucoma]. "
             "If omitted, weights are computed from the training split.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    L.seed_everything(42, workers=True)

    train_dl, val_dl, test_dl = build_dataloaders(
        data_dir=args.data_dir,
        backbone_name=args.student_backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        dataset=args.dataset,
    )

    class_weights = (
        args.class_weights
        if args.class_weights is not None
        else compute_class_weights(train_dl.dataset)
    )
    print(
        f"\nClass weights — non-glaucoma: {class_weights[0]:.4f}  "
        f"glaucoma: {class_weights[1]:.4f}"
    )

    # ---------------------------------------------------------
    # Load trained teacher
    # ---------------------------------------------------------
    teacher = DinoV3_1.load_from_checkpoint(args.teacher_ckpt)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---------------------------------------------------------
    # Student
    # ---------------------------------------------------------
    student = StudentGlaucomaDistilled(
        backbone_name=args.student_backbone,
        pretrained=args.student_pretrained,   # default True
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.image_size,
        class_weights=class_weights,
        teacher_model=teacher,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        freeze_backbone_epochs=0,   # no freezing at all
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="student_single-{epoch:02d}-{val_auc:.4f}",
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

    logger = CSVLogger(save_dir="lightning_logs", name="student_distilled")

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

    trainer.fit(student, train_dl, val_dl, ckpt_path=args.resume)

    print("\n" + "=" * 60)
    print("Testing distilled student on held-out test set")
    print("=" * 60)
    trainer.test(student, test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()