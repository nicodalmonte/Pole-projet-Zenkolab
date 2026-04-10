"""Distillation training script.

Usage
-----
python src/distillation/train.py \
    --ckpt_eva02_large    checkpoints/teachers/eva02_large/best.ckpt \
    --ckpt_dinov3_large   checkpoints/teachers/dinov3_large/best.ckpt \
    --ckpt_dinov3_huge    checkpoints/teachers/dinov3_huge_plus/best.ckpt \
    [options]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

from src.distillation.data import build_dataloaders
from src.distillation.model import DistillationModule, STUDENT_BACKBONE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-teacher distillation for glaucoma detection")

    # Teacher checkpoints
    p.add_argument("--ckpt_eva02_large",  default="", help="Path to fine-tuned EVA02-Large ckpt")
    p.add_argument("--ckpt_dinov3_large", default="", help="Path to fine-tuned DINOv3-Large ckpt")

    # Data
    p.add_argument("--data_dir",    default="data/datasets")
    p.add_argument("--image_size",  type=int, default=448)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--train_ratio", type=float, default=0.8)

    # Optimisation
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs",   type=int,   default=60)
    p.add_argument("--precision",    default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])

    # Distillation hyper-params
    p.add_argument("--temperature",      type=float, default=4.0)
    p.add_argument("--arcface_margin",   type=float, default=0.3)
    p.add_argument("--arcface_scale",    type=float, default=32.0)
    p.add_argument("--alpha",            type=float, default=1.0, help="Weight for L_KD")
    p.add_argument("--beta",             type=float, default=1.0, help="Weight for L_angular")
    p.add_argument("--gamma",            type=float, default=1.0, help="Weight for L_feat")
    p.add_argument("--phase1_epochs",    type=int,   default=5,
                   help="Epochs for feature-alignment warm-up (Phase 1)")
    p.add_argument("--warmup_epochs_p2", type=int,   default=5,
                   help="Epochs to ramp up alpha and beta in Phase 2")
    p.add_argument("--unfreeze_backbone_epoch", type=int, default=3)

    # Infrastructure
    p.add_argument("--checkpoint_dir", default="checkpoints/distillation/student")
    p.add_argument("--resume",         default=None)
    p.add_argument("--devices",        default="auto")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    print("\n" + "=" * 60)
    print("  Multi-Teacher Distillation — DINOv3-Small student")
    print("  Teachers: EVA02-Large | DINOv3-Large")
    print("=" * 60)

    train_dl, val_dl, test_dl, class_weights = build_dataloaders(
        data_dir=args.data_dir,
        backbone_name=STUDENT_BACKBONE,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
    )

    model = DistillationModule(
        ckpt_eva02_large=args.ckpt_eva02_large,
        ckpt_dinov3_large=args.ckpt_dinov3_large,
        image_size=args.image_size,
        num_classes=2,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        phase1_epochs=args.phase1_epochs,
        warmup_epochs_p2=args.warmup_epochs_p2,
        class_weights=class_weights,
        unfreeze_backbone_epoch=args.unfreeze_backbone_epoch,
    )

    logger = CSVLogger(save_dir="lightning_logs", name="distillation_student")
    version = logger.version
    print(f"Logging to: lightning_logs/distillation_student/version_{version}")

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.checkpoint_dir}/version_{version}",
            filename=f"distill_v{version}-{{epoch:02d}}-{{val_auc:.4f}}",
            monitor="val_auc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val_auc", mode="max", patience=12, min_delta=1e-3, verbose=True),
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

    trainer.fit(model, train_dl, val_dl, ckpt_path=args.resume)

    print("\n" + "=" * 60)
    print("Testing on ORIGA + G1020 + REFUGE2 (held-out)")
    print("=" * 60)
    trainer.test(model, test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()
