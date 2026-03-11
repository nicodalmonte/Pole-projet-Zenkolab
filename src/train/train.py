"""Training script for DinoV3_1 glaucoma classifier.

Dataset split strategy
----------------------
Train  : ACRIMA (all) + Fundus (train) + LAG (train) + ORIGA (all)
Val    : Fundus (validation) + LAG (validation)
Test   : REFUGE2 (train + val + test, held-out, evaluated after training)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import timm
import torch
from torch.utils.data import ConcatDataset, DataLoader
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
)
from src.models.dino_v3_1 import DinoV3_1


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(backbone_name: str, image_size: int = 224):
    """Return (train_transform, eval_transform) derived from the timm model config."""
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    train_tf = timm.data.create_transform(**data_cfg, is_training=True)
    eval_tf = timm.data.create_transform(**data_cfg, is_training=False)
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str,
    backbone_name: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms(backbone_name)

    # --- Train ---
    train_ds = ConcatDataset([
        ACRIMADataset(data_dir=data_dir, split="train", transforms=train_tf),
        FundusTrainValDataset(data_dir=data_dir, split="train", transforms=train_tf),
        LAGDataset(data_dir=data_dir, split="train", transforms=train_tf),
        ORIGADataset(data_dir=data_dir, split="train", transforms=train_tf),
    ])

    # --- Val ---
    val_ds = ConcatDataset([
        FundusTrainValDataset(data_dir=data_dir, split="validation", transforms=eval_tf),
        LAGDataset(data_dir=data_dir, split="validation", transforms=eval_tf),
    ])

    # --- Test (REFUGE2, all three splits) ---
    test_ds = ConcatDataset([
        REFUGE2Dataset(data_dir=data_dir, split="train", transforms=eval_tf),
        REFUGE2Dataset(data_dir=data_dir, split="val", transforms=eval_tf),
        REFUGE2Dataset(data_dir=data_dir, split="test", transforms=eval_tf),
    ])

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
    p = argparse.ArgumentParser(description="Train DinoV3_1 glaucoma classifier")
    p.add_argument("--data_dir", default="data/datasets")
    p.add_argument("--backbone", default="vit_small_patch16_dinov3")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--devices", default="auto")
    p.add_argument("--precision", default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    L.seed_everything(42, workers=True)

    train_dl, val_dl, test_dl = build_dataloaders(
        data_dir=args.data_dir,
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = DinoV3_1(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="dinov3_1-{epoch:02d}-{val_auc:.4f}",
            monitor="val_auc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(leave=True),
        RichModelSummary(max_depth=3),
    ]

    logger = CSVLogger(save_dir="lightning_logs", name="dinov3_1")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        precision=args.precision,
        devices=args.devices,
        log_every_n_steps=10,
        deterministic=False,  # faster with non-deterministic ops
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,  # handled by RichModelSummary
    )

    trainer.fit(model, train_dl, val_dl, ckpt_path=args.resume)

    # --- Test on REFUGE2 using the best checkpoint ---
    print("\n" + "=" * 60)
    print("Testing on REFUGE2 (held-out test set)")
    print("=" * 60)
    trainer.test(model, test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()
