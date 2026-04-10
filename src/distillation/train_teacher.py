"""Fine-tune teacher models on the distillation data split.

Same split as the distillation pipeline so results are comparable:
  Train : JRAIGS + ACRIMA + LAG  (80%)
  Val   : JRAIGS + ACRIMA + LAG  (20%)
  Test  : ORIGA + G1020 + REFUGE2  ← held-out

Supported models:
  --model eva02_large       EVA-02 Large (448px, patch14)
  --model dinov3_large      DINOv3 Large (448px, patch16)
  --model dinov3_huge_plus  DINOv3 Huge+ (448px, patch16)

Usage
-----
python src/distillation/train_teacher.py --model eva02_large [options]
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
from src.models.eva02_large import EVA02Large
from src.models.dino_v3_1 import DinoV3_1


# ---------------------------------------------------------------------------
# Teacher model registry
# ---------------------------------------------------------------------------

TEACHER_REGISTRY = {
    "eva02_large": {
        "class":           EVA02Large,
        "backbone":        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "image_size":      448,
        "default_batch":   8,
    },
    "dinov3_large": {
        "class":           DinoV3_1,
        "backbone":        "vit_large_patch16_dinov3.lvd1689m",
        "image_size":      448,
        "default_batch":   8,
    },
    "dinov3_huge_plus": {
        "class":           DinoV3_1,
        "backbone":        "vit_huge_plus_patch16_dinov3.lvd1689m",
        "image_size":      448,
        "default_batch":   4,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a teacher model on the distillation split")

    p.add_argument("--model", required=True, choices=list(TEACHER_REGISTRY.keys()),
                   help="Teacher architecture to fine-tune")
    p.add_argument("--data_dir",     default="data/datasets")
    p.add_argument("--batch_size",   type=int, default=None,
                   help="Override default batch size for the selected model")
    p.add_argument("--image_size",   type=int, default=None)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--train_ratio",  type=float, default=0.8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--max_epochs",   type=int,   default=50)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--hidden_dim",   type=int,   default=256)
    p.add_argument("--unfreeze_backbone_epoch", type=int, default=3)
    p.add_argument("--focal_gamma",   type=float, default=0.0,
                   help="Focal loss gamma (only used for eva02_large; 0=plain CE)")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing (only used for eva02_large)")
    p.add_argument("--no_class_weights", action="store_true",
                   help="Disable class-weight correction (use when sampler already balances batches)")
    p.add_argument("--precision",    default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--checkpoint_dir", default="checkpoints/teachers")
    p.add_argument("--resume",         default=None)
    p.add_argument("--devices",        default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    spec       = TEACHER_REGISTRY[args.model]
    backbone   = spec["backbone"]
    image_size = args.image_size or spec["image_size"]
    batch_size = args.batch_size or spec["default_batch"]
    ModelClass = spec["class"]

    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    print("\n" + "=" * 60)
    print(f"  Teacher fine-tuning: {args.model}")
    print(f"  Backbone : {backbone}")
    print(f"  Img size : {image_size}  |  Batch : {batch_size}")
    print("=" * 60)

    train_dl, val_dl, test_dl, class_weights = build_dataloaders(
        data_dir=args.data_dir,
        backbone_name=backbone,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
    )

    effective_class_weights = None if args.no_class_weights else class_weights

    model_kwargs = dict(
        backbone_name=backbone,
        pretrained=True,
        num_classes=2,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=image_size,
        class_weights=effective_class_weights,
        unfreeze_backbone_epoch=args.unfreeze_backbone_epoch,
    )
    if args.model == "eva02_large":
        model_kwargs["focal_gamma"]     = args.focal_gamma
        model_kwargs["label_smoothing"] = args.label_smoothing

    model = ModelClass(**model_kwargs)

    ckpt_dir = Path(args.checkpoint_dir) / args.model
    logger   = CSVLogger(save_dir="lightning_logs", name=f"teacher_{args.model}")
    version  = logger.version
    print(f"Logging to : lightning_logs/teacher_{args.model}/version_{version}")
    print(f"Checkpoints: {ckpt_dir}/version_{version}")

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir / f"version_{version}"),
            filename=f"{args.model}_v{version}-{{epoch:02d}}-{{val_auc:.4f}}",
            monitor="val_auc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val_auc", mode="max", patience=10, min_delta=1e-3, verbose=True),
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
    print(f"Testing {args.model} on ORIGA + G1020 + REFUGE2 (held-out)")
    print("=" * 60)
    trainer.test(model, test_dl, ckpt_path="best")


if __name__ == "__main__":
    main()
