"""Ensemble evaluation with model-specific transforms.

DINO is evaluated at 224 px (its training resolution).
EVA ViT is evaluated at 448 px (its training resolution).
Softmax probabilities from both models are averaged for the final prediction.

Usage
-----
uv run src/eval/ensemble_eval_dual_tf.py \
    --dino_ckpt  checkpoints/dino_large/dino_large_lag_v2-epoch=25-val_auc=0.9941.ckpt \
    --eva_ckpt   checkpoints/eva/eva_lag_v2-epoch=38-val_auc=0.9924.ckpt \
    --data_dir   data/datasets \
    --split      refuge2
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
)

from src.datasets import ACRIMADataset, ORIGADataset, LAGDataset, REFUGE2Dataset
from src.models.dino_v3_1 import DinoV3_1
from src.models.eva_vit import EvaViT


def build_eval_transform(backbone_name: str, image_size: int):
    # Resolve the backbone's default data config (mean, std, crop ratio, etc.) without downloading weights
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    # Override the spatial resolution to match the model's training resolution
    data_cfg["input_size"] = (3, image_size, image_size)
    # Build the inference-time transform pipeline (no random augmentation)
    return timm.data.create_transform(**data_cfg, is_training=False)


def build_loader(
    ds_specs: list[tuple],
    transform,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    # Instantiate each dataset with the given transform
    datasets = [cls(**kwargs, transforms=transform) for cls, kwargs in ds_specs]
    # Merge into a single dataset when there are multiple sources
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    # shuffle=False is critical: both loaders must iterate in the same sample order
    pin = torch.cuda.is_available()
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ensemble evaluation: DINO (224 px) + EVA ViT (448 px)"
    )
    # Path to the best DinoV3_1 Lightning checkpoint
    p.add_argument("--dino_ckpt", required=True, help="Path to trained DinoV3_1 checkpoint.")
    # Path to the best EvaViT Lightning checkpoint
    p.add_argument("--eva_ckpt", required=True, help="Path to trained EvaViT checkpoint.")
    p.add_argument("--data_dir", default="data/datasets")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--split",
        default="refuge2",
        choices=["refuge2", "val", "origa"],
        help="'refuge2' = REFUGE2 held-out test set; 'val' = ACRIMA+ORIGA+LAG; 'origa' = ORIGA held-out test set.",
    )
    return p.parse_args()


@torch.no_grad()
def run_ensemble(
    dino: DinoV3_1,
    eva: EvaViT,
    dino_loader: DataLoader,
    eva_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    # Put both models in eval mode (disables dropout, batchnorm uses running stats)
    dino.eval()
    eva.eval()

    # Initialise torchmetrics accumulators on the target device
    auc = BinaryAUROC().to(device)
    acc = BinaryAccuracy().to(device)
    f1 = BinaryF1Score().to(device)
    sensitivity = BinaryRecall().to(device)
    specificity = BinarySpecificity().to(device)

    # zip guarantees both loaders advance in lockstep — same samples, different resolutions
    for dino_batch, eva_batch in zip(dino_loader, eva_loader):
        # DINO images are 224×224
        dino_images = dino_batch["image"].to(device)
        # EVA images are 448×448
        eva_images = eva_batch["image"].to(device)
        # Labels are identical in both batches; use the DINO batch copy
        labels = dino_batch["label"].to(device)

        # Forward DINO and convert raw logits to class probabilities
        dino_probs = F.softmax(dino(dino_images), dim=-1)
        # Forward EVA and convert raw logits to class probabilities
        eva_probs = F.softmax(eva(eva_images), dim=-1)

        # Simple average ensemble: equal weight to both models
        ensemble_probs = 0.5 * dino_probs + 0.5 * eva_probs
        # Probability of the positive class (glaucoma = index 1) used by AUC
        pos_probs = ensemble_probs[:, 1]
        # Hard class prediction from the highest averaged probability
        preds = ensemble_probs.argmax(dim=-1)

        # Feed predictions and labels into each metric accumulator
        auc.update(pos_probs, labels)
        acc.update(preds, labels)
        f1.update(preds, labels)
        sensitivity.update(preds, labels)
        specificity.update(preds, labels)

    # Compute final metric values from the accumulated state
    return {
        "auc": auc.compute().item(),
        "acc": acc.compute().item(),
        "f1": f1.compute().item(),
        "sensitivity": sensitivity.compute().item(),
        "specificity": specificity.compute().item(),
    }


def main() -> None:
    args = parse_args()
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading DINO checkpoint...")
    # Lightning restores all hparams (including backbone_name) from the checkpoint
    dino = DinoV3_1.load_from_checkpoint(args.dino_ckpt)
    dino.to(device)
    dino.eval()
    # Freeze weights — no gradient computation needed during inference
    for p in dino.parameters():
        p.requires_grad = False

    print("Loading EVA ViT checkpoint...")
    # Lightning restores all hparams (including backbone_name and img_size) from the checkpoint
    eva = EvaViT.load_from_checkpoint(args.eva_ckpt)
    eva.to(device)
    eva.eval()
    # Freeze weights — no gradient computation needed during inference
    for p in eva.parameters():
        p.requires_grad = False

    # Build DINO eval transform using its backbone config at its training resolution (224 px)
    dino_tf = build_eval_transform(dino.hparams.backbone_name, 224)
    # Build EVA eval transform using its backbone config at its training resolution (448 px)
    eva_tf = build_eval_transform(eva.hparams.backbone_name, 448)

    # Dataset specs: list of (DatasetClass, init_kwargs_without_transforms)
    if args.split == "refuge2":
        ds_specs = [(REFUGE2Dataset, {"data_dir": args.data_dir, "split": "train"})]
        split_label = "REFUGE2 (test)"
    elif args.split == "origa":
        ds_specs = [(ORIGADataset, {"data_dir": args.data_dir, "split": "train"})]
        split_label = "ORIGA (test)"
    else:
        ds_specs = [
            (ACRIMADataset, {"data_dir": args.data_dir, "split": "train"}),
            (ORIGADataset, {"data_dir": args.data_dir, "split": "train"}),
            (LAGDataset, {"data_dir": args.data_dir, "split": "train"}),
        ]
        split_label = "Val (ACRIMA + ORIGA + LAG)"

    # Build two independent loaders — identical sample order, different transforms
    dino_loader = build_loader(ds_specs, dino_tf, args.batch_size, args.num_workers)
    eva_loader = build_loader(ds_specs, eva_tf, args.batch_size, args.num_workers)

    n_samples = len(dino_loader.dataset)
    print(f"\nEvaluating ensemble on {split_label} ({n_samples} samples)...")
    metrics = run_ensemble(dino, eva, dino_loader, eva_loader, device)

    W = 48
    print(f"\n{'=' * W}")
    print(f"  Ensemble results — {split_label}")
    print(f"{'=' * W}")
    for name, val in metrics.items():
        print(f"  {name:<14} {val:.4f}")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
