"""Ensemble evaluation: average softmax logits of DINO teacher + EVA ViT.

Usage
-----
uv run src/eval/ensemble_eval.py \
    --dino_ckpt  checkpoints/dino_large/dino_large_lag_v2-epoch=25-val_auc=0.9941.ckpt \
    --eva_ckpt   checkpoints/eva/eva_lag_v2-epoch=38-val_auc=0.9924.ckpt \
    --data_dir   data/datasets \
    --image_size 224
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity

from src.datasets import ACRIMADataset, ORIGADataset, LAGDataset, REFUGE2Dataset
from src.models.dino_v3_1 import DinoV3_1
from src.models.eva_vit import EvaViT


def build_eval_transform(backbone_name: str, image_size: int):
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    data_cfg["input_size"] = (3, image_size, image_size)
    return timm.data.create_transform(**data_cfg, is_training=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble evaluation: DINO + EVA ViT")
    p.add_argument("--dino_ckpt", required=True, help="Path to trained DinoV3_1 checkpoint.")
    p.add_argument("--eva_ckpt", required=True, help="Path to trained EvaViT checkpoint.")
    p.add_argument("--data_dir", default="data/datasets")
    p.add_argument("--image_size", type=int, default=224, help="Image size for eval transforms.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--split",
        default="refuge2",
        choices=["refuge2", "val"],
        help="Which split to evaluate on. 'val' = ACRIMA+ORIGA+LAG, 'refuge2' = REFUGE2 test.",
    )
    return p.parse_args()


@torch.no_grad()
def run_ensemble(
    dino: DinoV3_1,
    eva: EvaViT,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    dino.eval()
    eva.eval()

    auc = BinaryAUROC().to(device)
    acc = BinaryAccuracy().to(device)
    f1 = BinaryF1Score().to(device)
    sensitivity = BinaryRecall().to(device)
    specificity = BinarySpecificity().to(device)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        dino_probs = F.softmax(dino(images), dim=-1)
        eva_probs = F.softmax(eva(images), dim=-1)

        ensemble_probs = 0.5 * dino_probs + 0.5 * eva_probs
        pos_probs = ensemble_probs[:, 1]
        preds = ensemble_probs.argmax(dim=-1)

        auc.update(pos_probs, labels)
        acc.update(preds, labels)
        f1.update(preds, labels)
        sensitivity.update(preds, labels)
        specificity.update(preds, labels)

    return {
        "auc": auc.compute().item(),
        "acc": acc.compute().item(),
        "f1": f1.compute().item(),
        "sensitivity": sensitivity.compute().item(),
        "specificity": specificity.compute().item(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading DINO teacher...")
    dino = DinoV3_1.load_from_checkpoint(args.dino_ckpt)
    dino.to(device)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad = False

    print("Loading EVA ViT...")
    eva = EvaViT.load_from_checkpoint(args.eva_ckpt)
    eva.to(device)
    eva.eval()
    for p in eva.parameters():
        p.requires_grad = False

    # Use DINO's transform for the shared eval (both models receive the same images)
    dino_backbone = dino.hparams.backbone_name
    eval_tf = build_eval_transform(dino_backbone, args.image_size)

    if args.split == "refuge2":
        ds = ConcatDataset([REFUGE2Dataset(data_dir=args.data_dir, split="train", transforms=eval_tf)])
        split_label = "REFUGE2 (test)"
    else:
        ds = ConcatDataset([
            ACRIMADataset(data_dir=args.data_dir, split="train", transforms=eval_tf),
            ORIGADataset(data_dir=args.data_dir, split="train", transforms=eval_tf),
            LAGDataset(data_dir=args.data_dir, split="train", transforms=eval_tf),
        ])
        split_label = "Val (ACRIMA + ORIGA + LAG)"

    pin = torch.cuda.is_available()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    print(f"\nEvaluating ensemble on {split_label} ({len(ds)} samples)...")
    metrics = run_ensemble(dino, eva, loader, device)

    W = 48
    print(f"\n{'=' * W}")
    print(f"  Ensemble results — {split_label}")
    print(f"{'=' * W}")
    for name, val in metrics.items():
        print(f"  {name:<14} {val:.4f}")
    print(f"{'=' * W}")


if __name__ == "__main__":
    main()
