"""Threshold sweep — works for student and teacher models.

Collects all probas in one forward pass, then computes
Acc / F1 / Sens / Spec / AUC for every threshold in a grid.

Usage:
    python src/test/threshold_sweep.py \
        --checkpoint checkpoints/distillation/student/version_0/distill_v0-epoch=14-val_auc=0.9882.ckpt \
        --model distillation_student \
        --datasets ORIGA G1020_TEST:test REFUGE2:test
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.distillation.student_eval import StudentEval
from src.train.train import build_transforms
from src.datasets import (
    ACRIMADataset, FundusTrainValDataset, G1020Dataset,
    JRAIGSDataset, LAGDataset, ORIGADataset, REFUGE2Dataset,
)

MODEL_REGISTRY = {
    "distillation_student": None,  # handled separately via from_distillation_ckpt
    "dinov3_large":         "src.models.dino_v3_1:DinoV3_1",
    "eva02_large":          "src.models.eva02_large:EVA02Large",
    "dinov3_1":             "src.models.dino_v3_1:DinoV3_1",
}

DATASET_REGISTRY = {
    "ACRIMA":     ACRIMADataset,
    "FUNDUS":     FundusTrainValDataset,
    "LAG":        LAGDataset,
    "ORIGA":      ORIGADataset,
    "REFUGE2":    REFUGE2Dataset,
    "JRAIGS":     JRAIGSDataset,
    "G1020_TEST": G1020Dataset,
    "G1020_TRAIN":G1020Dataset,
}
DEFAULT_SPLITS = {
    "REFUGE2": "test", "G1020_TEST": "test", "G1020_TRAIN": "train",
    "ACRIMA": "test",  "ORIGA": "train",     "LAG": "test",
    "JRAIGS": "train", "FUNDUS": "validation",
}
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="distillation_student",
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--data_dir", default="data/datasets")
    p.add_argument("--datasets", nargs="+", default=["ORIGA"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def load_model(model_key: str, ckpt_path: str):
    if model_key == "distillation_student":
        return StudentEval.from_distillation_ckpt(ckpt_path)
    spec = MODEL_REGISTRY[model_key]
    module_name, class_name = spec.split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls.load_from_checkpoint(ckpt_path, map_location="cpu")
    return model


@torch.no_grad()
def collect_probs(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in dataloader:
        imgs   = batch["image"].to(device)
        labels = batch["label"]
        logits = model(imgs)
        probs  = F.softmax(logits, dim=-1)[:, 1].cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)


def sweep(probs, labels):
    auc = BinaryAUROC()(probs, labels).item()
    rows = []
    for t in THRESHOLDS:
        preds = (probs >= t).long()
        acc  = BinaryAccuracy()(preds, labels).item()
        f1   = BinaryF1Score()(preds, labels).item()
        sens = BinaryRecall()(preds, labels).item()
        spec = BinarySpecificity()(preds, labels).item()
        rows.append((t, acc, f1, sens, spec))
    return auc, rows


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, args.checkpoint)
    model.to(device)

    backbone_name = str(getattr(model.hparams, "backbone_name"))
    img_size_field = "image_size" if args.model == "distillation_student" else "img_size"
    image_size = int(getattr(model.hparams, img_size_field))

    _, eval_tf = build_transforms(backbone_name=backbone_name, image_size=image_size)

    print(f"\nModel     : {args.model}")
    print(f"Backbone  : {backbone_name}  |  img_size: {image_size}")
    print(f"Checkpoint: {args.checkpoint}")

    for spec in args.datasets:
        name, *rest = spec.split(":")
        name  = name.upper()
        split = rest[0] if rest else DEFAULT_SPLITS[name]

        ds = DATASET_REGISTRY[name](data_dir=args.data_dir, split=split, transforms=eval_tf)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

        print(f"\n{'='*66}")
        print(f"  {name} (split={split}, n={len(ds)})")
        print(f"{'='*66}")

        probs, labels = collect_probs(model, dl, device)

        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos
        print(f"  Class distribution: {n_neg} neg / {n_pos} pos  "
              f"({100*n_pos/len(labels):.1f}% positive)")
        print(f"  Prob range: min={probs.min():.4f}  max={probs.max():.4f}  "
              f"mean={probs.mean():.4f}  median={probs.median():.4f}\n")

        auc, rows = sweep(probs, labels)
        print(f"  AUC (threshold-free): {auc:.4f}\n")
        print(f"  {'Thresh':>6}  {'Acc':>7}  {'F1':>7}  {'Sens':>7}  {'Spec':>7}")
        print(f"  {'-'*42}")
        for t, acc, f1, sens, spec in rows:
            best = " <-- best F1" if f1 == max(r[2] for r in rows) else ""
            print(f"  {t:>6.2f}  {acc:>7.4f}  {f1:>7.4f}  {sens:>7.4f}  {spec:>7.4f}{best}")


if __name__ == "__main__":
    main()
