"""
Evaluation script for the trained RETFound glaucoma classifier.

Computes on the test set:
  - AUC-ROC
  - Accuracy
  - Sensitivity (recall for glaucoma class)
  - Specificity
  - F1-score
  - Confusion matrix

Usage:
    python src/evaluate.py --data_root datasets/ --checkpoint checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import GlaucomaDataset
from model import build_retfound


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RETFound glaucoma classifier")
    p.add_argument("--data_root",   type=str, default="datasets/")
    p.add_argument("--checkpoint",  type=str, default="checkpoints/best_model.pth")
    p.add_argument("--split",       type=str, default="test", choices=["test", "validation"])
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir",  type=str, default="results/")
    return p.parse_args()


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    paths  = [b["path"] for b in batch]
    return images, labels, paths


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs, all_paths = [], [], []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = F.softmax(logits.float(), dim=1)[:, 1]
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
            all_paths.extend(paths)

    return np.array(all_labels), np.array(all_probs), all_paths


def save_roc_curve(labels, probs, output_path: Path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve – RETFound Glaucoma Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[eval] ROC curve saved to {output_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset ---
    dataset = GlaucomaDataset(
        Path(args.data_root) / args.split,
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    print(f"[eval] {args.split} samples: {len(dataset)}")

    # --- Load model ---
    model = build_retfound(num_classes=2, load_pretrained=False).to(device)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"[eval] Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(best val AUC={ckpt.get('best_auc', '?'):.4f})")

    # --- Inference ---
    labels, probs, _ = evaluate(model, loader, device)
    preds = (probs >= 0.5).astype(int)

    # --- Metrics ---
    auc  = roc_auc_score(labels, probs)
    acc  = accuracy_score(labels, preds)
    cm   = confusion_matrix(labels, preds)  # [[TN, FP], [FN, TP]]

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "=" * 55)
    print(f"  Evaluation results on [{args.split}] split")
    print("=" * 55)
    print(f"  AUC-ROC      : {auc:.4f}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Sensitivity  : {sensitivity:.4f}  (recall for glaucoma)")
    print(f"  Specificity  : {specificity:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print("=" * 55)
    print(classification_report(labels, preds, target_names=["No glaucoma", "Glaucoma"]))

    # --- ROC curve ---
    save_roc_curve(labels, probs, output_dir / f"roc_{args.split}.png")

    # --- Save predictions to CSV ---
    import csv
    csv_path = output_dir / f"predictions_{args.split}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "prob_glaucoma", "pred"])
        for l, p, pr in zip(labels.tolist(), probs.tolist(), preds.tolist()):
            writer.writerow([l, round(p, 4), pr])
    print(f"[eval] Predictions saved to {csv_path}")


if __name__ == "__main__":
    main()
