"""Evaluate trained generalization models on unified test sets.

Test sets:
  - RIMONE (unified)  = train + test splits merged
  - Fundus (unified)  = train + val splits merged
  - AIRROGS

For each condition (A, B, C, D) loads the best phase-2 checkpoint,
finds the optimal threshold on a held-out val subset, then reports
AUC, Sensitivity, Specificity, F1, Accuracy on each test set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
import torch.nn.functional as F
import timm
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.datasets import (
    AIROGSLightDataset,
    FundusTrainValDataset,
    JRAIGSDataset,
)
from src.datasets.RIMONE import RIMONEDataset
from src.generalization.train_generalization_v3 import DinoV3_1_V2, build_transforms

DATA_DIR = "data/datasets"
FIGURES_DIR = "figures/generalization_v3"

CONDITIONS = {
    "A — Mixed-glaucoma": "checkpoints/generalization_v3/mixed_glaucoma/mixed_glaucoma_phase2_v0-epoch=14-val_auc=0.9717.ckpt",
    "B — JRAIGS-only":    "checkpoints/generalization_v3/jraigs_only/jraigs_only_phase2_v0-epoch=15-val_auc=0.9571.ckpt",
    "C — ACRIMA+ORIGA+LAG": "checkpoints/generalization_v3/acrima_origa_lag/acrima_origa_lag_phase2_v0-epoch=21-val_auc=0.9621.ckpt",
    "D — Harvard-only":   "checkpoints/generalization_v3/harvard_only/harvard_only_phase2_v0-epoch=07-val_auc=0.9684.ckpt",
}

VAL_SPLIT_RATIO = 0.15
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4


# ---------------------------------------------------------------------------
# Build test sets
# ---------------------------------------------------------------------------

def build_test_sets(eval_tf) -> dict:
    """
    RIMONE unified  = RIMONEDataset(train) + RIMONEDataset(test)
    Fundus unified  = FundusTrainValDataset(train) + FundusTrainValDataset(val)
    AIRROGS         = AIROGSLightDataset
    """
    sets = {}

    # RIMONE unified
    try:
        r_train = RIMONEDataset(data_dir=DATA_DIR, split="train", transforms=eval_tf)
        r_test  = RIMONEDataset(data_dir=DATA_DIR, split="test",  transforms=eval_tf)
        sets["RIMONE (unified)"] = ConcatDataset([r_train, r_test])
        print(f"  RIMONE (unified)   {len(r_train)} + {len(r_test)} = {len(sets['RIMONE (unified)'])} samples")
    except Exception as e:
        print(f"  RIMONE: SKIPPED ({e})")

    # Fundus unified
    try:
        f_train = FundusTrainValDataset(data_dir=DATA_DIR, split="train",      transforms=eval_tf)
        f_val   = FundusTrainValDataset(data_dir=DATA_DIR, split="validation", transforms=eval_tf)
        sets["Fundus (unified)"] = ConcatDataset([f_train, f_val])
        print(f"  Fundus (unified)   {len(f_train)} + {len(f_val)} = {len(sets['Fundus (unified)'])} samples")
    except Exception as e:
        print(f"  Fundus: SKIPPED ({e})")

    # AIRROGS
    try:
        sets["AIRROGS"] = AIROGSLightDataset(data_dir=DATA_DIR, transforms=eval_tf)
        print(f"  AIRROGS            {len(sets['AIRROGS'])} samples")
    except Exception as e:
        print(f"  AIRROGS: SKIPPED ({e})")

    return sets


def build_val_set(eval_tf) -> DataLoader:
    """Small held-out val set (from JRAIGS) for threshold calibration."""
    ds = JRAIGSDataset(data_dir=DATA_DIR, transforms=eval_tf)
    g  = torch.Generator().manual_seed(SEED)
    n_val = int(len(ds) * VAL_SPLIT_RATIO)
    idx = torch.randperm(len(ds), generator=g)[:n_val].tolist()
    return DataLoader(Subset(ds, idx), batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect(model, dl: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor]:
    probs_all, labels_all = [], []
    model.eval()
    for batch in dl:
        logits = model(batch["image"].to(device))
        probs_all.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        labels_all.append(batch["label"].cpu())
    return torch.cat(probs_all), torch.cat(labels_all)


def youden_threshold(probs, labels) -> float:
    best_j, best_t = -1.0, 0.5
    for t in torch.linspace(0.01, 0.99, 300):
        preds = (probs >= t).long()
        tp = ((preds == 1) & (labels == 1)).sum().float()
        tn = ((preds == 0) & (labels == 0)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        j = (tp / (tp + fn + 1e-8) + tn / (tn + fp + 1e-8) - 1).item()
        if j > best_j:
            best_j, best_t = j, t.item()
    return best_t


def metrics(probs, labels, thr) -> dict:
    from torchmetrics.classification import (
        BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity,
    )
    return {
        "AUC":         round(BinaryAUROC()(probs, labels).item(), 4),
        "Sensitivity": round(BinaryRecall(threshold=thr)(probs, labels).item(), 4),
        "Specificity": round(BinarySpecificity(threshold=thr)(probs, labels).item(), 4),
        "F1":          round(BinaryF1Score(threshold=thr)(probs, labels).item(), 4),
        "Accuracy":    round(BinaryAccuracy(threshold=thr)(probs, labels).item(), 4),
        "threshold":   round(thr, 3),
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_table(results: dict):
    """
    results = { condition_label: { test_set: {metric: value} } }
    One table per test set, conditions as columns.
    """
    test_sets   = list(next(iter(results.values())).keys())
    conditions  = list(results.keys())
    metric_keys = ["AUC", "Sensitivity", "Specificity", "F1", "Accuracy"]

    for ts in test_sets:
        print(f"\n{'─'*70}")
        print(f"  TEST SET: {ts}")
        print(f"{'─'*70}")
        col_w = 16
        header = f"  {'Metric':<14}" + "".join(f"{c[:col_w]:>{col_w}}" for c in conditions)
        print(header)
        print(f"  {'─'*14}" + "─" * col_w * len(conditions))
        for m in metric_keys:
            row = f"  {m:<14}"
            vals = [results[c][ts].get(m, float("nan")) for c in conditions]
            best = max(vals)
            for v in vals:
                cell = f"{v:.4f}"
                if v == best:
                    cell = f"[{cell}]"  # highlight best
                row += f"{cell:>{col_w}}"
            print(row)
        # threshold row
        row = f"  {'threshold':<14}"
        for c in conditions:
            row += f"{results[c][ts].get('threshold', '?'):>{col_w}}"
        print(row)

    # AUC summary across all test sets
    print(f"\n{'═'*70}")
    print("  AUC SUMMARY")
    print(f"{'═'*70}")
    col_w = 16
    print(f"  {'Test set':<22}" + "".join(f"{c[:col_w]:>{col_w}}" for c in conditions))
    print(f"  {'─'*22}" + "─" * col_w * len(conditions))
    for ts in test_sets:
        row = f"  {ts:<22}"
        vals = [results[c][ts]["AUC"] for c in conditions]
        best = max(vals)
        for v in vals:
            cell = f"{v:.4f}"
            if v == best:
                cell = f"[{cell}]"
            row += f"{cell:>{col_w}}"
        print(row)


def save_figure(results: dict, out_path: Path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    test_sets  = list(next(iter(results.values())).keys())
    conditions = list(results.keys())
    short      = ["A", "B", "C", "D"]
    colors     = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    metrics_to_plot = ["AUC", "Sensitivity", "Specificity", "F1"]

    n_ts = len(test_sets)
    fig, axes = plt.subplots(1, n_ts, figsize=(6 * n_ts, 6), sharey=True)
    if n_ts == 1:
        axes = [axes]

    bar_w = 0.18
    x = np.arange(len(metrics_to_plot))

    for ax, ts in zip(axes, test_sets):
        for i, (cond, s, col) in enumerate(zip(conditions, short, colors)):
            vals = [results[cond][ts].get(m, 0.0) for m in metrics_to_plot]
            bars = ax.bar(x + i * bar_w, vals, bar_w, label=f"{s}: {cond[4:]}", color=col, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

        ax.set_title(ts, fontsize=11, fontweight="bold")
        ax.set_xticks(x + bar_w * 1.5)
        ax.set_xticklabels(metrics_to_plot, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, linestyle="--", color="grey", alpha=0.3, linewidth=0.8)
        ax.grid(axis="y", alpha=0.25)

    handles = [mpatches.Patch(color=col, label=f"{s}: {c[4:]}")
               for s, col, c in zip(short, colors, conditions)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Generalization v3 — Unified test sets", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    _, eval_tf = build_transforms(224)

    print("--- Building test sets ---")
    test_sets = build_test_sets(eval_tf)
    test_dls  = {name: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True)
                 for name, ds in test_sets.items()}

    print("\n--- Calibration val set (JRAIGS subset) ---")
    val_dl = build_val_set(eval_tf)

    results = {}

    for cond_label, ckpt_path in CONDITIONS.items():
        print(f"\n{'='*70}")
        print(f"  {cond_label}")
        print(f"  checkpoint: {Path(ckpt_path).name}")
        print(f"{'='*70}")

        model = DinoV3_1_V2.load_from_checkpoint(ckpt_path).to(device)

        # Calibrate threshold on val set
        val_probs, val_labels = collect(model, val_dl, device)
        thr = youden_threshold(val_probs, val_labels)
        print(f"  Threshold (Youden on JRAIGS val): {thr:.3f}")

        results[cond_label] = {}
        for ts_name, dl in test_dls.items():
            probs, labels = collect(model, dl, device)
            m = metrics(probs, labels, thr)
            results[cond_label][ts_name] = m
            print(f"  {ts_name:<22}  AUC={m['AUC']:.4f}  "
                  f"Sens={m['Sensitivity']:.4f}  Spec={m['Specificity']:.4f}  "
                  f"F1={m['F1']:.4f}  Acc={m['Accuracy']:.4f}")

    print_table(results)

    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    out_json = Path(FIGURES_DIR) / "eval_unified_test_sets.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved: {out_json}")

    save_figure(results, Path(FIGURES_DIR) / "eval_unified_test_sets.png")


if __name__ == "__main__":
    main()
