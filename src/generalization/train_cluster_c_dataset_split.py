"""Train on old Cluster C, val on 2 entire new-Cluster-C datasets, test on the rest.

Setup
-----
  Train : JRAIGS + AIRROGS  (old Cluster C, subsampled 1000 pos + 1000 neg)
  Val   : BEH + FIVES        (entire datasets from new Cluster C — used for
                              checkpoint selection and Youden threshold)
  Test  : PAPILA, sjchoi86-HRF, OIA-ODIR, DRISHTI-GS1, CRFO-v4

Rationale: previous version validated on a held-out split of JRAIGS/AIRROGS
(same distribution as train), so early-stopping and threshold calibration were
not representative of the target domain.  Using whole target-domain datasets as
val gives a more honest generalisation estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import lightning as L
from torch.utils.data import ConcatDataset, DataLoader

from src.generalization.Archive.train_generalization_v3 import (
    SEED, BACKBONE,
    DinoV3_1_V2,
    build_transforms,
    get_labels_fast, class_counts,
    _subsample, _make_balanced_dl, _make_dl,
)
from src.datasets import (
    JRAIGSDataset,
    AIROGSLightDataset,
    MultichannelGlaucomaBenchmarkDataset,
)
from lightning.pytorch.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from torchmetrics.classification import (
    BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity,
)

# ── Dataset split ─────────────────────────────────────────────────────────────

VAL_SOURCES = {
    "BEH":   ["BEH"],
    "FIVES": ["FIVES"],
}

TEST_SOURCES = {
    "PAPILA":       ["PAPILA"],
    "sjchoi86-HRF": ["sjchoi86-HRF"],
    "OIA-ODIR":     ["OIA-ODIR-TRAIN", "OIA-ODIR-TEST-ONLINE", "OIA-ODIR-TEST-OFFLINE"],
    "DRISHTI-GS1":  ["DRISHTI-GS1-train", "DRISHTI-GS1-test"],
    "CRFO-v4":      ["CRFO-v4"],
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def build_train_dl(data_dir, train_tf, batch_size, num_workers):
    print("\n--- Train: JRAIGS + AIRROGS (old Cluster C, subsampled) ---")
    pool = ConcatDataset([
        JRAIGSDataset(data_dir=data_dir, transforms=train_tf),
        AIROGSLightDataset(data_dir=data_dir, transforms=train_tf),
    ])
    sub = _subsample(pool, n_pos=1000, n_neg=1000, seed=SEED)
    labels = get_labels_fast(sub)
    n_neg, n_pos = class_counts(labels)
    print(f"  Train subset: {len(sub)} images  (neg={n_neg}, pos={n_pos})")
    return _make_balanced_dl(sub, labels, batch_size, num_workers)


def _load_sources(data_dir, source_dict, tf, batch_size, num_workers, role):
    print(f"\n--- {role} datasets ---")
    dls = {}
    datasets = []
    for name, srcs in source_dict.items():
        try:
            ds = MultichannelGlaucomaBenchmarkDataset(
                data_dir=data_dir, sources=srcs, transforms=tf
            )
            labels = get_labels_fast(ds)
            n_neg, n_pos = class_counts(labels)
            dls[name] = _make_dl(ds, batch_size, num_workers, shuffle=False)
            datasets.append(ds)
            print(f"  {name:<18} {len(ds):>5} samples  (neg={n_neg}, pos={n_pos})")
        except Exception as e:
            print(f"  {name:<18} SKIPPED ({e})")
    return dls, datasets


def build_val_dl(data_dir, eval_tf, batch_size, num_workers):
    dls, datasets = _load_sources(data_dir, VAL_SOURCES, eval_tf, batch_size, num_workers, "Val")
    # Combined val loader (for Lightning fit)
    if datasets:
        combined = ConcatDataset(datasets)
        val_dl = _make_dl(combined, batch_size, num_workers, shuffle=False)
    else:
        raise RuntimeError("No val datasets loaded")
    return val_dl, dls


def build_test_dls(data_dir, eval_tf, batch_size, num_workers):
    dls, _ = _load_sources(data_dir, TEST_SOURCES, eval_tf, batch_size, num_workers, "Test")
    return dls


# ── Training ──────────────────────────────────────────────────────────────────

def train(train_dl, val_dl, args):
    ckpt_dir = str(Path(args.ckpt_dir))
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nSingle phase — backbone unfrozen from start\n{'='*60}")
    model = DinoV3_1_V2(
        backbone_name=BACKBONE, pretrained=True,
        lr=args.lr, img_size=args.img_size,
        dropout=0.35, weight_decay=5e-3,
        unfreeze_backbone_epoch=0,
    )
    logger = CSVLogger("lightning_logs/cluster_c_dataset_split", name="cluster_c_ds")
    v = logger.version
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"cluster_c_ds_v{v}-{{epoch:02d}}-{{val_auc:.4f}}",
            monitor="val_auc", mode="max", save_top_k=1,
        ),
        EarlyStopping(monitor="val_auc", mode="max", patience=8, min_delta=1e-3),
        LearningRateMonitor("epoch"),
        RichProgressBar(leave=True),
    ]
    trainer = L.Trainer(
        max_epochs=args.max_epochs, callbacks=callbacks,
        precision=args.precision, logger=logger,
        log_every_n_steps=10, deterministic=False,
        gradient_clip_val=1.0, gradient_clip_algorithm="norm",
    )
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, train_dl, val_dl)
    return trainer.checkpoint_callback.best_model_path


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect(model, dl, device):
    model.eval()
    ps, ls = [], []
    for batch in dl:
        logits = model(batch["image"].to(device))
        ps.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        ls.append(batch["label"].cpu())
    return torch.cat(ps), torch.cat(ls)


def youden_threshold(probs, labels):
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


def compute_metrics(probs, labels, thr):
    n_pos = int(labels.sum().item())
    n_neg = int((labels == 0).sum().item())
    return {
        "n_total":     n_pos + n_neg,
        "n_pos":       n_pos,
        "n_neg":       n_neg,
        "AUC":         round(BinaryAUROC()(probs, labels).item(), 4),
        "Sensitivity": round(BinaryRecall(threshold=thr)(probs, labels).item(), 4),
        "Specificity": round(BinarySpecificity(threshold=thr)(probs, labels).item(), 4),
        "F1":          round(BinaryF1Score(threshold=thr)(probs, labels).item(), 4),
        "Accuracy":    round(BinaryAccuracy(threshold=thr)(probs, labels).item(), 4),
        "threshold":   round(thr, 3),
    }


def evaluate(ckpt, val_per_ds_dls, test_dls, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV3_1_V2.load_from_checkpoint(ckpt).to(device)

    # Calibrate threshold on all val datasets combined
    all_vp, all_vl = [], []
    for name, dl in val_per_ds_dls.items():
        p, l = collect(model, dl, device)
        all_vp.append(p)
        all_vl.append(l)
    val_p = torch.cat(all_vp)
    val_l = torch.cat(all_vl)
    thr = youden_threshold(val_p, val_l)
    print(f"\n  Threshold (Youden on val BEH+FIVES): {thr:.3f}")

    results = {}

    def _eval_block(dls, role):
        print(f"\n{'='*70}")
        print(f"  {role}")
        print(f"  {'Dataset':<18}  {'n':>5}  {'pos':>5}  {'neg':>5}  {'AUC':>6}  {'Sens':>6}  {'Spec':>6}  {'F1':>6}")
        print(f"  {'-'*66}")
        for name, dl in dls.items():
            p, l = collect(model, dl, device)
            m = compute_metrics(p, l, thr)
            results[name] = m
            print(f"  {name:<18}  {m['n_total']:>5}  {m['n_pos']:>5}  {m['n_neg']:>5}"
                  f"  {m['AUC']:>6.4f}  {m['Sensitivity']:>6.4f}  {m['Specificity']:>6.4f}  {m['F1']:>6.4f}")

    _eval_block(val_per_ds_dls, "Val datasets (BEH, FIVES) — threshold calibration source")
    _eval_block(test_dls,       "Test datasets (held-out)")

    out_path = Path(out_dir) / "results_cluster_c_dataset_split.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    plot_results(results, val_per_ds_dls, Path(out_dir) / "results_cluster_c_dataset_split.png")
    return results


def plot_results(results, val_dls, out_path):
    import matplotlib.pyplot as plt
    import numpy as np

    val_names  = list(val_dls.keys())
    test_names = [k for k in results if k not in val_names]
    datasets   = val_names + test_names
    metrics    = ["AUC", "Sensitivity", "Specificity", "F1"]
    colors     = ["#4363d8", "#e6194b", "#3cb44b", "#f58231"]

    x = np.arange(len(datasets))
    bar_w = 0.18
    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 1.6), 6))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results[d][metric] for d in datasets]
        bars = ax.bar(x + i * bar_w, vals, bar_w, label=metric, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x + bar_w * 1.5)
    ax.set_xticklabels(
        [f"{d}\n({'val' if d in val_names else 'test'})\n(pos={results[d]['n_pos']}/{results[d]['n_total']})"
         for d in datasets],
        fontsize=8,
    )
    ax.set_ylim(0, 1.18)
    ax.axhline(0.5, linestyle="--", color="grey", alpha=0.3, linewidth=0.8)

    # Vertical separator between val and test
    sep = len(val_names) - 0.3
    ax.axvline(sep, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.text(sep / 2, 1.12, "Val (threshold calib.)", ha="center", fontsize=9, color="dimgray")
    ax.text(sep + (len(test_names)) / 2, 1.12, "Test (held-out)", ha="center", fontsize=9, color="dimgray")

    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=10)
    ax.set_title(
        "Train: JRAIGS+AIRROGS  |  Val: BEH+FIVES (entire)  |  Test: 5 held-out datasets",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out_path}")


# ── Args & main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/datasets")
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--max_epochs",  type=int,   default=25)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--ckpt_dir",    default="checkpoints/cluster_c_dataset_split")
    p.add_argument("--out_dir",     default="figures/cluster_c_dataset_split")
    p.add_argument("--precision",   default="16-mixed")
    p.add_argument("--ckpt",        default=None,
                   help="Skip training and load an existing checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(SEED, workers=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.img_size)

    train_dl = build_train_dl(args.data_dir, train_tf, args.batch_size, args.num_workers)
    val_dl, val_per_ds_dls = build_val_dl(args.data_dir, eval_tf, args.batch_size, args.num_workers)
    test_dls = build_test_dls(args.data_dir, eval_tf, args.batch_size, args.num_workers)

    if args.ckpt:
        print(f"\nSkipping training — loading checkpoint: {args.ckpt}")
        ckpt = args.ckpt
    else:
        ckpt = train(train_dl, val_dl, args)
        print(f"\nBest checkpoint: {ckpt}")

    evaluate(ckpt, val_per_ds_dls, test_dls, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
