"""DINOv3-Small generalization experiment.

Goal: show that DINOv3-Small trained on a *diverse* mix of datasets generalises
better to completely held-out datasets (REFUGE2, RIM-ONE) than the same model
trained on JRAIGS only.

Two conditions trained sequentially with identical hyperparameters:
  Condition A — JRAIGS-only : 1 000 balanced samples from JRAIGS
  Condition B — Mixed        : 1 000 JRAIGS + 1 000 diverse pool (balanced each)

Class balance is enforced with WeightedRandomSampler (50/50 glaucoma/non-glaucoma).

Diverse pool (Condition B): ACRIMA · ORIGA · LAG(train) · AIROGSLight
                             FundusTrainVal(train+val) · Harvard(binary)
Excluded from training: G1020 (unusable) · REFUGE2 · RIM-ONE (held-out test)

Test (zero-shot, never seen during training):
  - REFUGE2  : train split (only split with public labels)
  - RIM-ONE  : train + test splits combined

Training strategy (identical for both conditions):
  Phase 1 : backbone frozen,   lr=1e-3, 25 epochs
  Phase 2 : backbone unfrozen, lr=1e-4, 25 epochs

Outputs
-------
  lightning_logs/generalization/
  checkpoints/generalization/{jraigs_only,mixed}/
  figures/generalization/
    comparison_auc.png   test_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import timm
import torch
from torch.utils.data import (
    ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler,
)
from torchvision import transforms
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import CSVLogger

from src.datasets import (
    ACRIMADataset,
    ORIGADataset,
    LAGDataset,
    AIROGSLightDataset,
    FundusTrainValDataset,
    HarvardGlaucomaDataset,
    JRAIGSDataset,
    REFUGE2Dataset,
)
from src.datasets.RIMONE import RIMONEDataset
from src.datasets.augmentations import AUGMENTATION_TRANSFORMS
from src.models.dino_v3_1 import DinoV3_1

SEED = 42
BACKBONE = "vit_small_patch16_dinov3.lvd1689m"
JRAIGS_N = 1_000
POOL_N = 1_000


# ---------------------------------------------------------------------------
# Harvard binary wrapper  (0=normal → 0 ; 1=early, 2=advanced → 1)
# ---------------------------------------------------------------------------

class BinaryMappedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        sample["label"] = torch.tensor(min(int(sample["label"]), 1), dtype=torch.long)
        return sample


# ---------------------------------------------------------------------------
# Fast label extraction (no image loading)
# ---------------------------------------------------------------------------

def get_labels_fast(dataset: Dataset) -> list[int]:
    """Extract integer labels for every sample without loading images."""
    if isinstance(dataset, Subset):
        parent = get_labels_fast(dataset.dataset)
        return [parent[i] for i in dataset.indices]

    if isinstance(dataset, ConcatDataset):
        out: list[int] = []
        for ds in dataset.datasets:
            out.extend(get_labels_fast(ds))
        return out

    if isinstance(dataset, BinaryMappedDataset):
        return [min(int(l), 1) for l in get_labels_fast(dataset.dataset)]

    # Datasets with explicit .labels list
    if hasattr(dataset, "labels"):
        return [int(l) for l in dataset.labels]  # type: ignore[attr-defined]

    # Datasets with .samples list of (path, label) tuples
    if hasattr(dataset, "samples"):
        return [int(s[1]) for s in dataset.samples]  # type: ignore[attr-defined]

    # ACRIMA: label encoded in filename ("_g_" → glaucoma)
    if isinstance(dataset, ACRIMADataset):
        return [1 if "_g_" in Path(p).name else 0 for p in dataset.image_paths]

    # LAG: label encoded in filename prefix ("g." → glaucoma)
    if isinstance(dataset, LAGDataset):
        return [1 if Path(p).name.startswith("g.") else 0 for p in dataset.image_paths]

    raise ValueError(f"Cannot extract labels from {type(dataset).__name__}")


def class_counts(labels: list[int]) -> tuple[int, int]:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    return n_neg, n_pos


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(img_size: int = 224):
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(BACKBONE, pretrained=False, num_classes=0)
    )
    data_cfg["input_size"] = (3, img_size, img_size)
    eval_tf = timm.data.create_transform(**data_cfg, is_training=False)
    train_tf = transforms.Compose([AUGMENTATION_TRANSFORMS, eval_tf])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _subsample(dataset: Dataset, n: int, seed: int) -> Subset:
    total = len(dataset)  # type: ignore[arg-type]
    if total <= n:
        return Subset(dataset, list(range(total)))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total, generator=g)[:n].tolist()
    return Subset(dataset, idx)


def _train_val_split(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    k = int(n * (1.0 - val_ratio))
    return perm[:k], perm[k:]


def _make_dl(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _make_balanced_dl(
    ds: Dataset,
    labels: list[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """DataLoader with WeightedRandomSampler enforcing 50/50 class balance."""
    n_neg, n_pos = class_counts(labels)
    if n_pos == 0 or n_neg == 0:
        print(f"  WARNING: degenerate class distribution (neg={n_neg}, pos={n_pos}), using shuffle")
        return _make_dl(ds, batch_size, num_workers, shuffle=True)

    w_pos = 1.0 / n_pos
    w_neg = 1.0 / n_neg
    weights = torch.tensor([w_pos if l == 1 else w_neg for l in labels])
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    print(f"  Class balance  neg={n_neg}  pos={n_pos}  (WeightedRandomSampler)")
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _load_jraigs(data_dir: str, tf, seed: int) -> Subset:
    ds = JRAIGSDataset(data_dir=data_dir, transforms=tf)
    sampled = _subsample(ds, JRAIGS_N, seed)
    labels = get_labels_fast(sampled)
    n_neg, n_pos = class_counts(labels)
    print(f"  JRAIGS: {len(ds)} total → {len(sampled)} sampled  (neg={n_neg} pos={n_pos})")
    return sampled


def _load_pool(data_dir: str, tf, seed: int) -> Subset:
    candidates: list[Dataset] = []

    def try_load(name: str, fn):
        try:
            ds = fn()
            candidates.append(ds)
            labels = get_labels_fast(ds)
            n_neg, n_pos = class_counts(labels)
            print(f"  Pool | {name}: {len(ds)} samples  (neg={n_neg} pos={n_pos})")  # type: ignore[arg-type]
        except Exception as e:
            print(f"  Pool | {name}: SKIPPED ({e})")

    try_load("ACRIMA",                lambda: ACRIMADataset(data_dir=data_dir, transforms=tf))
    try_load("ORIGA",                 lambda: ORIGADataset(data_dir=data_dir, transforms=tf))
    try_load("LAG(train)",            lambda: LAGDataset(data_dir=data_dir, split="train", transforms=tf))
    try_load("AIROGSLight",           lambda: AIROGSLightDataset(data_dir=data_dir, transforms=tf))
    try_load("FundusTrainVal(train)",  lambda: FundusTrainValDataset(data_dir=data_dir, split="train", transforms=tf))
    try_load("FundusTrainVal(val)",   lambda: FundusTrainValDataset(data_dir=data_dir, split="validation", transforms=tf))
    try_load("Harvard(binary)",       lambda: BinaryMappedDataset(HarvardGlaucomaDataset(data_dir=data_dir, transforms=tf)))

    if not candidates:
        raise RuntimeError("No pool datasets could be loaded.")

    combined = ConcatDataset(candidates)
    sampled = _subsample(combined, POOL_N, seed)
    labels = get_labels_fast(sampled)
    n_neg, n_pos = class_counts(labels)
    print(f"  Pool total: {len(combined)} → {len(sampled)} sampled  (neg={n_neg} pos={n_pos})")
    return sampled


def _split_and_build_loaders(
    train_ds: Subset,
    eval_ds: Subset,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    label: str,
) -> tuple[DataLoader, DataLoader]:
    n = len(train_ds)  # type: ignore[arg-type]
    train_idx, val_idx = _train_val_split(n, val_ratio, SEED)

    train_subset = Subset(train_ds, train_idx)
    val_subset   = Subset(eval_ds,  val_idx)

    train_labels = get_labels_fast(train_subset)
    print(f"  {label} → train={len(train_idx)}  val={len(val_idx)}")

    train_dl = _make_balanced_dl(train_subset, train_labels, batch_size, num_workers)
    val_dl   = _make_dl(val_subset, batch_size, num_workers, shuffle=False)
    return train_dl, val_dl


def build_jraigs_only_dataloaders(
    data_dir: str, train_tf, eval_tf,
    batch_size: int, num_workers: int, val_ratio: float,
) -> tuple[DataLoader, DataLoader]:
    print("\n--- Condition A: JRAIGS-only ---")
    jraigs_train = _load_jraigs(data_dir, train_tf, SEED)
    jraigs_eval  = _load_jraigs(data_dir, eval_tf,  SEED)
    return _split_and_build_loaders(
        jraigs_train, jraigs_eval, val_ratio, batch_size, num_workers, "JRAIGS-only"
    )


def build_mixed_dataloaders(
    data_dir: str, train_tf, eval_tf,
    batch_size: int, num_workers: int, val_ratio: float,
) -> tuple[DataLoader, DataLoader]:
    print("\n--- Condition B: Mixed (JRAIGS + diverse pool) ---")
    jraigs_train = _load_jraigs(data_dir, train_tf, SEED)
    jraigs_eval  = _load_jraigs(data_dir, eval_tf,  SEED)
    pool_train   = _load_pool(data_dir, train_tf, SEED)
    pool_eval    = _load_pool(data_dir, eval_tf,  SEED)

    combined_train = ConcatDataset([jraigs_train, pool_train])
    combined_eval  = ConcatDataset([jraigs_eval,  pool_eval])
    return _split_and_build_loaders(
        combined_train, combined_eval, val_ratio, batch_size, num_workers, "Mixed"  # type: ignore[arg-type]
    )


def build_test_dataloaders(data_dir: str, eval_tf, batch_size: int, num_workers: int) -> dict[str, DataLoader]:
    print("\n--- Test datasets (held-out, zero-shot) ---")
    factories = {
        "REFUGE2(train)": lambda: REFUGE2Dataset(data_dir=data_dir, split="train", transforms=eval_tf),
        "RIMONE(train)":  lambda: RIMONEDataset(data_dir=data_dir, split="train",  transforms=eval_tf),
        "RIMONE(test)":   lambda: RIMONEDataset(data_dir=data_dir, split="test",   transforms=eval_tf),
    }
    dls: dict[str, DataLoader] = {}
    for name, fn in factories.items():
        try:
            ds = fn()
            labels = get_labels_fast(ds)
            n_neg, n_pos = class_counts(labels)
            dls[name] = _make_dl(ds, batch_size, num_workers)
            print(f"  {name}: {len(ds)} samples  (neg={n_neg} pos={n_pos})")
        except Exception as e:
            print(f"  {name}: SKIPPED ({e})")
    return dls


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_phase(
    model: DinoV3_1,
    train_dl: DataLoader,
    val_dl: DataLoader,
    max_epochs: int,
    log_name: str,
    ckpt_dir: str,
    precision: str,
) -> tuple[str, str]:
    logger = CSVLogger(save_dir="lightning_logs/generalization", name=log_name)
    v = logger.version

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{log_name}_v{v}" + "-{epoch:02d}-{val_auc:.4f}",
            monitor="val_auc", mode="max",
            save_top_k=1, save_last=False,
        ),
        EarlyStopping(monitor="val_auc", mode="max", patience=7, min_delta=1e-3, verbose=True),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(leave=True),
    ]
    trainer = L.Trainer(
        max_epochs=max_epochs, callbacks=callbacks,
        precision=precision, logger=logger,
        log_every_n_steps=10, deterministic=False,
    )
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, train_dl, val_dl)
    best = trainer.checkpoint_callback.best_model_path
    print(f"  Best checkpoint: {best}")
    return best, logger.log_dir


def train_condition(
    label: str,
    train_dl: DataLoader,
    val_dl: DataLoader,
    args,
    ckpt_dir: str,
    figures_dir: str,
) -> str:
    slug = label.lower().replace(" ", "_").replace("-", "_")

    print("\n" + "=" * 60)
    print(f"[{label}] Phase 1 — backbone frozen, lr={args.lr1}")
    print("=" * 60)
    model_p1 = DinoV3_1(
        backbone_name=BACKBONE, pretrained=True,
        lr=args.lr1, img_size=args.img_size,
        unfreeze_backbone_epoch=args.max_epochs + 1,
    )
    best1, log_dir1 = run_phase(
        model_p1, train_dl, val_dl,
        max_epochs=args.max_epochs,
        log_name=f"{slug}_phase1",
        ckpt_dir=ckpt_dir,
        precision=args.precision,
    )
    plot_training_curves(log_dir1, f"{label} — Phase 1", figures_dir)

    print("\n" + "=" * 60)
    print(f"[{label}] Phase 2 — backbone unfrozen, lr={args.lr2}")
    print("=" * 60)
    model_p2 = DinoV3_1.load_from_checkpoint(best1, lr=args.lr2, unfreeze_backbone_epoch=0)
    best2, log_dir2 = run_phase(
        model_p2, train_dl, val_dl,
        max_epochs=args.max_epochs,
        log_name=f"{slug}_phase2",
        ckpt_dir=ckpt_dir,
        precision=args.precision,
    )
    plot_training_curves(log_dir2, f"{label} — Phase 2", figures_dir)
    return best2


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all(ckpt_path: str, test_dls: dict[str, DataLoader], precision: str) -> dict:
    results: dict[str, dict] = {}
    for ds_name, dl in test_dls.items():
        m = DinoV3_1.load_from_checkpoint(ckpt_path)
        t = L.Trainer(logger=False, enable_progress_bar=False, precision=precision)
        r = t.test(m, dl, verbose=False)[0]
        results[ds_name] = {k: float(v) for k, v in r.items()}
        print(
            f"  {ds_name:<22}  AUC={r.get('test_auc', 0):.4f}  "
            f"Acc={r.get('test_acc', 0):.4f}  F1={r.get('test_f1', 0):.4f}  "
            f"Sens={r.get('test_sensitivity', 0):.4f}  Spec={r.get('test_specificity', 0):.4f}"
        )
    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_training_curves(log_dir: str, phase_label: str, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_path = Path(log_dir) / "metrics.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df_ep = df.dropna(subset=["val_auc"]).copy()
    if df_ep.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    if "train_loss_epoch" in df_ep.columns:
        ax1.plot(df_ep["epoch"], df_ep["train_loss_epoch"], label="Train loss")
    if "val_loss" in df_ep.columns:
        ax1.plot(df_ep["epoch"], df_ep["val_loss"], label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{phase_label} — Loss"); ax1.legend()

    ax2.plot(df_ep["epoch"], df_ep["val_auc"], color="darkorange", label="Val AUC")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC-ROC")
    ax2.set_title(f"{phase_label} — Val AUC"); ax2.legend(); ax2.set_ylim(0, 1)

    plt.tight_layout()
    slug = "".join(c if c.isalnum() else "_" for c in phase_label.lower())
    out = Path(out_dir) / f"{slug}_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_comparison(
    results_a: dict, results_b: dict,
    label_a: str, label_b: str,
    out_dir: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ds_names = sorted(set(results_a) | set(results_b))
    auc_a = [results_a.get(ds, {}).get("test_auc", 0.0) for ds in ds_names]
    auc_b = [results_b.get(ds, {}).get("test_auc", 0.0) for ds in ds_names]

    x = np.arange(len(ds_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(ds_names)), 5))
    bars_a = ax.bar(x - w / 2, auc_a, w, label=label_a, color="#607D8B", alpha=0.9)
    bars_b = ax.bar(x + w / 2, auc_b, w, label=label_b, color="#2196F3", alpha=0.9)

    for bar, v in zip(bars_a, auc_a):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    for bar, v in zip(bars_b, auc_b):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Test dataset (held-out)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("DINOv3-Small — Generalisation: JRAIGS-only vs Mixed training")
    ax.set_xticks(x); ax.set_xticklabels(ds_names, rotation=20, ha="right")
    ax.legend(); ax.set_ylim(0, 1.12)
    ax.axhline(0.5, linestyle="--", color="grey", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(out_dir) / "comparison_auc.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_full_metrics(results: dict, condition_label: str, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ds_names = list(results.keys())
    metrics = ["test_auc", "test_acc", "test_f1", "test_sensitivity", "test_specificity"]
    labels  = ["AUC",      "Accuracy", "F1",      "Sensitivity",      "Specificity"]
    colors  = ["#2196F3",  "#4CAF50",  "#FF9800", "#E91E63",          "#9C27B0"]

    x = np.arange(len(ds_names))
    w = 0.15
    fig, ax = plt.subplots(figsize=(max(10, 2 * len(ds_names)), 6))
    for i, (metric, lbl, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[ds].get(metric, 0.0) for ds in ds_names]
        ax.bar(x + i * w, vals, w, label=lbl, color=color)

    ax.set_xlabel("Test dataset"); ax.set_ylabel("Score")
    ax.set_title(f"DINOv3-Small [{condition_label}] — Zero-shot test metrics")
    ax.set_xticks(x + 2 * w); ax.set_xticklabels(ds_names, rotation=30, ha="right")
    ax.legend(loc="upper right"); ax.set_ylim(0, 1.05)
    ax.axhline(0.5, linestyle="--", color="grey", alpha=0.4, linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    slug = "".join(c if c.isalnum() else "_" for c in condition_label.lower())
    out = Path(out_dir) / f"{slug}_test_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DINOv3-Small generalisation experiment")
    p.add_argument("--data_dir",    default="data/datasets")
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--max_epochs",  type=int,   default=25)
    p.add_argument("--lr1",         type=float, default=1e-3)
    p.add_argument("--lr2",         type=float, default=1e-4)
    p.add_argument("--val_ratio",   type=float, default=0.15)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--ckpt_dir",    default="checkpoints/generalization")
    p.add_argument("--figures_dir", default="figures/generalization")
    p.add_argument("--precision",   default="16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    L.seed_everything(SEED, workers=True)

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.img_size)

    test_dls = build_test_dataloaders(
        args.data_dir, eval_tf, args.batch_size, args.num_workers,
    )

    # ----------------------------------------------------------------
    # Condition A — JRAIGS-only
    # ----------------------------------------------------------------
    train_dl_a, val_dl_a = build_jraigs_only_dataloaders(
        args.data_dir, train_tf, eval_tf,
        args.batch_size, args.num_workers, args.val_ratio,
    )
    ckpt_a = train_condition(
        label="JRAIGS-only",
        train_dl=train_dl_a, val_dl=val_dl_a, args=args,
        ckpt_dir=str(Path(args.ckpt_dir) / "jraigs_only"),
        figures_dir=args.figures_dir,
    )
    print("\n" + "=" * 60)
    print("Condition A — JRAIGS-only — Zero-shot evaluation")
    print("=" * 60)
    results_a = evaluate_all(ckpt_a, test_dls, args.precision)
    plot_full_metrics(results_a, "JRAIGS-only", args.figures_dir)

    # ----------------------------------------------------------------
    # Condition B — Mixed
    # ----------------------------------------------------------------
    train_dl_b, val_dl_b = build_mixed_dataloaders(
        args.data_dir, train_tf, eval_tf,
        args.batch_size, args.num_workers, args.val_ratio,
    )
    ckpt_b = train_condition(
        label="Mixed",
        train_dl=train_dl_b, val_dl=val_dl_b, args=args,
        ckpt_dir=str(Path(args.ckpt_dir) / "mixed"),
        figures_dir=args.figures_dir,
    )
    print("\n" + "=" * 60)
    print("Condition B — Mixed — Zero-shot evaluation")
    print("=" * 60)
    results_b = evaluate_all(ckpt_b, test_dls, args.precision)
    plot_full_metrics(results_b, "Mixed", args.figures_dir)

    # ----------------------------------------------------------------
    # Comparison
    # ----------------------------------------------------------------
    plot_comparison(results_a, results_b, "JRAIGS-only", "Mixed", args.figures_dir)

    print("\n" + "=" * 60)
    print("SUMMARY — AUC  (JRAIGS-only  vs  Mixed)")
    print("=" * 60)
    for ds in sorted(set(results_a) | set(results_b)):
        auc_a = results_a.get(ds, {}).get("test_auc", float("nan"))
        auc_b = results_b.get(ds, {}).get("test_auc", float("nan"))
        delta = auc_b - auc_a
        sign = "+" if delta >= 0 else ""
        print(f"  {ds:<22}  JRAIGS-only={auc_a:.4f}  Mixed={auc_b:.4f}  Δ={sign}{delta:.4f}")

    out_json = Path(args.figures_dir) / "test_results.json"
    with open(out_json, "w") as f:
        json.dump({"jraigs_only": results_a, "mixed": results_b}, f, indent=2)
    print(f"\nResults saved to {out_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()
