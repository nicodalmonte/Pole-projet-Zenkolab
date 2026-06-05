"""Cluster-aware generalization experiment.

Based on UMAP clustering, datasets fall into 3 visual clusters:
  Cluster A (JRAIGS-style)  : JRAIGS, AIRROGS        — dist to RIMONE ~3.1–3.3
  Cluster B (Fundus-style)  : ORIGA, LAG, Fundus      — dist to RIMONE ~2.7–3.4
  Cluster C (Clinical disc) : ACRIMA, Harvard, RIMONE — dist to RIMONE  1.4–1.7

RIMONE is left out entirely as the held-out test cluster.
Secondary test sets: AIRROGS (cluster A), Fundus (cluster B).

Training conditions (all 2000 images, balanced 50/50 per condition):
  single_A   : JRAIGS only             (cluster A — far from RIMONE)
  single_B   : ORIGA + LAG             (cluster B — far from RIMONE)
  single_C   : ACRIMA + Harvard        (cluster C — close to RIMONE, same cluster)
  multi_AB   : JRAIGS + ORIGA/LAG      (2 clusters — diverse but no close data)
  multi_ABC  : JRAIGS + ORIGA/LAG + ACRIMA/Harvard  (all 3 clusters)

Key questions:
  1. Does same-cluster proximity help?     (single_C vs single_A/single_B on RIMONE)
  2. Does 2-cluster diversity help?        (multi_AB vs single_A/single_B on RIMONE)
  3. Does adding close data on top help?   (multi_ABC vs multi_AB on RIMONE)
  4. Do these gains come at a cost on      (all conditions on AIRROGS + Fundus)
     other test sets?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import lightning as L
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from src.generalization.train_generalization_v3 import (
    SEED, BACKBONE,
    DinoV3_1_V2,
    build_transforms,
    build_test_dataloaders,
    get_labels_fast, class_counts,
    _subsample, _stratified_subsample, _split_and_build_loaders,
    _find_optimal_threshold, _metrics_at_threshold,
    plot_training_curves, plot_full_metrics,
    BinaryMappedDataset,
)
from src.datasets import (
    JRAIGSDataset, ORIGADataset, LAGDataset,
    ACRIMADataset, HarvardGlaucomaDataset,
    AIROGSLightDataset, FundusTrainValDataset,
)
from src.datasets.RIMONE import RIMONEDataset

from lightning.pytorch.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateMonitor, RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

# ── Budget ─────────────────────────────────────────────────────────────────
N_TOTAL      = 2000   # total per condition
N_PER_CLUSTER = N_TOTAL // 3   # ~667 per cluster in multi conditions


# ---------------------------------------------------------------------------
# Test-set builder (RIMONE fully excluded from training)
# ---------------------------------------------------------------------------

def build_test_dls(data_dir: str, eval_tf, batch_size: int, num_workers: int) -> dict:
    """RIMONE (held-out cluster), AIRROGS (cluster A), Fundus (cluster B)."""
    def make(ds: Dataset) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True,
                         persistent_workers=num_workers > 0)
    dls: dict[str, DataLoader] = {}
    factories = {
        "RIMONE (unified)": lambda: ConcatDataset([
            RIMONEDataset(data_dir=data_dir, split="train", transforms=eval_tf),
            RIMONEDataset(data_dir=data_dir, split="test",  transforms=eval_tf),
        ]),
        "AIRROGS": lambda: AIROGSLightDataset(data_dir=data_dir, transforms=eval_tf),
        "Fundus (unified)": lambda: ConcatDataset([
            FundusTrainValDataset(data_dir=data_dir, split="train",      transforms=eval_tf),
            FundusTrainValDataset(data_dir=data_dir, split="validation", transforms=eval_tf),
        ]),
    }
    for name, fn in factories.items():
        try:
            ds = fn()
            labels = get_labels_fast(ds)
            n_neg, n_pos = class_counts(labels)
            dls[name] = make(ds)
            print(f"  {name:<22} {len(ds)} samples  (neg={n_neg} pos={n_pos})")
        except Exception as e:
            print(f"  {name:<22} SKIPPED ({e})")
    return dls


# ---------------------------------------------------------------------------
# Condition builders
# ---------------------------------------------------------------------------

def _pool(datasets_train: list, datasets_eval: list,
          n_pos: int, n_neg: int,
          batch_size: int, num_workers: int, val_ratio: float,
          label: str) -> tuple[DataLoader, DataLoader]:
    combined_train = ConcatDataset(datasets_train)
    combined_eval  = ConcatDataset(datasets_eval)
    sub_train = _subsample(combined_train, n_pos, n_neg, SEED)
    sub_eval  = _subsample(combined_eval,  n_pos, n_neg, SEED)
    return _split_and_build_loaders(sub_train, sub_eval, val_ratio,
                                    batch_size, num_workers, label)


def build_single_A(data_dir, train_tf, eval_tf, batch_size, num_workers, val_ratio):
    """Cluster A only: JRAIGS — 1000 pos + 1000 neg."""
    print("\n--- single_A: JRAIGS (cluster A, far from RIMONE) ---")
    return _pool(
        [JRAIGSDataset(data_dir=data_dir, transforms=train_tf)],
        [JRAIGSDataset(data_dir=data_dir, transforms=eval_tf)],
        1000, 1000, batch_size, num_workers, val_ratio, "single_A",
    )


def build_single_B(data_dir, train_tf, eval_tf, batch_size, num_workers, val_ratio):
    """Cluster B only: ORIGA + LAG — 1000 pos + 1000 neg."""
    print("\n--- single_B: ORIGA+LAG (cluster B, far from RIMONE) ---")
    return _pool(
        [ORIGADataset(data_dir=data_dir, transforms=train_tf),
         LAGDataset(data_dir=data_dir, split="train", transforms=train_tf)],
        [ORIGADataset(data_dir=data_dir, transforms=eval_tf),
         LAGDataset(data_dir=data_dir, split="train", transforms=eval_tf)],
        1000, 1000, batch_size, num_workers, val_ratio, "single_B",
    )


def build_single_C(data_dir, train_tf, eval_tf, batch_size, num_workers, val_ratio):
    """Cluster C (excl. RIMONE): ACRIMA + Harvard — 1000 pos + 1000 neg."""
    print("\n--- single_C: ACRIMA+Harvard (cluster C, close to RIMONE) ---")
    return _pool(
        [ACRIMADataset(data_dir=data_dir, transforms=train_tf),
         BinaryMappedDataset(HarvardGlaucomaDataset(data_dir=data_dir, transforms=train_tf))],
        [ACRIMADataset(data_dir=data_dir, transforms=eval_tf),
         BinaryMappedDataset(HarvardGlaucomaDataset(data_dir=data_dir, transforms=eval_tf))],
        1000, 1000, batch_size, num_workers, val_ratio, "single_C",
    )


def build_multi_AB(data_dir, train_tf, eval_tf, batch_size, num_workers, val_ratio):
    """Clusters A+B (no cluster C): JRAIGS + ORIGA + LAG — 1000 pos + 1000 neg.
    Tests whether 2-cluster diversity helps without having close data.
    """
    print("\n--- multi_AB: JRAIGS+ORIGA+LAG (clusters A+B, no cluster C) ---")
    # Each cluster contributes equally: ~500/500 from A, ~500/500 from B
    n_each = N_TOTAL // 2  # 1000 from A, 1000 from B

    def _sub(ds_tr, ds_ev, n):
        sub_tr = _stratified_subsample(ds_tr, n // 2, SEED)
        sub_ev = _stratified_subsample(ds_ev, n // 2, SEED)
        return sub_tr, sub_ev

    jraigs_tr = JRAIGSDataset(data_dir=data_dir, transforms=train_tf)
    jraigs_ev = JRAIGSDataset(data_dir=data_dir, transforms=eval_tf)
    sub_a_tr, sub_a_ev = _sub(jraigs_tr, jraigs_ev, n_each)

    b_tr = ConcatDataset([
        ORIGADataset(data_dir=data_dir, transforms=train_tf),
        LAGDataset(data_dir=data_dir, split="train", transforms=train_tf),
    ])
    b_ev = ConcatDataset([
        ORIGADataset(data_dir=data_dir, transforms=eval_tf),
        LAGDataset(data_dir=data_dir, split="train", transforms=eval_tf),
    ])
    sub_b_tr, sub_b_ev = _sub(b_tr, b_ev, n_each)

    combined_tr = ConcatDataset([sub_a_tr, sub_b_tr])
    combined_ev = ConcatDataset([sub_a_ev, sub_b_ev])

    labels = get_labels_fast(combined_tr)
    n_neg, n_pos = class_counts(labels)
    print(f"  multi_AB combined: {len(combined_tr)} total  (neg={n_neg} pos={n_pos})")
    return _split_and_build_loaders(combined_tr, combined_ev, val_ratio,
                                    batch_size, num_workers, "multi_AB")


def build_multi_ABC(data_dir, train_tf, eval_tf, batch_size, num_workers, val_ratio):
    """All 3 clusters: JRAIGS + ORIGA+LAG + ACRIMA+Harvard — ~2000 total.
    Each cluster contributes equally (~667 images).
    """
    print("\n--- multi_ABC: JRAIGS+ORIGA+LAG+ACRIMA+Harvard (clusters A+B+C) ---")
    n_each = N_TOTAL // 3  # ~667 per cluster

    def _sub(ds_tr, ds_ev, n):
        sub_tr = _stratified_subsample(ds_tr, n // 2, SEED)
        sub_ev = _stratified_subsample(ds_ev, n // 2, SEED)
        return sub_tr, sub_ev

    # Cluster A
    sub_a_tr, sub_a_ev = _sub(
        JRAIGSDataset(data_dir=data_dir, transforms=train_tf),
        JRAIGSDataset(data_dir=data_dir, transforms=eval_tf),
        n_each,
    )
    # Cluster B
    sub_b_tr, sub_b_ev = _sub(
        ConcatDataset([ORIGADataset(data_dir=data_dir, transforms=train_tf),
                       LAGDataset(data_dir=data_dir, split="train", transforms=train_tf)]),
        ConcatDataset([ORIGADataset(data_dir=data_dir, transforms=eval_tf),
                       LAGDataset(data_dir=data_dir, split="train", transforms=eval_tf)]),
        n_each,
    )
    # Cluster C (no RIMONE)
    sub_c_tr, sub_c_ev = _sub(
        ConcatDataset([ACRIMADataset(data_dir=data_dir, transforms=train_tf),
                       BinaryMappedDataset(HarvardGlaucomaDataset(data_dir=data_dir, transforms=train_tf))]),
        ConcatDataset([ACRIMADataset(data_dir=data_dir, transforms=eval_tf),
                       BinaryMappedDataset(HarvardGlaucomaDataset(data_dir=data_dir, transforms=eval_tf))]),
        n_each,
    )

    combined_tr = ConcatDataset([sub_a_tr, sub_b_tr, sub_c_tr])
    combined_ev = ConcatDataset([sub_a_ev, sub_b_ev, sub_c_ev])

    labels = get_labels_fast(combined_tr)
    n_neg, n_pos = class_counts(labels)
    print(f"  multi_ABC combined: {len(combined_tr)} total  (neg={n_neg} pos={n_pos})")
    return _split_and_build_loaders(combined_tr, combined_ev, val_ratio,
                                    batch_size, num_workers, "multi_ABC")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def run_phase(model, train_dl, val_dl, max_epochs, log_name, ckpt_dir, precision, patience):
    logger = CSVLogger("lightning_logs/cluster_generalization", name=log_name)
    v = logger.version
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{log_name}_v{v}-{{epoch:02d}}-{{val_auc:.4f}}",
            monitor="val_auc", mode="max", save_top_k=1,
        ),
        EarlyStopping(monitor="val_auc", mode="max", patience=patience, min_delta=1e-3),
        LearningRateMonitor("epoch"),
        RichProgressBar(leave=True),
    ]
    trainer = L.Trainer(
        max_epochs=max_epochs, callbacks=callbacks,
        precision=precision, logger=logger,
        log_every_n_steps=10, deterministic=False,
        gradient_clip_val=1.0, gradient_clip_algorithm="norm",
    )
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, train_dl, val_dl)
    return trainer.checkpoint_callback.best_model_path, logger.log_dir


def train_condition(label, train_dl, val_dl, args, ckpt_dir, figures_dir):
    slug = label.lower().replace(" ", "_").replace("-", "_")

    print(f"\n{'='*60}\n[{label}] Phase 1 — backbone frozen\n{'='*60}")
    m1 = DinoV3_1_V2(
        backbone_name=BACKBONE, pretrained=True,
        lr=args.lr1, img_size=args.img_size,
        dropout=0.35, weight_decay=5e-3,
        unfreeze_backbone_epoch=args.max_epochs + 1,
    )
    best1, log1 = run_phase(m1, train_dl, val_dl, args.max_epochs,
                             f"{slug}_phase1", ckpt_dir, args.precision, patience=8)
    plot_training_curves(log1, f"{label} — Phase 1", figures_dir)

    print(f"\n{'='*60}\n[{label}] Phase 2 — backbone unfrozen\n{'='*60}")
    m2 = DinoV3_1_V2.load_from_checkpoint(
        best1, lr=args.lr2, weight_decay=5e-3, unfreeze_backbone_epoch=0,
    )
    best2, log2 = run_phase(m2, train_dl, val_dl, args.max_epochs,
                             f"{slug}_phase2", ckpt_dir, args.precision, patience=5)
    plot_training_curves(log2, f"{label} — Phase 2", figures_dir)
    return best2


@torch.no_grad()
def _collect(model, dl):
    device = next(model.parameters()).device
    model.eval()
    ps, ls = [], []
    for batch in dl:
        logits = model(batch["image"].to(device))
        ps.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        ls.append(batch["label"].cpu())
    return torch.cat(ps), torch.cat(ls)


def evaluate(ckpt, val_dl, test_dls, figures_dir, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = DinoV3_1_V2.load_from_checkpoint(ckpt).to(device)
    val_p, val_l = _collect(m, val_dl)
    thr = _find_optimal_threshold(val_p, val_l)
    print(f"  Threshold (Youden on val): {thr:.3f}")
    results = {}
    for name, dl in test_dls.items():
        p, l = _collect(m, dl)
        r = _metrics_at_threshold(p, l, thr)
        results[name] = r
        print(f"  {name:<22}  AUC={r['test_auc']:.4f}  "
              f"Sens={r['test_sensitivity']:.4f}  Spec={r['test_specificity']:.4f}")
    plot_full_metrics(results, label, figures_dir)
    return results


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="data/datasets")
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--max_epochs",  type=int,   default=25)
    p.add_argument("--lr1",         type=float, default=1e-3)
    p.add_argument("--lr2",         type=float, default=1e-4)
    p.add_argument("--val_ratio",   type=float, default=0.15)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--ckpt_dir",    default="checkpoints/cluster_generalization")
    p.add_argument("--figures_dir", default="figures/cluster_generalization")
    p.add_argument("--precision",   default="16-mixed")
    p.add_argument(
        "--conditions", default="all",
        help="Comma-separated subset, e.g. 'single_A,single_C,multi_ABC', or 'all'",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONDITION_MAP = {
    "single_A":  ("JRAIGS only (cluster A, far)",            build_single_A),
    "single_B":  ("ORIGA+LAG (cluster B, far)",              build_single_B),
    "single_C":  ("ACRIMA+Harvard (cluster C, close)",       build_single_C),
    "multi_AB":  ("JRAIGS+ORIGA/LAG (A+B, no close data)",  build_multi_AB),
    "multi_ABC": ("JRAIGS+ORIGA/LAG+ACRIMA/Harvard (A+B+C)", build_multi_ABC),
}


def main():
    args = parse_args()
    L.seed_everything(SEED, workers=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.img_size)

    print("\n--- Test sets (RIMONE cluster held out entirely) ---")
    test_dls = build_test_dls(args.data_dir, eval_tf, args.batch_size, args.num_workers)

    if args.conditions == "all":
        to_run = list(CONDITION_MAP.keys())
    else:
        to_run = [c.strip() for c in args.conditions.split(",")]

    print(f"\nConditions to run: {to_run}")
    all_results: dict[str, dict] = {}

    for cond_key in to_run:
        if cond_key not in CONDITION_MAP:
            print(f"Unknown condition '{cond_key}', skipping.")
            continue
        label, builder = CONDITION_MAP[cond_key]

        train_dl, val_dl = builder(
            args.data_dir, train_tf, eval_tf,
            args.batch_size, args.num_workers, args.val_ratio,
        )
        ckpt = train_condition(
            label, train_dl, val_dl, args,
            ckpt_dir=str(Path(args.ckpt_dir) / cond_key),
            figures_dir=args.figures_dir,
        )
        print(f"\n{'='*60}\n[{label}] Zero-shot evaluation\n{'='*60}")
        all_results[label] = evaluate(ckpt, val_dl, test_dls, args.figures_dir, label)

        with open(Path(args.figures_dir) / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CLUSTER GENERALIZATION — AUC SUMMARY")
    print(f"  Test cluster (RIMONE): held out — never seen during training")
    print(f"{'='*70}")

    ds_order = ["RIMONE (unified)", "AIRROGS", "Fundus (unified)"]
    header = f"  {'Condition':<42}" + "".join(f"  {d:<20}" for d in ds_order)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, res in all_results.items():
        row = f"  {label:<42}" + "".join(
            f"  {res.get(d, {}).get('test_auc', float('nan')):.4f}{'':14}"
            for d in ds_order
        )
        print(row)

    # Reference baselines from v3
    print(f"\n  {'--- v3 baselines (for reference) ---':<42}")
    baselines = {
        "v3-A Mixed (all datasets)":    [0.7509, 0.9285, 0.7890],
        "v3-B JRAIGS-only":             [0.7684, 0.9382, 0.8145],
        "v3-C ACRIMA+ORIGA+LAG":        [0.6585, 0.8447, 0.8422],
    }
    for name, aucs in baselines.items():
        row = f"  {name:<42}" + "".join(f"  {a:.4f}{'':14}" for a in aucs)
        print(row)

    with open(Path(args.figures_dir) / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.figures_dir}/results.json")
    print("Done.")


if __name__ == "__main__":
    main()
