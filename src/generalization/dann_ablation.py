"""DANN ablation study.

Experiments
-----------
  For each (condition, lambda_domain) pair:

  condition = "mixed"  : 5 domains — JRAIGS + ACRIMA + ORIGA + LAG + Harvard
  condition = "c"      : 3 domains — ACRIMA + ORIGA + LAG  (mirrors v3 condition C)

  lambda_domain ∈ {0.0, 0.5, 1.0, 2.0}

λ=0.0 is the ablation baseline: same architecture and data but no adversarial signal
(the domain head exists but its loss is multiplied by 0, making it a plain classifier).

Comparison table at the end lets us directly read whether adversarial training helps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import lightning as L
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from src.generalization.train_dann import (
    SEED, BACKBONE,
    DANNModel,
    DomainLabeledDataset,
    build_transforms,
    build_test_dataloaders,
    subsample_balanced,
    train_val_split,
    get_labels,
    run_phase,
    evaluate_all,
    plot_training_curves,
)
from src.datasets import (
    ACRIMADataset, ORIGADataset, LAGDataset,
    JRAIGSDataset, HarvardGlaucomaDataset,
)

N_PER_DOMAIN = 200   # 200 pos + 200 neg per domain


# ---------------------------------------------------------------------------
# Dataloader builders
# ---------------------------------------------------------------------------

def _make_domain_dl(
    domain_specs: list[tuple],   # (name, id, fn_train, fn_eval)
    train_tf, eval_tf,
    batch_size: int, num_workers: int, val_ratio: float,
) -> tuple[DataLoader, DataLoader, torch.Tensor, int]:
    """Generic builder: takes a list of (name, id, fn_train, fn_eval) specs."""
    all_train, all_val = [], []

    for name, did, fn_train, fn_eval in domain_specs:
        try:
            ds_tr = DomainLabeledDataset(fn_train(), did)
            ds_ev = DomainLabeledDataset(fn_eval(),  did)
            sub_tr = subsample_balanced(ds_tr, N_PER_DOMAIN, SEED + did)
            sub_ev = subsample_balanced(ds_ev, N_PER_DOMAIN, SEED + did)
            n = len(sub_tr)  # type: ignore[arg-type]
            tr_idx, val_idx = train_val_split(n, val_ratio)
            all_train.append(Subset(sub_tr, tr_idx))
            all_val.append(Subset(sub_ev, val_idx))
            print(f"  {name:<12} domain={did}  train={len(tr_idx)}  val={len(val_idx)}")
        except Exception as e:
            print(f"  {name:<12} SKIPPED ({e})")

    if not all_train:
        raise RuntimeError("No datasets loaded.")

    combined_train = ConcatDataset(all_train)
    combined_val   = ConcatDataset(all_val)

    train_labels = []
    for sub in all_train:
        train_labels.extend(get_labels(sub))
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    w = torch.tensor([1.0 / n_neg if l == 0 else 1.0 / n_pos for l in train_labels])
    sampler      = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    class_weight = torch.tensor([1.0 / n_neg, 1.0 / n_pos])
    class_weight = class_weight / class_weight.sum() * 2

    train_dl = DataLoader(combined_train, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)
    val_dl   = DataLoader(combined_val,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)

    n_domains = len({did for _, did, _, _ in domain_specs})
    print(f"\n  Total train={len(combined_train)}  val={len(combined_val)}  "
          f"pos={n_pos}  neg={n_neg}  n_domains={n_domains}")
    return train_dl, val_dl, class_weight, n_domains


def build_mixed_dls(data_dir, train_tf, eval_tf, batch_size, num_workers, val_ratio):
    """5 domains: JRAIGS(0) + ACRIMA(1) + ORIGA(2) + LAG(3) + Harvard(4)."""
    print("\n--- Condition Mixed (5 domains) ---")
    specs = [
        ("JRAIGS",   0,
         lambda: JRAIGSDataset(data_dir=data_dir, transforms=train_tf),
         lambda: JRAIGSDataset(data_dir=data_dir, transforms=eval_tf)),
        ("ACRIMA",   1,
         lambda: ACRIMADataset(data_dir=data_dir, transforms=train_tf),
         lambda: ACRIMADataset(data_dir=data_dir, transforms=eval_tf)),
        ("ORIGA",    2,
         lambda: ORIGADataset(data_dir=data_dir, transforms=train_tf),
         lambda: ORIGADataset(data_dir=data_dir, transforms=eval_tf)),
        ("LAG",      3,
         lambda: LAGDataset(data_dir=data_dir, split="train", transforms=train_tf),
         lambda: LAGDataset(data_dir=data_dir, split="train", transforms=eval_tf)),
        ("Harvard",  4,
         lambda: HarvardGlaucomaDataset(data_dir=data_dir, transforms=train_tf),
         lambda: HarvardGlaucomaDataset(data_dir=data_dir, transforms=eval_tf)),
    ]
    return _make_domain_dl(specs, train_tf, eval_tf, batch_size, num_workers, val_ratio)


def build_c_dls(data_dir, train_tf, eval_tf, batch_size, num_workers, val_ratio):
    """3 domains: ACRIMA(0) + ORIGA(1) + LAG(2) — mirrors v3 condition C."""
    print("\n--- Condition C (3 domains: ACRIMA+ORIGA+LAG) ---")
    specs = [
        ("ACRIMA",   0,
         lambda: ACRIMADataset(data_dir=data_dir, transforms=train_tf),
         lambda: ACRIMADataset(data_dir=data_dir, transforms=eval_tf)),
        ("ORIGA",    1,
         lambda: ORIGADataset(data_dir=data_dir, transforms=train_tf),
         lambda: ORIGADataset(data_dir=data_dir, transforms=eval_tf)),
        ("LAG",      2,
         lambda: LAGDataset(data_dir=data_dir, split="train", transforms=train_tf),
         lambda: LAGDataset(data_dir=data_dir, split="train", transforms=eval_tf)),
    ]
    return _make_domain_dl(specs, train_tf, eval_tf, batch_size, num_workers, val_ratio)


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    label: str,
    train_dl: DataLoader,
    val_dl: DataLoader,
    class_weight: torch.Tensor,
    n_domains: int,
    lambda_domain: float,
    test_dls: dict,
    args,
    device: torch.device,
) -> dict:
    slug      = label.replace(" ", "_").replace(".", "p").replace("=", "")
    ckpt_dir  = str(Path(args.ckpt_dir) / slug)
    log_root  = f"lightning_logs/dann_ablation/{slug}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_dl)
    total_steps     = steps_per_epoch * args.max_epochs

    print(f"\n{'='*60}")
    print(f"[{label}]  lambda={lambda_domain}  n_domains={n_domains}")
    print(f"{'='*60}")

    # Phase 1 — backbone frozen
    print(f"\nPhase 1 — backbone frozen, lr={args.lr1}")
    model_p1 = DANNModel(
        pretrained=True, lr=args.lr1, dropout=0.35, weight_decay=5e-3,
        n_domains=n_domains, lambda_domain=lambda_domain,
        total_steps=total_steps, unfreeze_backbone_epoch=args.max_epochs + 1,
    )
    model_p1.task_weight = class_weight

    # Use a CSVLogger-compatible log_name with no slashes
    best1 = run_phase(
        model_p1, train_dl, val_dl,
        args.max_epochs, f"{slug}_phase1", ckpt_dir, args.precision, patience=8,
    )
    log_dir1 = Path(f"lightning_logs/dann/{slug}_phase1/version_0")
    if log_dir1.exists():
        plot_training_curves(log_dir1, f"{label} — Phase 1", args.figures_dir)

    # Phase 2 — backbone unfrozen
    print(f"\nPhase 2 — backbone unfrozen, lr={args.lr2}")
    model_p2 = DANNModel.load_from_checkpoint(
        best1, lr=args.lr2, weight_decay=5e-3,
        total_steps=total_steps, unfreeze_backbone_epoch=0,
        lambda_domain=lambda_domain,
    )
    best2 = run_phase(
        model_p2, train_dl, val_dl,
        args.max_epochs, f"{slug}_phase2", ckpt_dir, args.precision, patience=5,
    )
    log_dir2 = Path(f"lightning_logs/dann/{slug}_phase2/version_0")
    if log_dir2.exists():
        plot_training_curves(log_dir2, f"{label} — Phase 2", args.figures_dir)

    # Evaluation
    print(f"\n{'-'*40}")
    print(f"[{label}] Zero-shot evaluation")
    print(f"{'-'*40}")
    results = evaluate_all(best2, val_dl, test_dls, device)
    return results


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     default="data/datasets")
    p.add_argument("--ckpt_dir",     default="checkpoints/dann_ablation")
    p.add_argument("--figures_dir",  default="figures/dann_ablation")
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--max_epochs",   type=int,   default=25)
    p.add_argument("--lr1",          type=float, default=1e-3)
    p.add_argument("--lr2",          type=float, default=1e-4)
    p.add_argument("--val_ratio",    type=float, default=0.15)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--precision",    default="16-mixed")
    p.add_argument(
        "--lambdas", default="0.0,0.5,1.0,2.0",
        help="Comma-separated λ values to sweep (e.g. '0.0,0.5,1.0,2.0')",
    )
    p.add_argument(
        "--conditions", default="mixed,c",
        help="Comma-separated conditions: 'mixed' and/or 'c'",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    L.seed_everything(SEED, workers=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf, eval_tf = build_transforms()

    lambdas    = [float(x) for x in args.lambdas.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    print(f"\nDevice: {device}")
    print(f"Conditions : {conditions}")
    print(f"Lambda values: {lambdas}")

    print("\n--- Building test sets ---")
    test_dls = build_test_dataloaders(args.data_dir, eval_tf,
                                       args.batch_size, args.num_workers)

    all_results: dict[str, dict] = {}

    for cond in conditions:
        if cond == "mixed":
            train_dl, val_dl, class_weight, n_domains = build_mixed_dls(
                args.data_dir, train_tf, eval_tf,
                args.batch_size, args.num_workers, args.val_ratio,
            )
            cond_label = "Mixed"
        elif cond == "c":
            train_dl, val_dl, class_weight, n_domains = build_c_dls(
                args.data_dir, train_tf, eval_tf,
                args.batch_size, args.num_workers, args.val_ratio,
            )
            cond_label = "ACRIMA+ORIGA+LAG"
        else:
            print(f"Unknown condition '{cond}', skipping.")
            continue

        for lam in lambdas:
            label = f"DANN-{cond_label}_lambda={lam:.2f}"
            results = run_experiment(
                label=label,
                train_dl=train_dl, val_dl=val_dl,
                class_weight=class_weight,
                n_domains=n_domains,
                lambda_domain=lam,
                test_dls=test_dls,
                args=args,
                device=device,
            )
            all_results[label] = results

            # Save intermediate results after each experiment
            with open(Path(args.figures_dir) / "ablation_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("DANN ABLATION — AUC SUMMARY")
    print(f"{'='*70}")

    datasets = ["RIMONE (unified)", "Fundus (unified)", "AIRROGS"]

    # Header
    header = f"  {'Experiment':<38}" + "".join(f"  {ds:<18}" for ds in datasets) + "  Mean"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for exp_label, res in all_results.items():
        aucs = [res.get(ds, {}).get("AUC", float("nan")) for ds in datasets]
        mean = sum(a for a in aucs if not (a != a)) / len([a for a in aucs if not (a != a)])
        row  = f"  {exp_label:<38}" + "".join(f"  {a:.4f}{'':12}" for a in aucs) + f"  {mean:.4f}"
        print(row)

    # Add reference baselines from v3 for comparison
    v3_baselines = {
        "v3-A Mixed (no DANN)":          [0.7509, 0.7890, 0.9285],
        "v3-B JRAIGS-only (no DANN)":    [0.7684, 0.8145, 0.9382],
        "v3-C ACRIMA+ORIGA+LAG (no DANN)": [0.6585, 0.8422, 0.8447],
    }
    print(f"\n  {'--- v3 baselines (no DANN) ---':<38}")
    for name, aucs in v3_baselines.items():
        mean = sum(aucs) / len(aucs)
        row  = f"  {name:<38}" + "".join(f"  {a:.4f}{'':12}" for a in aucs) + f"  {mean:.4f}"
        print(row)

    with open(Path(args.figures_dir) / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {args.figures_dir}/ablation_results.json")
    print("Done.")


if __name__ == "__main__":
    main()
