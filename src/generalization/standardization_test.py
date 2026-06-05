"""Ablation test: effect of each standardization step on dataset indistinguishability.

For each ablation step (cumulative), runs:
  - Linear domain classifier (5-fold CV) → balanced accuracy
  - Centroid distance mean

Ablation steps (cumulative):
  0. raw          — no preprocessing
  1. +fov_crop    — FOV detection + square crop
  2. +illumination — fast illumination normalization
  3. +clahe       — CLAHE on green channel
  4. +mask        — circular mask
  5. +zscore      — per-image z-score within FOV

Outputs:
  figures/standardization_test/
    ablation_summary.json        ← accuracy + centroid dist per step
    ablation_accuracy.png        ← accuracy curve across steps
    ablation_centroid.png        ← centroid dist curve across steps
    {step}/umap.png              ← UMAP per step
    {step}/centroid_heatmap.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import timm
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from src.datasets import (
    ACRIMADataset, ORIGADataset, LAGDataset,
    AIROGSLightDataset, FundusTrainValDataset,
    HarvardGlaucomaDataset, JRAIGSDataset,
)
from src.datasets.RIMONE import RIMONEDataset
from src.preprocessing.fundus_standardize import FundusStandardize

BACKBONE = "vit_small_patch16_dinov3.lvd1689m"
SEED = 42
N_PER_DATASET = 300

TRAIN_DATASETS = {"JRAIGS", "ACRIMA", "ORIGA", "LAG", "Harvard"}
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
]

# Ablation steps — each dict adds one more flag to FundusStandardize
ABLATION_STEPS = [
    ("raw",          dict()),
    ("+fov_crop",    dict(do_fov_crop=True)),
    ("+illumination",dict(do_fov_crop=True, do_illumination=True)),
    ("+clahe",       dict(do_fov_crop=True, do_illumination=True, do_clahe=True)),
    ("+mask",        dict(do_fov_crop=True, do_illumination=True, do_clahe=True, do_mask=True)),
    ("+zscore",      dict(do_fov_crop=True, do_illumination=True, do_clahe=True, do_mask=True, do_zscore=True)),
]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _base_timm_tf(img_size: int = 224):
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(BACKBONE, pretrained=False, num_classes=0)
    )
    data_cfg["input_size"] = (3, img_size, img_size)
    return timm.data.create_transform(**data_cfg, is_training=False)


def build_transform(flags: dict, img_size: int = 224):
    base = _base_timm_tf(img_size)
    if not flags:
        return base
    # All unspecified flags default to False
    full_flags = dict(
        do_fov_crop=False, do_illumination=False,
        do_clahe=False, do_mask=False, do_zscore=False,
    )
    full_flags.update(flags)
    return T.Compose([FundusStandardize(target_size=512, **full_flags), base])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def subsample(dataset, n: int) -> Subset:
    g = torch.Generator().manual_seed(SEED)
    n = min(n, len(dataset))
    return Subset(dataset, torch.randperm(len(dataset), generator=g)[:n].tolist())


def load_datasets(data_dir: str, tf) -> dict[str, Subset]:
    factories = {
        "JRAIGS":        lambda: JRAIGSDataset(data_dir=data_dir, transforms=tf),
        "ACRIMA":        lambda: ACRIMADataset(data_dir=data_dir, transforms=tf),
        "ORIGA":         lambda: ORIGADataset(data_dir=data_dir, transforms=tf),
        "LAG":           lambda: LAGDataset(data_dir=data_dir, split="train", transforms=tf),
        "Harvard":       lambda: HarvardGlaucomaDataset(data_dir=data_dir, transforms=tf),
        "RIMONE(train)": lambda: RIMONEDataset(data_dir=data_dir, split="train", transforms=tf),
        "RIMONE(test)":  lambda: RIMONEDataset(data_dir=data_dir, split="test", transforms=tf),
        "AIRROGS":       lambda: AIROGSLightDataset(data_dir=data_dir, transforms=tf),
        "Fundus(train)": lambda: FundusTrainValDataset(data_dir=data_dir, split="train", transforms=tf),
        "Fundus(val)":   lambda: FundusTrainValDataset(data_dir=data_dir, split="validation", transforms=tf),
    }
    loaded = {}
    for name, fn in factories.items():
        try:
            ds = fn()
            loaded[name] = subsample(ds, N_PER_DATASET)
        except Exception as e:
            print(f"  {name:<20} SKIPPED ({e})")
    return loaded


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, dl: DataLoader, device: torch.device) -> np.ndarray:
    feats = []
    for batch in dl:
        out = model(batch["image"].to(device))
        if out.dim() == 4:
            out = out.mean(dim=(2, 3))
        feats.append(out.cpu().float().numpy())
    return np.concatenate(feats, axis=0)


def extract_all(datasets, model, device, batch_size, num_workers):
    all_X, all_labels = [], []
    for name, ds in datasets.items():
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=num_workers > 0)
        feats = extract_features(model, dl, device)
        all_X.append(feats)
        all_labels.extend([name] * len(feats))
    return np.concatenate(all_X, axis=0), all_labels, list(datasets.keys())


# ---------------------------------------------------------------------------
# Domain classifier
# ---------------------------------------------------------------------------

def domain_classifier_accuracy(X, labels) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    y = LabelEncoder().fit_transform(labels)
    n_classes = len(set(labels))
    X_s = StandardScaler().fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    bal_accs = []
    for tr, te in skf.split(X_s, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
        clf.fit(X_s[tr], y[tr])
        bal_accs.append(balanced_accuracy_score(y[te], clf.predict(X_s[te])))

    return {
        "mean_bal_acc": float(np.mean(bal_accs)),
        "std_bal_acc":  float(np.std(bal_accs)),
        "chance":       float(1.0 / n_classes),
        "ratio":        float(np.mean(bal_accs) / (1.0 / n_classes)),
    }


# ---------------------------------------------------------------------------
# Centroid distances
# ---------------------------------------------------------------------------

def centroid_dist_stats(X, labels, names) -> tuple[np.ndarray, dict]:
    centroids = np.stack([X[np.array(labels) == n].mean(0) for n in names])
    D = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)
    off = D[np.triu_indices(len(names), k=1)]
    return D, {"mean": float(off.mean()), "std": float(off.std()),
                "min": float(off.min()), "max": float(off.max())}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_umap(X, labels, names, title, out_path: Path):
    import umap as umap_lib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    emb = umap_lib.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                         metric="cosine", random_state=SEED, verbose=False).fit_transform(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(names):
        mask = np.array(labels) == name
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=PALETTE[i % len(PALETTE)], s=12, alpha=0.6,
                   marker="o" if name in TRAIN_DATASETS else "^", linewidths=0)
    handles = [mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=n)
               for i, n in enumerate(names)]
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.set_title(title, fontsize=11); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()


def plot_heatmap(D, names, title, out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(D, cmap="viridis_r")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    plt.colorbar(im, ax=ax, label="L2")
    for i in range(len(names)):
        for j in range(len(names)):
            v = D[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v > D.max() * 0.5 else "black")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close()


def plot_ablation_curves(results: list[dict], out_dir: Path):
    import matplotlib.pyplot as plt

    step_names = [r["step"] for r in results]
    accs   = [r["classifier"]["mean_bal_acc"] for r in results]
    chance = results[0]["classifier"]["chance"]
    dists  = [r["centroid"]["mean"] for r in results]
    x = range(len(step_names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x, accs, "o-", color="#e6194b", linewidth=2, markersize=8)
    ax1.axhline(chance, linestyle="--", color="grey", alpha=0.6, label=f"chance={chance:.2f}")
    ax1.set_xticks(x); ax1.set_xticklabels(step_names, rotation=30, ha="right")
    ax1.set_ylabel("Balanced accuracy (domain classifier)")
    ax1.set_title("Domain distinguishability per step\n↓ better (closer to chance)")
    ax1.set_ylim(0, 1); ax1.legend(); ax1.grid(alpha=0.3)
    for xi, yi in zip(x, accs):
        ax1.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

    ax2.plot(x, dists, "o-", color="#4363d8", linewidth=2, markersize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(step_names, rotation=30, ha="right")
    ax2.set_ylabel("Mean centroid L2 distance")
    ax2.set_title("Centroid distance per step\n↓ better (datasets closer in feature space)")
    ax2.grid(alpha=0.3)
    for xi, yi in zip(x, dists):
        ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

    plt.suptitle("FundusStandardize ablation — DINOv3-Small features", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir}/ablation_curves.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default="data/datasets")
    p.add_argument("--out_dir",       default="figures/standardization_test")
    p.add_argument("--n_per_dataset", type=int, default=N_PER_DATASET)
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--num_workers",   type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    global N_PER_DATASET
    N_PER_DATASET = args.n_per_dataset

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Backbone: {BACKBONE}\n")

    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    all_results = []

    for step_name, flags in ABLATION_STEPS:
        print(f"\n{'='*60}")
        print(f"  STEP: {step_name}   flags={flags}")
        print(f"{'='*60}")

        tf = build_transform(flags)
        datasets = load_datasets(args.data_dir, tf)
        names = list(datasets.keys())

        X, labels, names = extract_all(datasets, model, device,
                                        args.batch_size, args.num_workers)
        print(f"  {X.shape[0]} images, dim={X.shape[1]}")

        step_dir = out_root / step_name.lstrip("+").replace(" ", "_")
        step_dir.mkdir(parents=True, exist_ok=True)

        # Centroid distances
        D, dist_stats = centroid_dist_stats(X, labels, names)
        plot_heatmap(D, names, f"Centroid distances [{step_name}]",
                     step_dir / "centroid_heatmap.png")

        # UMAP
        print("  Running UMAP...")
        plot_umap(X, labels, names,
                  f"Feature space [{step_name}] — DINOv3-Small",
                  step_dir / "umap.png")

        # Domain classifier
        print("  Running domain classifier (5-fold)...")
        clf = domain_classifier_accuracy(X, labels)

        print(f"  bal_acc={clf['mean_bal_acc']:.3f}±{clf['std_bal_acc']:.3f}  "
              f"chance={clf['chance']:.3f}  ×{clf['ratio']:.1f}  "
              f"centroid_mean={dist_stats['mean']:.3f}")

        all_results.append({"step": step_name, "classifier": clf, "centroid": dist_stats})

    # Summary curves
    plot_ablation_curves(all_results, out_root)

    # Print table
    print(f"\n{'='*60}")
    print(f"{'Step':<18} {'Bal.Acc':>8} {'±':>6} {'Ratio':>7} {'Cent.dist':>10}")
    print(f"{'='*60}")
    for r in all_results:
        clf, cd = r["classifier"], r["centroid"]
        print(f"  {r['step']:<16} {clf['mean_bal_acc']:>8.3f} "
              f"{clf['std_bal_acc']:>6.3f} {clf['ratio']:>7.1f}× {cd['mean']:>10.3f}")
    print(f"  {'chance':<16} {all_results[0]['classifier']['chance']:>8.3f}")

    with open(out_root / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_root}/ablation_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
