"""Project new datasets onto the existing UMAP space.

Loads features extracted by dataset_clustering.py (after job 5039), splits them
into old vs new datasets, fits UMAP on the old ones only (same hyperparams +
seed → same 2-D layout), then transforms the new datasets into that space and
plots everything superimposed.

Usage:
    uv run python -m src.generalization.project_new_datasets \
        --features_dir figures/clustering/vit_small_patch16_dinov3_lvd1689m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OLD_DATASETS = {
    "JRAIGS", "ACRIMA", "ORIGA", "LAG", "Harvard",
    "RIMONE(train)", "RIMONE(test)", "AIRROGS",
    "Fundus(train)", "Fundus(val)", "REFUGE(train)",
}

NEW_DATASETS = {
    "G1020", "BEH", "FIVES", "PAPILA", "sjchoi86-HRF",
    "OIA-ODIR", "DRISHTI-GS1", "CRFO-v4",
}

TRAIN_DATASETS = {"JRAIGS", "ACRIMA", "ORIGA", "LAG", "Harvard"}

OLD_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#800000",  # REFUGE(train)
]
NEW_PALETTE = [
    "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#c77dff",
    "#f4a261", "#2ec4b6", "#e76f51",
]

UMAP_NEIGHBORS = 30
UMAP_MIN_DIST  = 0.1
SEED           = 42


def load_data(features_dir: Path):
    X      = np.load(features_dir / "features.npy")
    labels = json.loads((features_dir / "labels.json").read_text())
    return X, labels


def split_old_new(X, labels):
    labels_arr = np.array(labels)
    old_mask = np.array([l in OLD_DATASETS for l in labels])
    new_mask = np.array([l in NEW_DATASETS for l in labels])

    X_old      = X[old_mask]
    labs_old   = labels_arr[old_mask].tolist()
    X_new      = X[new_mask]
    labs_new   = labels_arr[new_mask].tolist()

    unknown = set(labels) - OLD_DATASETS - NEW_DATASETS
    if unknown:
        print(f"  WARNING: unrecognised datasets (ignored): {unknown}")

    return X_old, labs_old, X_new, labs_new


def fit_umap_on_old(X_old):
    import umap as umap_lib
    print("  Fitting UMAP on old datasets …")
    reducer = umap_lib.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="cosine",
        random_state=SEED,
        verbose=False,
    )
    emb_old = reducer.fit_transform(X_old)
    return reducer, emb_old


def project_new(reducer, X_new):
    print("  Projecting new datasets into existing UMAP space …")
    return reducer.transform(X_new)


def plot_superimposed(emb_old, labs_old, emb_new, labs_new, out_path: Path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    old_names = sorted(set(labs_old), key=lambda n: list(OLD_DATASETS).index(n) if n in OLD_DATASETS else 99)
    new_names = sorted(set(labs_new))

    fig, ax = plt.subplots(figsize=(14, 10))

    # --- old datasets (circles, muted alpha) ---
    for i, name in enumerate(old_names):
        mask = np.array(labs_old) == name
        marker = "o" if name in TRAIN_DATASETS else "^"
        ax.scatter(
            emb_old[mask, 0], emb_old[mask, 1],
            c=OLD_PALETTE[i % len(OLD_PALETTE)],
            s=10, alpha=0.35, marker=marker, linewidths=0,
        )

    # --- new datasets (diamonds, full alpha, larger) ---
    for i, name in enumerate(new_names):
        mask = np.array(labs_new) == name
        ax.scatter(
            emb_new[mask, 0], emb_new[mask, 1],
            c=NEW_PALETTE[i % len(NEW_PALETTE)],
            s=22, alpha=0.85, marker="D", linewidths=0,
        )

    # Legend — old
    old_handles = [
        mpatches.Patch(color=OLD_PALETTE[i % len(OLD_PALETTE)], alpha=0.5, label=n)
        for i, n in enumerate(old_names)
    ]
    # Legend — new
    new_handles = [
        mpatches.Patch(color=NEW_PALETTE[i % len(NEW_PALETTE)], label=n)
        for i, n in enumerate(new_names)
    ]
    # Marker legend
    marker_handles = [
        plt.Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=6, label="train pool (old)"),
        plt.Line2D([0], [0], marker="^", color="grey", linestyle="None", markersize=6, label="test set (old)"),
        plt.Line2D([0], [0], marker="D", color="grey", linestyle="None", markersize=6, label="new datasets"),
    ]

    leg1 = ax.legend(handles=old_handles, loc="upper left",  fontsize=7, ncol=2, title="Old datasets",  title_fontsize=8)
    leg2 = ax.legend(handles=new_handles, loc="upper right", fontsize=7, ncol=2, title="New datasets",  title_fontsize=8)
    leg3 = ax.legend(handles=marker_handles, loc="lower right", fontsize=7, title="Marker type", title_fontsize=8)
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    ax.set_title("New datasets projected onto existing UMAP space (DINOv3-Small)", fontsize=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def print_centroid_distances(emb_old, labs_old, emb_new, labs_new):
    """Print L2 distances from each new dataset centroid to each old dataset centroid."""
    old_names = sorted(set(labs_old))
    new_names = sorted(set(labs_new))

    old_centroids = {n: emb_old[np.array(labs_old) == n].mean(axis=0) for n in old_names}
    new_centroids = {n: emb_new[np.array(labs_new) == n].mean(axis=0) for n in new_names}

    print("\n  Nearest old dataset for each new dataset (2-D UMAP distance):")
    for new_name, new_c in sorted(new_centroids.items()):
        dists = {old_name: float(np.linalg.norm(new_c - old_c))
                 for old_name, old_c in old_centroids.items()}
        nearest = sorted(dists.items(), key=lambda x: x[1])[:3]
        nearest_str = ", ".join(f"{n}={d:.2f}" for n, d in nearest)
        print(f"    {new_name:<18} → {nearest_str}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default="figures/clustering/vit_small_patch16_dinov3_lvd1689m")
    p.add_argument("--out_dir",      default=None, help="Output directory (defaults to features_dir)")
    return p.parse_args()


def main():
    args = parse_args()
    feat_dir = Path(args.features_dir)
    out_dir  = Path(args.out_dir) if args.out_dir else feat_dir

    print(f"Loading features from {feat_dir} …")
    X, labels = load_data(feat_dir)
    print(f"  Total: {X.shape[0]} samples, dim={X.shape[1]}")
    print(f"  Datasets found: {sorted(set(labels))}")

    X_old, labs_old, X_new, labs_new = split_old_new(X, labels)
    print(f"  Old: {X_old.shape[0]} samples ({len(set(labs_old))} datasets)")
    print(f"  New: {X_new.shape[0]} samples ({len(set(labs_new))} datasets)")

    reducer, emb_old = fit_umap_on_old(X_old)
    emb_new = project_new(reducer, X_new)

    print_centroid_distances(emb_old, labs_old, emb_new, labs_new)

    out_path = out_dir / "umap_new_projected.png"
    plot_superimposed(emb_old, labs_old, emb_new, labs_new, out_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
