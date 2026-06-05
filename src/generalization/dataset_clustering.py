"""Dataset clustering via frozen backbone features + UMAP.

Extracts embeddings from a frozen backbone (timm) for all datasets, then runs
UMAP and plots the 2-D projection coloured by dataset. A per-dataset centroid
distance matrix is also saved. Supports any timm backbone via --backbone.

Typical backbones to compare:
  - vit_small_patch16_dinov3.lvd1689m   (self-supervised ViT)
  - resnet50.a1_in1k                     (supervised CNN, ImageNet-1k)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import timm
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.datasets import (
    ACRIMADataset,
    ORIGADataset,
    LAGDataset,
    AIROGSLightDataset,
    FundusTrainValDataset,
    HarvardGlaucomaDataset,
    JRAIGSDataset,
    REFUGE2Dataset,
    G1020Dataset,
    MultichannelGlaucomaBenchmarkDataset,
)
from src.datasets.RIMONE import RIMONEDataset

SEED = 42
N_PER_DATASET = 500

TRAIN_DATASETS = {"JRAIGS", "ACRIMA", "ORIGA", "LAG", "Harvard"}

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
]


# ---------------------------------------------------------------------------
# Backbone + transforms
# ---------------------------------------------------------------------------

def build_eval_transform(backbone_name: str, img_size: int = 224):
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(backbone_name, pretrained=False, num_classes=0)
    )
    data_cfg["input_size"] = (3, img_size, img_size)
    return timm.data.create_transform(**data_cfg, is_training=False)


def load_backbone(backbone_name: str, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  Backbone: {backbone_name}  —  feature dim: {model.num_features}")
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, dl: DataLoader, device: torch.device) -> tuple[np.ndarray, list[int], list[str]]:
    feats, glaucoma_labels, paths = [], [], []
    for batch in dl:
        imgs = batch["image"].to(device)
        out = model(imgs)
        if out.dim() == 4:
            out = out.mean(dim=(2, 3))
        feats.append(out.cpu().float().numpy())
        glaucoma_labels.extend(batch["label"].tolist())
        paths.extend(batch.get("path", [""] * len(batch["label"])))
    return np.concatenate(feats, axis=0), glaucoma_labels, paths


def get_image_resolutions(paths: list[str]) -> list[int]:
    """Return the shorter edge (px) of each original image."""
    from PIL import Image
    sizes = []
    for p in paths:
        if p:
            try:
                w, h = Image.open(p).size
                sizes.append(min(w, h))
            except Exception:
                sizes.append(0)
        else:
            sizes.append(0)
    return sizes


def subsample(dataset, n: int, seed: int = SEED) -> Subset:
    g = torch.Generator().manual_seed(seed)
    n = min(n, len(dataset))
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return Subset(dataset, idx)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_all_datasets(data_dir: str, tf, n_per_dataset: int) -> dict[str, Subset]:
    factories = {
        "JRAIGS":         lambda: JRAIGSDataset(data_dir=data_dir, transforms=tf),
        "ACRIMA":         lambda: ACRIMADataset(data_dir=data_dir, transforms=tf),
        "ORIGA":          lambda: ORIGADataset(data_dir=data_dir, transforms=tf),
        "LAG":            lambda: LAGDataset(data_dir=data_dir, split="train", transforms=tf),
        "Harvard":        lambda: HarvardGlaucomaDataset(data_dir=data_dir, transforms=tf),
        "RIMONE(train)":  lambda: RIMONEDataset(data_dir=data_dir, split="train", transforms=tf),
        "RIMONE(test)":   lambda: RIMONEDataset(data_dir=data_dir, split="test", transforms=tf),
        "AIRROGS":        lambda: AIROGSLightDataset(data_dir=data_dir, transforms=tf),
        "Fundus(train)":  lambda: FundusTrainValDataset(data_dir=data_dir, split="train", transforms=tf),
        "Fundus(val)":    lambda: FundusTrainValDataset(data_dir=data_dir, split="validation", transforms=tf),
        "REFUGE(train)":  lambda: REFUGE2Dataset(data_dir=data_dir, split="train", transforms=tf),
        "G1020":          lambda: G1020Dataset(data_dir=data_dir, transforms=tf),
        # Multichannel Glaucoma Benchmark — one entry per source to avoid blending heterogeneous domains
        "BEH":            lambda: MultichannelGlaucomaBenchmarkDataset(data_dir=data_dir, sources=["BEH"], transforms=tf),
        "FIVES":          lambda: MultichannelGlaucomaBenchmarkDataset(data_dir=data_dir, sources=["FIVES"], transforms=tf),
        "PAPILA":         lambda: MultichannelGlaucomaBenchmarkDataset(data_dir=data_dir, sources=["PAPILA"], transforms=tf),
        "sjchoi86-HRF":   lambda: MultichannelGlaucomaBenchmarkDataset(data_dir=data_dir, sources=["sjchoi86-HRF"], transforms=tf),
        "OIA-ODIR":       lambda: MultichannelGlaucomaBenchmarkDataset(data_dir=data_dir, sources=["OIA-ODIR-TRAIN", "OIA-ODIR-TEST-ONLINE", "OIA-ODIR-TEST-OFFLINE"], transforms=tf),
        "DRISHTI-GS1":    lambda: MultichannelGlaucomaBenchmarkDataset(data_dir=data_dir, sources=["DRISHTI-GS1-train", "DRISHTI-GS1-test"], transforms=tf),
        "CRFO-v4":        lambda: MultichannelGlaucomaBenchmarkDataset(data_dir=data_dir, sources=["CRFO-v4"], transforms=tf),
    }
    loaded: dict[str, Subset] = {}
    for name, fn in factories.items():
        try:
            ds = fn()
            sub = subsample(ds, n_per_dataset)
            print(f"  {name:<20} {len(ds):>6} total  →  {len(sub)} sampled")
            loaded[name] = sub
        except Exception as e:
            print(f"  {name:<20} SKIPPED  ({e})")
    return loaded


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_umap(embedding, labels, dataset_names, title, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12, 9))
    for i, name in enumerate(dataset_names):
        mask = np.array(labels) == name
        color = PALETTE[i % len(PALETTE)]
        marker = "o" if name in TRAIN_DATASETS else "^"
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=12, alpha=0.6, marker=marker, linewidths=0)

    handles = [mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=n)
               for i, n in enumerate(dataset_names)]
    train_m = plt.Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=6, label="train pool")
    test_m  = plt.Line2D([0], [0], marker="^", color="grey", linestyle="None", markersize=6, label="test set")
    ax.legend(handles=handles + [train_m, test_m], loc="best", fontsize=8, ncol=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_umap_glaucoma(embedding, glaucoma_labels, dataset_labels, dataset_names, title, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    GLAUCOMA_COLORS = {0: "#3cb44b", 1: "#e6194b"}
    GLAUCOMA_NAMES  = {0: "Non-glaucoma", 1: "Glaucoma"}

    fig, ax = plt.subplots(figsize=(12, 9))
    gl = np.array(glaucoma_labels)
    dl = np.array(dataset_labels)
    for glc in [0, 1]:
        for name in dataset_names:
            mask = (gl == glc) & (dl == name)
            if not mask.any():
                continue
            marker = "o" if name in TRAIN_DATASETS else "^"
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=GLAUCOMA_COLORS[glc], s=12, alpha=0.55,
                       marker=marker, linewidths=0)

    color_handles = [mpatches.Patch(color=GLAUCOMA_COLORS[k], label=GLAUCOMA_NAMES[k]) for k in [0, 1]]
    train_m = plt.Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=6, label="train pool")
    test_m  = plt.Line2D([0], [0], marker="^", color="grey", linestyle="None", markersize=6, label="test set")
    ax.legend(handles=color_handles + [train_m, test_m], loc="best", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_umap_resolution(embedding, resolutions, dataset_labels, dataset_names, title, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    res = np.array(resolutions, dtype=float)
    # replace zeros with nan so they don't affect the colormap
    valid = res > 0
    vmin, vmax = res[valid].min(), res[valid].max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(13, 9))
    dl = np.array(dataset_labels)
    for name in dataset_names:
        mask = dl == name
        if not mask.any():
            continue
        marker = "o" if name in TRAIN_DATASETS else "^"
        colors = cmap(norm(res[mask]))
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=colors, s=12, alpha=0.6, marker=marker, linewidths=0)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Resolution (shorter edge, px)")

    train_m = plt.Line2D([0], [0], marker="o", color="grey", linestyle="None", markersize=6, label="train pool")
    test_m  = plt.Line2D([0], [0], marker="^", color="grey", linestyle="None", markersize=6, label="test set")
    ax.legend(handles=[train_m, test_m], loc="best", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_umap_pair(embedding, dataset_labels, glaucoma_labels,
                   name_a: str, name_b: str, title, out_path: Path) -> None:
    """Color = dataset, marker = glaucoma status (o=non-glaucoma, *=glaucoma)."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    COLOR_A, COLOR_B = "#e6194b", "#4363d8"
    GLAUCOMA_MARKER, NONGLAUCOMA_MARKER = "*", "o"
    GLAUCOMA_SIZE, NONGLAUCOMA_SIZE = 60, 18

    dl = np.array(dataset_labels)
    gl = np.array(glaucoma_labels)

    fig, ax = plt.subplots(figsize=(12, 9))
    for name, color in [(name_a, COLOR_A), (name_b, COLOR_B)]:
        mask_ds = dl == name
        if not mask_ds.any():
            print(f"  WARNING: '{name}' not found in labels, skipping.")
            continue
        for glc, marker, size in [(0, NONGLAUCOMA_MARKER, NONGLAUCOMA_SIZE),
                                   (1, GLAUCOMA_MARKER,    GLAUCOMA_SIZE)]:
            mask = mask_ds & (gl == glc)
            if not mask.any():
                continue
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, s=size, alpha=0.65, marker=marker,
                       linewidths=0)

    # Legend
    handles = [
        mlines.Line2D([], [], color=COLOR_A, marker="s", linestyle="None", markersize=8, label=name_a),
        mlines.Line2D([], [], color=COLOR_B, marker="s", linestyle="None", markersize=8, label=name_b),
        mlines.Line2D([], [], color="grey",  marker=NONGLAUCOMA_MARKER, linestyle="None", markersize=6, label="Non-glaucoma"),
        mlines.Line2D([], [], color="grey",  marker=GLAUCOMA_MARKER,    linestyle="None", markersize=9, label="Glaucoma"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="best")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_centroid_heatmap(centroid_dist, dataset_names, title, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(centroid_dist, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(dataset_names))); ax.set_xticklabels(dataset_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(dataset_names))); ax.set_yticklabels(dataset_names, fontsize=9)
    plt.colorbar(im, ax=ax, label="L2 distance between centroids")
    for i in range(len(dataset_names)):
        for j in range(len(dataset_names)):
            v = centroid_dist[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > centroid_dist.max() * 0.5 else "black")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Rank correlation between two distance matrices
# ---------------------------------------------------------------------------

def spearman_rank_correlation(A: np.ndarray, B: np.ndarray) -> float:
    from scipy.stats import spearmanr
    # upper triangle only, exclude diagonal
    idx = np.triu_indices_from(A, k=1)
    r, _ = spearmanr(A[idx], B[idx])
    return float(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def backbone_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", name.lower())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       default="data/datasets")
    p.add_argument("--out_dir",        default="figures/clustering")
    p.add_argument("--backbone",       default="vit_small_patch16_dinov3.lvd1689m")
    p.add_argument("--n_per_dataset",  type=int,   default=N_PER_DATASET)
    p.add_argument("--umap_neighbors", type=int,   default=30)
    p.add_argument("--umap_min_dist",  type=float, default=0.1)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--compare_with",   default=None,
                   help="Path to a centroid_distances.json from another backbone run to compute rank correlation")
    return p.parse_args()


def main():
    args = parse_args()
    slug = backbone_slug(args.backbone)

    out_dir = Path(args.out_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")

    tf = build_eval_transform(args.backbone)
    model = load_backbone(args.backbone, device)

    print("\n--- Loading datasets ---")
    datasets = load_all_datasets(args.data_dir, tf, args.n_per_dataset)
    dataset_names = list(datasets.keys())

    print("\n--- Extracting features ---")
    all_feats, all_labels, all_glaucoma, all_paths = [], [], [], []
    for name, ds in datasets.items():
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=device.type == "cuda",
                        persistent_workers=args.num_workers > 0)
        feats, glc_labels, paths = extract_features(model, dl, device)
        print(f"  {name:<20} {feats.shape}")
        all_feats.append(feats)
        all_labels.extend([name] * len(feats))
        all_glaucoma.extend(glc_labels)
        all_paths.extend(paths)

    X = np.concatenate(all_feats, axis=0)
    labels = all_labels
    print(f"\nTotal: {X.shape[0]} images, dim={X.shape[1]}")

    print("\n--- Reading original image resolutions ---")
    all_resolutions = get_image_resolutions(all_paths)
    print(f"  Resolution range: {min(r for r in all_resolutions if r > 0)}–{max(all_resolutions)} px")

    np.save(out_dir / "features.npy", X)
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f)
    with open(out_dir / "glaucoma_labels.json", "w") as f:
        json.dump(all_glaucoma, f)

    # Centroid distances
    print("\n--- Centroid distances ---")
    centroids = np.stack([X[np.array(labels) == n].mean(axis=0) for n in dataset_names])
    centroid_dist = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)
    dist_dict = {dataset_names[i]: {dataset_names[j]: float(centroid_dist[i, j])
                                     for j in range(len(dataset_names))}
                 for i in range(len(dataset_names))}
    with open(out_dir / "centroid_distances.json", "w") as f:
        json.dump(dist_dict, f, indent=2)
    plot_centroid_heatmap(centroid_dist, dataset_names,
                          f"Centroid distances — {args.backbone}",
                          out_dir / "centroid_heatmap.png")

    # Rank correlation with another backbone (if provided)
    if args.compare_with:
        other_path = Path(args.compare_with)
        if other_path.exists():
            with open(other_path) as f:
                other_dict = json.load(f)
            # align on same dataset names
            shared = [n for n in dataset_names if n in other_dict]
            idx = [dataset_names.index(n) for n in shared]
            A = centroid_dist[np.ix_(idx, idx)]
            B = np.array([[other_dict[shared[i]][shared[j]] for j in range(len(shared))]
                          for i in range(len(shared))])
            r = spearman_rank_correlation(A, B)
            print(f"\n  Spearman rank correlation with {other_path.parent.name}: r={r:.4f}")
            with open(out_dir / "rank_correlation.json", "w") as f:
                json.dump({"compared_with": str(other_path), "spearman_r": r, "n_datasets": len(shared)}, f, indent=2)
        else:
            print(f"  WARNING: --compare_with path not found: {other_path}")

    # UMAP
    print("\n--- Running UMAP ---")
    import umap as umap_lib
    reducer = umap_lib.UMAP(n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
                             n_components=2, metric="cosine", random_state=SEED, verbose=True)
    embedding = reducer.fit_transform(X)
    np.save(out_dir / "umap_embedding.npy", embedding)
    plot_umap(embedding, labels, dataset_names,
              f"DINOv3-Small feature space — {args.backbone} (UMAP)",
              out_dir / "umap_all_datasets.png")
    plot_umap_glaucoma(embedding, all_glaucoma, labels, dataset_names,
                       f"Glaucoma vs Non-glaucoma — {args.backbone} (UMAP)",
                       out_dir / "umap_glaucoma.png")
    plot_umap_resolution(embedding, all_resolutions, labels, dataset_names,
                         f"Image resolution — {args.backbone} (UMAP)",
                         out_dir / "umap_resolution.png")
    pair_dir = out_dir / "pair_plots"
    pair_dir.mkdir(exist_ok=True)
    for other in dataset_names:
        if other == "REFUGE(train)":
            continue
        slug = other.replace("(", "").replace(")", "").replace(" ", "_").lower()
        plot_umap_pair(embedding, labels, all_glaucoma,
                       "REFUGE(train)", other,
                       f"REFUGE(train) vs {other} — feature space (UMAP)",
                       pair_dir / f"umap_refuge_vs_{slug}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
