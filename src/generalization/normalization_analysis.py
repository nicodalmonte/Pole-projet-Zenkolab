"""
Analyse l'impact de différentes stratégies de préprocessing sur le clustering UMAP.

Stratégies testées :
  raw        - resize uniquement, aucune normalisation couleur
  imagenet   - normalisation ImageNet standard (baseline déjà calculée)
  grayscale  - image en niveaux de gris (3 canaux identiques)
  green_ch   - canal vert uniquement
  clahe      - CLAHE sur chaque canal (H, V, S dans HSV)
  disc_crop  - recadrage sur le disque optique (région la plus lumineuse)
  ben_graham - préprocessing Ben Graham (soustrait la moyenne locale)

Sorties (dans --out_dir) :
  umap_<strategy>.png         — UMAP coloré par dataset
  umap_rgb.png                — UMAP coloré par couleur RGB moyenne réelle
  centroid_distances_<s>.json — matrices de distances centroid
  spearman_matrix.png/.json   — corrélation rang entre stratégies
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from src.datasets import (
    ACRIMADataset, ORIGADataset, LAGDataset, AIROGSLightDataset,
    FundusTrainValDataset, HarvardGlaucomaDataset, JRAIGSDataset, REFUGE2Dataset,
)
from src.datasets.RIMONE import RIMONEDataset
from src.generalization.dataset_clustering import (
    load_backbone, subsample, plot_umap,
    spearman_rank_correlation, SEED, N_PER_DATASET, TRAIN_DATASETS,
)

IMG_SIZE   = 224
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Preprocessing strategies
# ---------------------------------------------------------------------------

def _base_tensor_ops():
    return T.Compose([T.ToTensor()])


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")


class _WrapTransform:
    """Wraps a preprocessing fn (PIL → PIL) + final resize+totensor."""
    def __init__(self, fn, size=IMG_SIZE):
        self.fn   = fn
        self.post = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])

    def __call__(self, img):
        img = _to_pil(img)
        img = self.fn(img)
        return self.post(img)


def _imagenet_norm(size=IMG_SIZE):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def _raw(pil: Image.Image) -> Image.Image:
    return pil


def _grayscale(pil: Image.Image) -> Image.Image:
    gray = pil.convert("L")
    return Image.merge("RGB", [gray, gray, gray])


def _green_ch(pil: Image.Image) -> Image.Image:
    g = pil.split()[1]
    return Image.merge("RGB", [g, g, g])


def _clahe(pil: Image.Image) -> Image.Image:
    arr = np.array(pil)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def _disc_crop(pil: Image.Image, crop_frac: float = 0.4) -> Image.Image:
    """Crop around the brightest region (optic disc approximation)."""
    arr = np.array(pil.convert("L"), dtype=np.float32)
    k = max(arr.shape[0] // 8, 3)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(arr, (k, k), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    cx, cy = max_loc
    h, w = arr.shape
    half = int(min(h, w) * crop_frac / 2)
    x1 = max(cx - half, 0);  x2 = min(cx + half, w)
    y1 = max(cy - half, 0);  y2 = min(cy + half, h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return pil
    return pil.crop((x1, y1, x2, y2))


def _ben_graham(pil: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Ben Graham preprocessing: subtract local average, add 128."""
    arr = np.array(pil, dtype=np.float32)
    sigma = size // 8
    blurred = cv2.GaussianBlur(arr, (0, 0), sigma)
    out = cv2.addWeighted(arr, 4, blurred, -4, 128)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


STRATEGIES = {
    "raw":        _WrapTransform(_raw),
    "grayscale":  _WrapTransform(_grayscale),
    "green_ch":   _WrapTransform(_green_ch),
    "clahe":      _WrapTransform(_clahe),
    "disc_crop":  _WrapTransform(_disc_crop),
    "ben_graham": _WrapTransform(_ben_graham),
    "imagenet":   _imagenet_norm(),
}


# ---------------------------------------------------------------------------
# Dataset loading (shared subset indices via same seed)
# ---------------------------------------------------------------------------

def load_all_datasets(data_dir: str, tf, n: int) -> dict[str, Subset]:
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
        "REFUGE(train)": lambda: REFUGE2Dataset(data_dir=data_dir, split="train", transforms=tf),
    }
    loaded = {}
    for name, fn in factories.items():
        try:
            ds = fn()
            loaded[name] = subsample(ds, n)
            print(f"  {name:<20} {len(loaded[name])} samples")
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
        imgs = batch["image"].to(device)
        out = model(imgs)
        if out.dim() == 4:
            out = out.mean(dim=(2, 3))
        feats.append(out.cpu().float().numpy())
    return np.concatenate(feats, axis=0)


def collect_all_features(model, datasets: dict, device, num_workers: int) -> tuple[np.ndarray, list[str]]:
    all_feats, all_labels = [], []
    for name, ds in datasets.items():
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=num_workers > 0)
        feats = extract_features(model, dl, device)
        all_feats.append(feats)
        all_labels.extend([name] * len(feats))
        print(f"    {name:<20} {feats.shape}")
    return np.concatenate(all_feats, axis=0), all_labels


# ---------------------------------------------------------------------------
# RGB mean coloring (no GPU needed)
# ---------------------------------------------------------------------------

def collect_rgb_means(datasets_no_tf: dict) -> list[tuple[float, float, float]]:
    """Return mean RGB (0–1) for each sample across all datasets."""
    rgb_means = []
    for name, sub in datasets_no_tf.items():
        print(f"  Reading RGB: {name}")
        for idx in sub.indices:
            item = sub.dataset.dataset[idx] if hasattr(sub.dataset, 'dataset') else sub.dataset[idx]
            path = item.get("path", "")
            if path:
                try:
                    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
                    rgb_means.append(tuple(arr.mean(axis=(0, 1)).tolist()))
                    continue
                except Exception:
                    pass
            rgb_means.append((0.5, 0.5, 0.5))
    return rgb_means


def collect_rgb_from_paths(paths: list[str]) -> list[tuple]:
    rgb_means = []
    for p in paths:
        if p:
            try:
                arr = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                rgb_means.append(tuple(arr.mean(axis=(0, 1)).tolist()))
                continue
            except Exception:
                pass
        rgb_means.append((0.5, 0.5, 0.5))
    return rgb_means


def plot_umap_rgb(embedding, rgb_means, dataset_labels, dataset_names, title, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    colors = np.array(rgb_means, dtype=np.float32).clip(0, 1)
    dl = np.array(dataset_labels)

    fig, ax = plt.subplots(figsize=(12, 9))
    for name in dataset_names:
        mask = dl == name
        if not mask.any():
            continue
        marker = "o" if name in TRAIN_DATASETS else "^"
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=colors[mask], s=16, alpha=0.7,
                   marker=marker, linewidths=0)

    train_m = mlines.Line2D([], [], marker="o", color="grey", linestyle="None", markersize=6, label="train pool")
    test_m  = mlines.Line2D([], [], marker="^", color="grey", linestyle="None", markersize=6, label="test set")
    ax.legend(handles=[train_m, test_m], fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(X: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap as umap_lib
    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                             n_components=2, metric="cosine",
                             random_state=SEED, verbose=False)
    return reducer.fit_transform(X)


def centroid_dist_matrix(X: np.ndarray, labels: list[str], names: list[str]) -> np.ndarray:
    centroids = np.stack([X[np.array(labels) == n].mean(axis=0) for n in names])
    return np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)


# ---------------------------------------------------------------------------
# Spearman comparison heatmap
# ---------------------------------------------------------------------------

def plot_spearman_matrix(strategies: list[str], matrix: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(strategies))); ax.set_xticklabels(strategies, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(strategies))); ax.set_yticklabels(strategies, fontsize=9)
    plt.colorbar(im, ax=ax, label="Spearman r (centroid distances)")
    for i in range(len(strategies)):
        for j in range(len(strategies)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(v) < 0.7 else "white")
    ax.set_title("Centroid distance rank correlation between preprocessing strategies", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       default="data/datasets")
    p.add_argument("--out_dir",        default="figures/clustering/normalization_analysis")
    p.add_argument("--backbone",       default="vit_small_patch16_dinov3.lvd1689m")
    p.add_argument("--ref_embedding",  default=None,
                   help="Path to existing umap_embedding.npy to reuse for RGB plot")
    p.add_argument("--ref_labels",     default=None,
                   help="Path to existing labels.json to reuse for RGB plot")
    p.add_argument("--n_per_dataset",  type=int, default=N_PER_DATASET)
    p.add_argument("--umap_neighbors", type=int, default=30)
    p.add_argument("--umap_min_dist",  type=float, default=0.1)
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--rgb_only",       action="store_true",
                   help="Only generate the RGB-colored UMAP (no GPU needed)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # RGB UMAP (reuses existing DINOv3 embedding)
    # -----------------------------------------------------------------------
    if args.ref_embedding and args.ref_labels:
        print("\n--- RGB-colored UMAP (existing embedding) ---")
        embedding_ref = np.load(args.ref_embedding)
        with open(args.ref_labels) as f:
            labels_ref = json.load(f)

        # Collect paths by loading datasets without transform
        print("  Collecting image paths …")
        factories_notf = {
            "JRAIGS":        lambda: JRAIGSDataset(data_dir=args.data_dir),
            "ACRIMA":        lambda: ACRIMADataset(data_dir=args.data_dir),
            "ORIGA":         lambda: ORIGADataset(data_dir=args.data_dir),
            "LAG":           lambda: LAGDataset(data_dir=args.data_dir, split="train"),
            "Harvard":       lambda: HarvardGlaucomaDataset(data_dir=args.data_dir),
            "RIMONE(train)": lambda: RIMONEDataset(data_dir=args.data_dir, split="train"),
            "RIMONE(test)":  lambda: RIMONEDataset(data_dir=args.data_dir, split="test"),
            "AIRROGS":       lambda: AIROGSLightDataset(data_dir=args.data_dir),
            "Fundus(train)": lambda: FundusTrainValDataset(data_dir=args.data_dir, split="train"),
            "Fundus(val)":   lambda: FundusTrainValDataset(data_dir=args.data_dir, split="validation"),
            "REFUGE(train)": lambda: REFUGE2Dataset(data_dir=args.data_dir, split="train"),
        }
        all_paths = []
        for name, fn in factories_notf.items():
            try:
                ds = fn()
                sub = subsample(ds, args.n_per_dataset)
                for idx in sub.indices:
                    item = ds[idx]
                    all_paths.append(item.get("path", ""))
                print(f"    {name:<20} {len(sub)} samples")
            except Exception as e:
                print(f"    {name:<20} SKIPPED ({e})")

        print(f"  Computing RGB means for {len(all_paths)} images …")
        rgb_means = collect_rgb_from_paths(all_paths)
        np.save(out_dir / "rgb_means.npy", np.array(rgb_means))

        dataset_names_ref = list(dict.fromkeys(labels_ref))
        plot_umap_rgb(
            embedding_ref, rgb_means, labels_ref, dataset_names_ref,
            "Feature space colored by mean RGB — DINOv3-Small (UMAP)",
            out_dir / "umap_rgb.png",
        )

    if args.rgb_only:
        print("\nDone (rgb_only mode).")
        return

    # -----------------------------------------------------------------------
    # GPU: multi-normalization analysis
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    model = load_backbone(args.backbone, device)

    centroid_matrices: dict[str, np.ndarray] = {}
    strategy_names_done: list[str] = []

    for strat_name, tf in STRATEGIES.items():
        print(f"\n=== Strategy: {strat_name} ===")
        strat_dir = out_dir / strat_name
        strat_dir.mkdir(exist_ok=True)

        print("  Loading datasets …")
        datasets = load_all_datasets(args.data_dir, tf, args.n_per_dataset)
        if not datasets:
            print("  No datasets loaded, skipping.")
            continue
        dataset_names = list(datasets.keys())

        print("  Extracting features …")
        X, labels = collect_all_features(model, datasets, device, args.num_workers)
        np.save(strat_dir / "features.npy", X)

        print("  Running UMAP …")
        embedding = run_umap(X, args.umap_neighbors, args.umap_min_dist)
        np.save(strat_dir / "umap_embedding.npy", embedding)

        plot_umap(embedding, labels, dataset_names,
                  f"Preprocessing: {strat_name} (UMAP)",
                  strat_dir / f"umap_{strat_name}.png")

        cdm = centroid_dist_matrix(X, labels, dataset_names)
        centroid_matrices[strat_name] = (cdm, dataset_names)
        dist_dict = {dataset_names[i]: {dataset_names[j]: float(cdm[i, j])
                                         for j in range(len(dataset_names))}
                     for i in range(len(dataset_names))}
        with open(strat_dir / "centroid_distances.json", "w") as f:
            json.dump(dist_dict, f, indent=2)
        strategy_names_done.append(strat_name)

    # -----------------------------------------------------------------------
    # Spearman correlation matrix across strategies
    # -----------------------------------------------------------------------
    if len(strategy_names_done) >= 2:
        print("\n--- Computing Spearman correlation matrix ---")
        n = len(strategy_names_done)
        spearman_mat = np.ones((n, n))
        for i, si in enumerate(strategy_names_done):
            for j, sj in enumerate(strategy_names_done):
                if i == j:
                    continue
                cdm_i, names_i = centroid_matrices[si]
                cdm_j, names_j = centroid_matrices[sj]
                shared = [name for name in names_i if name in names_j]
                if len(shared) < 3:
                    spearman_mat[i, j] = float("nan")
                    continue
                idx_i = [names_i.index(n) for n in shared]
                idx_j = [names_j.index(n) for n in shared]
                A = cdm_i[np.ix_(idx_i, idx_i)]
                B = cdm_j[np.ix_(idx_j, idx_j)]
                spearman_mat[i, j] = spearman_rank_correlation(A, B)

        plot_spearman_matrix(strategy_names_done, spearman_mat, out_dir / "spearman_matrix.png")
        with open(out_dir / "spearman_matrix.json", "w") as f:
            json.dump({
                "strategies": strategy_names_done,
                "matrix": spearman_mat.tolist(),
            }, f, indent=2)
        print("\nSpearman matrix:")
        print(f"{'':12}" + "".join(f"{s:>12}" for s in strategy_names_done))
        for i, si in enumerate(strategy_names_done):
            row = "".join(f"{spearman_mat[i,j]:>12.3f}" for j in range(n))
            print(f"  {si:<10} {row}")

    print("\nDone.")


if __name__ == "__main__":
    main()
