"""Test embedding-level synthesis methods.

Loads precomputed DINOv3 features and measures whether artificial transforms
can shift source-domain embeddings closer to a target cluster (RIMONE).

Methods tested (all in feature space, no image re-loading):
  1. Baseline        — raw ACRIMA / Harvard features
  2. Feature interp  — linear interpolation toward RIMONE centroid (alpha sweep)
  3. AdaIN           — replace mean+std of source features with RIMONE stats
  4. MixStyle        — convex mix of source and RIMONE mean+std per sample
  5. ZCA             — covariance transfer in PCA-128 space

For each method we report:
  - centroid_dist(transformed_source → RIMONE)           [domain gap]
  - class-conditional centroid dist glaucoma / healthy    [label alignment]
  - class separability = dist(c_pos, c_neg) in transformed space  [discriminability]
  - intra_spread                                          [diversity preserved]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import Subset

SEED = 42
N_PER_DATASET = 500

BASE_DIR   = Path(__file__).parent.parent.parent
FEAT_PATH   = BASE_DIR / "figures/clustering/features.npy"
LABELS_PATH = BASE_DIR / "figures/clustering/labels.json"

SOURCE_SETS = ["ACRIMA", "Harvard"]
TARGET_SET  = "RIMONE(train)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def centroid(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)

def intra_spread(X: np.ndarray) -> float:
    return float(np.linalg.norm(X - centroid(X), axis=1).mean())

def class_sep(X: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between pos and neg centroids — measures discriminability."""
    c_pos = centroid(X[y == 1])
    c_neg = centroid(X[y == 0])
    return l2(c_pos, c_neg)


def subsample_indices(n_total: int, n: int, seed: int = SEED) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    n = min(n, n_total)
    return torch.randperm(n_total, generator=g)[:n].tolist()


# ---------------------------------------------------------------------------
# Load class labels by replaying the same subsampling used during clustering
# ---------------------------------------------------------------------------

def load_class_labels(data_dir: str = "data/datasets") -> dict[str, np.ndarray]:
    """Returns {dataset_name: array of 0/1 class labels} for source+target."""
    from src.datasets import ACRIMADataset, HarvardGlaucomaDataset
    from src.datasets.RIMONE import RIMONEDataset

    factories = {
        "ACRIMA":        lambda: ACRIMADataset(data_dir=data_dir),
        "Harvard":       lambda: HarvardGlaucomaDataset(data_dir=data_dir),
        "RIMONE(train)": lambda: RIMONEDataset(data_dir=data_dir, split="train"),
    }
    out = {}
    for name, fn in factories.items():
        ds = fn()
        idx = subsample_indices(len(ds), N_PER_DATASET)
        labels = []
        for i in idx:
            item = ds[i]
            lbl = item["label"]
            labels.append(int(lbl.item() if hasattr(lbl, "item") else lbl))
        out[name] = np.array(labels, dtype=np.int32)
        pos = sum(labels); neg = len(labels) - pos
        print(f"  {name:<20} n={len(labels)}  pos(glaucoma)={pos}  neg(healthy)={neg}")
    return out


# ---------------------------------------------------------------------------
# Synthesis methods
# ---------------------------------------------------------------------------

def method_interp(X: np.ndarray, c_tgt: np.ndarray, alpha: float) -> np.ndarray:
    return X + alpha * (c_tgt - centroid(X))

def method_adain(X: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
    mu_s = X.mean(axis=0);    std_s = X.std(axis=0) + 1e-8
    mu_t = X_tgt.mean(axis=0); std_t = X_tgt.std(axis=0) + 1e-8
    return (X - mu_s) / std_s * std_t + mu_t

def method_mixstyle(X: np.ndarray, X_tgt: np.ndarray, lam: float) -> np.ndarray:
    mu_s = X.mean(axis=0);    std_s = X.std(axis=0) + 1e-8
    mu_t = X_tgt.mean(axis=0); std_t = X_tgt.std(axis=0) + 1e-8
    mu_m  = (1 - lam) * mu_s  + lam * mu_t
    std_m = (1 - lam) * std_s + lam * std_t
    return (X - mu_s) / std_s * std_m + mu_m

def method_zca(X_src: np.ndarray, X_tgt: np.ndarray, k: int = 128, eps: float = 1e-6) -> np.ndarray:
    k = min(k, X_src.shape[0] - 1, X_src.shape[1])
    mu_s = X_src.mean(axis=0)
    Xs = X_src - mu_s
    U_s, S_s, Vt_s = np.linalg.svd(Xs, full_matrices=False)
    Xs_k = U_s[:, :k] * S_s[:k]
    mu_t = X_tgt.mean(axis=0)
    Xt_k = (X_tgt - mu_t) @ Vt_s[:k].T
    std_s = Xs_k.std(axis=0) + eps
    std_t = Xt_k.std(axis=0) + eps
    return Xs_k * (std_t / std_s) @ Vt_s[:k] + mu_t


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(name: str,
             X_transformed: np.ndarray, y_src: np.ndarray,
             X_tgt: np.ndarray,         y_tgt: np.ndarray,
             baseline_dist: float) -> dict:
    c_tr  = centroid(X_transformed)
    c_tgt = centroid(X_tgt)

    dist_global = l2(c_tr, c_tgt)
    delta = baseline_dist - dist_global

    # class-conditional centroids
    c_tr_pos = centroid(X_transformed[y_src == 1])
    c_tr_neg = centroid(X_transformed[y_src == 0])
    c_tgt_pos = centroid(X_tgt[y_tgt == 1])
    c_tgt_neg = centroid(X_tgt[y_tgt == 0])

    dist_pos = l2(c_tr_pos, c_tgt_pos)   # glaucoma centroid alignment
    dist_neg = l2(c_tr_neg, c_tgt_neg)   # healthy centroid alignment

    sep_src = class_sep(X_transformed, y_src)   # discriminability in transformed space
    sep_tgt = class_sep(X_tgt, y_tgt)           # discriminability in RIMONE space

    spread = intra_spread(X_transformed)

    return dict(
        name=name,
        dist_global=dist_global,
        delta=delta,
        dist_pos=dist_pos,
        dist_neg=dist_neg,
        sep_src=sep_src,
        sep_tgt=sep_tgt,
        spread=spread,
    )


def print_results(r: dict, show_ref_sep: float | None = None) -> None:
    sign = "▼" if r["delta"] > 0 else "▲"
    print(f"\n  [{r['name']}]")
    print(f"    dist(centroïde → RIMONE)        : {r['dist_global']:.4f}  ({sign}{abs(r['delta']):.4f} vs baseline)")
    print(f"    dist(c_glaucome → c_glaucome_RIM): {r['dist_pos']:.4f}")
    print(f"    dist(c_sain    → c_sain_RIM)    : {r['dist_neg']:.4f}")
    print(f"    séparabilité classes (transformed): {r['sep_src']:.4f}  {'(RIMONE ref: {:.4f})'.format(r['sep_tgt']) if show_ref_sep else ''}")
    print(f"    spread (diversité)              : {r['spread']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Chargement des features précomputées ===")
    X_all = np.load(FEAT_PATH).astype(np.float32)
    with open(LABELS_PATH) as f:
        ds_labels = np.array(json.load(f))

    datasets_feat = {n: X_all[ds_labels == n] for n in np.unique(ds_labels)}

    print("\n=== Chargement des labels de classe (glaucome/sain) ===")
    class_labels = load_class_labels()

    X_src = np.concatenate([datasets_feat[s] for s in SOURCE_SETS if s in datasets_feat])
    y_src = np.concatenate([class_labels[s]  for s in SOURCE_SETS if s in class_labels])
    X_tgt = datasets_feat[TARGET_SET]
    y_tgt = class_labels[TARGET_SET]
    c_tgt = centroid(X_tgt)

    baseline_dist = l2(centroid(X_src), c_tgt)
    sep_tgt       = class_sep(X_tgt, y_tgt)

    print(f"\n{'='*65}")
    print(f"Source : {SOURCE_SETS}  ({len(X_src)} samples, dim={X_src.shape[1]})")
    print(f"Cible  : {TARGET_SET}  ({len(X_tgt)} samples)")
    print(f"BASELINE dist global → RIMONE : {baseline_dist:.4f}")
    print(f"Séparabilité RIMONE (ref)      : {sep_tgt:.4f}")
    print(f"{'='*65}")

    configs = [
        ("Baseline",          X_src),
        ("Interp α=0.25",     method_interp(X_src, c_tgt, 0.25)),
        ("Interp α=0.5",      method_interp(X_src, c_tgt, 0.5)),
        ("Interp α=0.75",     method_interp(X_src, c_tgt, 0.75)),
        ("AdaIN",             method_adain(X_src, X_tgt)),
        ("MixStyle λ=0.25",   method_mixstyle(X_src, X_tgt, 0.25)),
        ("MixStyle λ=0.5",    method_mixstyle(X_src, X_tgt, 0.5)),
        ("MixStyle λ=0.75",   method_mixstyle(X_src, X_tgt, 0.75)),
        ("ZCA-128",           method_zca(X_src, X_tgt, k=128)),
        ("RIMONE (oracle)",   X_tgt),
    ]

    results = []
    for name, X_t in configs:
        y = y_tgt if name == "RIMONE (oracle)" else y_src
        r = evaluate(name, X_t, y, X_tgt, y_tgt, baseline_dist)
        results.append(r)
        print_results(r, show_ref_sep=True)

    # -----------------------------------------------------------------------
    # Tableau récapitulatif
    # -----------------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"RÉCAPITULATIF")
    print(f"{'Méthode':<22} {'dist→RIM':>9} {'Δ':>8} {'dist_glau':>10} {'dist_sain':>10} {'sep_class':>10} {'spread':>8}")
    print(f"{'-'*90}")
    for r in results:
        sign = "▼" if r["delta"] > 0 else " "
        print(f"  {r['name']:<20} {r['dist_global']:>9.4f} {sign}{abs(r['delta']):>7.4f} "
              f"{r['dist_pos']:>10.4f} {r['dist_neg']:>10.4f} "
              f"{r['sep_src']:>10.4f} {r['spread']:>8.4f}")

    print(f"\n  Référence RIMONE : sep_class={sep_tgt:.4f}")
    print(f"\n  → sep_class mesure dist(c_glaucome, c_sain) dans l'espace transformé.")
    print(f"    Plus c'est proche de {sep_tgt:.4f}, mieux le signal discriminant est aligné sur RIMONE.")


if __name__ == "__main__":
    main()
