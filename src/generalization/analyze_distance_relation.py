"""Rigorous analysis on top of the JRAIGS distance/generalization run.

Two questions:

  (A) WHAT EXPLAINS THE DINO CLUSTERS?
      For each dataset we extract simple, interpretable image-level features
      (resolution, black-border ratio, center/edge brightness ratio, colour
      stats…), compute the pairwise dataset-level distance in this feature
      space, and Mantel-test it against the DINO centroid distance matrix.
      We also report each feature's individual Mantel-Spearman with DINO to
      identify which physical property drives the clustering.

  (B) ROBUST RELATION DISTANCE -> AUC
      The JRAIGS run produced 16 (distance, AUC) pairs.  Two datasets are
      visibly broken (G1020 AUC=0.50 random; CRFO-v4 AUC=0.43 sub-random,
      n=79).  We:
        - flag outliers via the Huber regression residuals (no manual cut-off);
        - fit OLS, Huber, log-distance and quadratic models;
        - fit a multivariate OLS adding two confounders
          (class_imbalance, log(n_test)) to quantify what distance adds
          on top of trivial test-set properties;
        - cross-validate (LOOCV) and report RMSE on held-out points.

Outputs in `figures/distance_analysis/`:
  cluster_explanation.json     numbers (all per-feature Mantel + global Mantel)
  feature_vs_dino.png          scatter of physical-feature distance vs DINO
  per_feature_bars.png         per-feature Mantel-Spearman bar chart
  robust_distance_auc.json     numbers (Huber/OLS/multivariate + LOOCV)
  robust_distance_auc.png      scatter with robust line + outlier callouts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from PIL import Image

# Reuse the same dataset roster as dataset_clustering.py — same names so
# distances/labels line up with the existing JSON.
from src.datasets import (
    ACRIMADataset, ORIGADataset, LAGDataset, AIROGSLightDataset,
    HarvardGlaucomaDataset, JRAIGSDataset, REFUGE2Dataset, G1020Dataset,
    MultichannelGlaucomaBenchmarkDataset,
)
from src.datasets.RIMONE import RIMONEDataset


SEED = 42
N_PER_DATASET = 100          # images sampled per dataset for stats extraction
STAT_THUMBNAIL = 256         # resize to this on the shorter edge before computing stats
NUM_WORKERS    = 16          # threads for I/O-bound image reads
DINO_DIST_PATH = (
    "figures/clustering/vit_small_patch16_dinov3_lvd1689m/centroid_distances.json"
)
RESULTS_PATH = "figures/distance_generalization/results_jraigs.json"
OUT_DIR = Path("figures/distance_analysis")


# ---------------------------------------------------------------------------
# Dataset roster — must match centroid_distances.json keys
# ---------------------------------------------------------------------------

def dataset_factories(data_dir: str):
    """Return {name: callable that builds a dataset with no transforms}.

    Names match the keys in centroid_distances.json.
    """
    return {
        "JRAIGS":        lambda: JRAIGSDataset(data_dir=data_dir),
        "ACRIMA":        lambda: ACRIMADataset(data_dir=data_dir),
        "ORIGA":         lambda: ORIGADataset(data_dir=data_dir),
        "LAG":           lambda: LAGDataset(data_dir=data_dir, split="train"),
        "Harvard":       lambda: HarvardGlaucomaDataset(data_dir=data_dir),
        "RIMONE(train)": lambda: RIMONEDataset(data_dir=data_dir, split="train"),
        "RIMONE(test)":  lambda: RIMONEDataset(data_dir=data_dir, split="test"),
        "AIRROGS":       lambda: AIROGSLightDataset(data_dir=data_dir),
        "REFUGE(train)": lambda: REFUGE2Dataset(data_dir=data_dir, split="train"),
        "G1020":         lambda: G1020Dataset(data_dir=data_dir),
        "BEH":           lambda: MultichannelGlaucomaBenchmarkDataset(
                              data_dir=data_dir, sources=["BEH"]),
        "FIVES":         lambda: MultichannelGlaucomaBenchmarkDataset(
                              data_dir=data_dir, sources=["FIVES"]),
        "PAPILA":        lambda: MultichannelGlaucomaBenchmarkDataset(
                              data_dir=data_dir, sources=["PAPILA"]),
        "sjchoi86-HRF":  lambda: MultichannelGlaucomaBenchmarkDataset(
                              data_dir=data_dir, sources=["sjchoi86-HRF"]),
        "OIA-ODIR":      lambda: MultichannelGlaucomaBenchmarkDataset(
                              data_dir=data_dir,
                              sources=["OIA-ODIR-TRAIN", "OIA-ODIR-TEST-ONLINE", "OIA-ODIR-TEST-OFFLINE"]),
        "DRISHTI-GS1":   lambda: MultichannelGlaucomaBenchmarkDataset(
                              data_dir=data_dir,
                              sources=["DRISHTI-GS1-train", "DRISHTI-GS1-test"]),
        "CRFO-v4":       lambda: MultichannelGlaucomaBenchmarkDataset(
                              data_dir=data_dir, sources=["CRFO-v4"]),
    }


# ---------------------------------------------------------------------------
# Path extraction (works across all our dataset classes)
# ---------------------------------------------------------------------------

def get_paths(dataset, n: int, seed: int = SEED) -> list[str]:
    """Sample up to n image paths from a dataset, regardless of internal layout."""
    paths: list[str] = []
    if hasattr(dataset, "image_paths") and dataset.image_paths:
        paths = [str(p) for p in dataset.image_paths]
    elif hasattr(dataset, "samples") and dataset.samples:
        paths = [str(s[0]) for s in dataset.samples]
    else:
        raise AttributeError(f"{type(dataset).__name__} exposes neither image_paths nor samples")

    rng = np.random.default_rng(seed)
    if len(paths) > n:
        idx = rng.choice(len(paths), size=n, replace=False)
        paths = [paths[i] for i in idx]
    return paths


# ---------------------------------------------------------------------------
# Per-image physical feature extraction
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "resolution",          # min(w, h) — px
    "aspect_ratio",        # w / h
    "mean_luminance",      # 0-255
    "std_luminance",
    "black_border_ratio",  # frac of pixels with luminance < 15
    "center_edge_ratio",   # center luminance / edge luminance (vignetting proxy)
    "red_green_ratio",     # mean R / mean G  (fundus colour cast)
    "mean_saturation",     # HSV S, 0-1
]


def image_features(path: str) -> dict | None:
    """Compute simple physical features for one image. Returns None on failure.

    Resolution/aspect_ratio are computed from the ORIGINAL size; everything else
    from a small thumbnail to keep numpy ops cheap. Black-border / brightness
    ratios are scale-invariant, so this is faithful enough for clustering.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    orig_w, orig_h = img.size
    # downscale aggressively for the numerical features
    w_t = max(1, orig_w * STAT_THUMBNAIL // max(orig_w, orig_h))
    h_t = max(1, orig_h * STAT_THUMBNAIL // max(orig_w, orig_h))
    img_small = img.resize((w_t, h_t), Image.BILINEAR)
    arr = np.asarray(img_small, dtype=np.float32)  # H x W x 3
    gray = arr.mean(axis=2)
    H, W = gray.shape

    # Black border ratio
    black_ratio = float((gray < 15).mean())

    # Center vs edge luminance (vignetting / disc-crop proxy)
    rh, rw = H // 4, W // 4
    cy, cx = H // 2, W // 2
    center_box = gray[cy - rh:cy + rh, cx - rw:cx + rw]
    border = np.concatenate([
        gray[:max(1, H // 8), :].ravel(),
        gray[-max(1, H // 8):, :].ravel(),
        gray[:, :max(1, W // 8)].ravel(),
        gray[:, -max(1, W // 8):].ravel(),
    ])
    edge_mean = float(border.mean()) if border.size else 1.0
    center_edge_ratio = float(center_box.mean()) / max(edge_mean, 1.0)

    # Colour
    mean_rgb = arr.mean(axis=(0, 1))
    red_green_ratio = float(mean_rgb[0] / max(mean_rgb[1], 1.0))

    # HSV saturation — compute directly from RGB to skip a PIL conv
    rmax = arr.max(axis=2); rmin = arr.min(axis=2)
    sat = np.where(rmax > 0, (rmax - rmin) / np.maximum(rmax, 1.0), 0.0)
    mean_sat = float(sat.mean())

    return {
        "resolution":         float(min(orig_w, orig_h)),
        "aspect_ratio":       float(orig_w / max(orig_h, 1)),
        "mean_luminance":     float(gray.mean()),
        "std_luminance":      float(gray.std()),
        "black_border_ratio": black_ratio,
        "center_edge_ratio":  center_edge_ratio,
        "red_green_ratio":    red_green_ratio,
        "mean_saturation":    mean_sat,
    }


def dataset_feature_means(name: str, paths: list[str]) -> dict[str, float]:
    """Average each per-image feature across the dataset sample (threaded I/O)."""
    from concurrent.futures import ThreadPoolExecutor
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for feat in ex.map(image_features, paths, chunksize=4):
            if feat is not None:
                rows.append(feat)
    if not rows:
        raise RuntimeError(f"No features extracted for {name} — all images failed")
    out = {k: float(np.mean([r[k] for r in rows])) for k in FEATURE_NAMES}
    out["_n_sampled"] = len(rows)
    return out


# ---------------------------------------------------------------------------
# Part A — what explains the DINO clusters?
# ---------------------------------------------------------------------------

def upper_triangle(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(mat, k=1)
    return mat[iu]


def mantel_spearman(D1: np.ndarray, D2: np.ndarray) -> tuple[float, float]:
    """Spearman of the upper triangles (Mantel-like test)."""
    from scipy.stats import spearmanr
    rho, p = spearmanr(upper_triangle(D1), upper_triangle(D2))
    return float(rho), float(p)


def part_a_explain_clusters(args) -> dict:
    print("\n" + "=" * 70)
    print(" PART A — what explains the DINO clusters?")
    print("=" * 70)

    factories = dataset_factories(args.data_dir)
    names = list(factories.keys())

    # ── 1. Extract per-dataset means of physical features ────────────────
    print(f"\n[1] Extracting physical features ({N_PER_DATASET} images/dataset)")
    stats: dict[str, dict[str, float]] = {}
    for name, build in factories.items():
        try:
            ds = build()
            paths = get_paths(ds, N_PER_DATASET)
            stats[name] = dataset_feature_means(name, paths)
            print(f"  {name:<16} n={stats[name]['_n_sampled']:>3}  "
                  f"res={stats[name]['resolution']:.0f}  "
                  f"black={stats[name]['black_border_ratio']:.2f}  "
                  f"c/e={stats[name]['center_edge_ratio']:.2f}  "
                  f"R/G={stats[name]['red_green_ratio']:.2f}")
        except Exception as e:
            print(f"  {name:<16} SKIPPED ({e})")

    # ── 2. Build a feature matrix and z-score (column-wise) ──────────────
    valid_names = [n for n in names if n in stats]
    feat_mat = np.array([[stats[n][f] for f in FEATURE_NAMES] for n in valid_names])  # D x F
    # z-score per feature so each contributes on the same scale
    mu = feat_mat.mean(axis=0); sd = feat_mat.std(axis=0); sd[sd == 0] = 1.0
    Z = (feat_mat - mu) / sd

    # ── 3. Pairwise dataset distance in physical-feature space ───────────
    phys_dist = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)

    # ── 4. Load DINO centroid distance, aligned on shared datasets ───────
    with open(args.dino_dist) as f:
        dino_dict = json.load(f)
    shared = [n for n in valid_names if n in dino_dict]
    dino_idx_self = [valid_names.index(n) for n in shared]
    dino_mat = np.array([[dino_dict[a].get(b, np.nan) for b in shared] for a in shared])
    phys_mat = phys_dist[np.ix_(dino_idx_self, dino_idx_self)]
    print(f"\n[2] {len(shared)} datasets shared with DINO centroids")

    # ── 5. Mantel-Spearman (global) ──────────────────────────────────────
    rho_all, p_all = mantel_spearman(phys_mat, dino_mat)
    print(f"\n[3] Mantel-Spearman (all 8 features → DINO): rho={rho_all:.3f}  p={p_all:.2g}")

    # ── 6. Per-feature Mantel-Spearman (single feature dist matrix) ──────
    per_feat = {}
    for j, fname in enumerate(FEATURE_NAMES):
        zcol = Z[:, j:j+1]
        d1 = np.abs(zcol - zcol.T)
        d1 = d1[np.ix_(dino_idx_self, dino_idx_self)]
        rho_j, p_j = mantel_spearman(d1, dino_mat)
        per_feat[fname] = {"rho": rho_j, "p": p_j}
        print(f"      {fname:<22} rho={rho_j:+.3f}  p={p_j:.2g}")

    # ── 7. Plot ──────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(upper_triangle(phys_mat), upper_triangle(dino_mat),
               s=24, alpha=0.55, edgecolor="black", linewidth=0.3)
    # OLS line for visual
    from sklearn.linear_model import LinearRegression
    x = upper_triangle(phys_mat).reshape(-1, 1)
    y = upper_triangle(dino_mat)
    lr = LinearRegression().fit(x, y)
    xline = np.linspace(x.min(), x.max(), 50)
    ax.plot(xline, lr.predict(xline.reshape(-1, 1)), "--", color="red",
            label=f"linear fit (R²={lr.score(x, y):.2f})")
    ax.set_xlabel("Physical-feature distance (z-scored, 8 features)")
    ax.set_ylabel("DINOv3 centroid distance")
    ax.set_title(f"Physical features vs DINO clustering — Mantel ρ={rho_all:.3f} (p={p_all:.2g}, n={len(shared)})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_vs_dino.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Plot → {OUT_DIR / 'feature_vs_dino.png'}")

    # Per-feature bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = sorted(per_feat.items(), key=lambda kv: -kv[1]["rho"])
    fnames = [n for n, _ in ordered]
    rhos = [v["rho"] for _, v in ordered]
    colors = ["#2ca02c" if r > 0.5 else "#1f77b4" if r > 0 else "#d62728" for r in rhos]
    ax.barh(fnames, rhos, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="grey", linewidth=0.7)
    ax.set_xlabel("Per-feature Mantel Spearman ρ vs DINO distance")
    ax.set_title("Which physical feature best reproduces DINO clustering?")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_feature_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Plot → {OUT_DIR / 'per_feature_bars.png'}")

    # Sorted summary
    print("\n[4] Per-feature Mantel-Spearman, ranked:")
    for fn, v in ordered:
        print(f"      {fn:<22}  rho={v['rho']:+.3f}   p={v['p']:.2g}")

    out = {
        "n_datasets": len(shared),
        "dataset_order": shared,
        "global_mantel": {"rho": rho_all, "p": p_all},
        "per_feature_mantel": per_feat,
        "feature_means": {n: {k: stats[n][k] for k in FEATURE_NAMES} for n in shared},
    }
    with open(OUT_DIR / "cluster_explanation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n      Results JSON → {OUT_DIR / 'cluster_explanation.json'}")
    return out


# ---------------------------------------------------------------------------
# Part B — robust distance↔AUC relation
# ---------------------------------------------------------------------------

def part_b_robust_regression(args) -> dict:
    print("\n" + "=" * 70)
    print(" PART B — robust relation distance ↔ AUC")
    print("=" * 70)

    with open(args.results) as f:
        run = json.load(f)
    recs = run["results"]

    # Build arrays (exclude in-distribution self at distance=0)
    oof = [r for r in recs if r["distance"] > 0.01]
    d   = np.array([r["distance"]   for r in oof])
    a   = np.array([r["test_auc"]   for r in oof])
    n   = np.array([r["n"]          for r in oof], dtype=float)
    pos = np.array([r["n_pos"]      for r in oof], dtype=float)
    imb = np.abs(0.5 - pos / n)              # class-imbalance gap from 50/50
    log_n = np.log(n)
    names = [r["dataset"] for r in oof]

    print(f"\n[1] {len(oof)} OOD points (excluding source's in-distribution test)")

    # ── OLS ───────────────────────────────────────────────────────────────
    from sklearn.linear_model import LinearRegression, HuberRegressor
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from scipy.stats import spearmanr, pearsonr

    ols = LinearRegression().fit(d.reshape(-1, 1), a)
    r2_ols = ols.score(d.reshape(-1, 1), a)

    # ── Huber (robust) ────────────────────────────────────────────────────
    hub = HuberRegressor(epsilon=1.35, max_iter=500).fit(d.reshape(-1, 1), a)
    a_pred_hub = hub.predict(d.reshape(-1, 1))
    r2_hub = 1.0 - np.sum((a - a_pred_hub) ** 2) / np.sum((a - a.mean()) ** 2)
    outliers_mask = hub.outliers_      # True for points Huber down-weighted
    flagged = [n for n, o in zip(names, outliers_mask) if o]

    print(f"\n[2] OLS    : AUC = {ols.coef_[0]:+.4f}·d + {ols.intercept_:.4f}   R²={r2_ols:.3f}")
    print(f"    Huber  : AUC = {hub.coef_[0]:+.4f}·d + {hub.intercept_:.4f}   R²={r2_hub:.3f}")
    print(f"    Huber flagged as outliers: {flagged}")

    # ── Multivariate OLS: AUC ~ d + imb + log_n ───────────────────────────
    X3 = np.column_stack([d, imb, log_n])
    ols3 = LinearRegression().fit(X3, a)
    r2_3 = ols3.score(X3, a)
    coefs = ols3.coef_
    print(f"\n[3] Multivariate OLS (d + |imb-0.5| + log(n_test)):")
    print(f"    AUC = {coefs[0]:+.4f}·d + {coefs[1]:+.4f}·imb + {coefs[2]:+.4f}·log(n) + {ols3.intercept_:.4f}")
    print(f"    R² = {r2_3:.3f}   (vs OLS d-only R² = {r2_ols:.3f}  → +{r2_3 - r2_ols:.3f})")

    # Partial correlation: residualize AUC on (imb, log_n), then correlate residual with d
    from numpy.linalg import lstsq
    Xc = np.column_stack([np.ones_like(d), imb, log_n])
    beta_a, *_ = lstsq(Xc, a, rcond=None)
    beta_d, *_ = lstsq(Xc, d, rcond=None)
    res_a = a - Xc @ beta_a
    res_d = d - Xc @ beta_d
    rho_part_s, p_part_s = spearmanr(res_d, res_a)
    rho_part_p, p_part_p = pearsonr(res_d, res_a)
    print(f"    Partial corr (Spearman, dist | imb, log_n): rho={rho_part_s:+.3f}  p={p_part_s:.3g}")
    print(f"    Partial corr (Pearson, dist | imb, log_n) : r  ={rho_part_p:+.3f}  p={p_part_p:.3g}")

    # ── LOOCV ────────────────────────────────────────────────────────────
    loo = LeaveOneOut()
    def loocv_rmse(estimator, X, y):
        pred = cross_val_predict(estimator, X, y, cv=loo)
        return float(np.sqrt(np.mean((y - pred) ** 2))), pred

    rmse_ols, _ = loocv_rmse(LinearRegression(), d.reshape(-1, 1), a)
    rmse_hub, _ = loocv_rmse(HuberRegressor(epsilon=1.35, max_iter=500), d.reshape(-1, 1), a)
    rmse_3,   _ = loocv_rmse(LinearRegression(), X3, a)
    print(f"\n[4] LOOCV RMSE:")
    print(f"      OLS (d only)         : {rmse_ols:.4f}")
    print(f"      Huber (d only)       : {rmse_hub:.4f}")
    print(f"      OLS (d + imb + log_n): {rmse_3:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Bootstrap CIs from the run JSON
    cilo = np.array([r["auc_ci_lo"] for r in oof])
    cihi = np.array([r["auc_ci_hi"] for r in oof])

    fig, ax = plt.subplots(figsize=(11, 7))
    # Normal points
    mask_ok = ~outliers_mask
    ax.errorbar(d[mask_ok], a[mask_ok], yerr=[a[mask_ok] - cilo[mask_ok], cihi[mask_ok] - a[mask_ok]],
                fmt="o", capsize=3, ecolor="grey", markersize=9,
                markerfacecolor="#1f77b4", markeredgecolor="black",
                label="zero-shot targets")
    # Huber outliers
    if outliers_mask.any():
        ax.errorbar(d[outliers_mask], a[outliers_mask],
                    yerr=[a[outliers_mask] - cilo[outliers_mask], cihi[outliers_mask] - a[outliers_mask]],
                    fmt="X", capsize=3, ecolor="grey", markersize=14,
                    markerfacecolor="#d62728", markeredgecolor="black",
                    label="Huber-flagged outliers")
    for x, y, n_ in zip(d, a, names):
        ax.annotate(n_, (x, y), fontsize=8, xytext=(6, 4), textcoords="offset points")

    xline = np.linspace(d.min(), d.max(), 100)
    ax.plot(xline, ols.predict(xline.reshape(-1, 1)),
            "--", color="grey", linewidth=1.5,
            label=f"OLS    : {ols.coef_[0]:+.3f}·d+{ols.intercept_:.2f}  R²={r2_ols:.2f}")
    ax.plot(xline, hub.predict(xline.reshape(-1, 1)),
            "-", color="red", linewidth=2,
            label=f"Huber : {hub.coef_[0]:+.3f}·d+{hub.intercept_:.2f}  R²={r2_hub:.2f}")
    ax.axhline(0.5, ls=":", color="grey", alpha=0.5)
    ax.set_xlabel(f"DINOv3 centroid distance from {run['source']}")
    ax.set_ylabel("Zero-shot AUC (95% bootstrap CI)")
    ax.set_title(
        f"Robust distance ↔ AUC fit  ({run['source']} as source, n={len(oof)})\n"
        f"Partial Spearman after controlling imb & log(n) : ρ={rho_part_s:+.3f}  p={p_part_s:.3g}"
    )
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "robust_distance_auc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n      Plot → {OUT_DIR / 'robust_distance_auc.png'}")

    out = {
        "n_points": len(oof),
        "ols": {
            "slope": float(ols.coef_[0]),
            "intercept": float(ols.intercept_),
            "r2": float(r2_ols),
            "loocv_rmse": rmse_ols,
        },
        "huber": {
            "slope": float(hub.coef_[0]),
            "intercept": float(hub.intercept_),
            "r2": float(r2_hub),
            "loocv_rmse": rmse_hub,
            "outliers_flagged": flagged,
        },
        "multivariate_ols": {
            "coef_distance":          float(coefs[0]),
            "coef_class_imbalance":   float(coefs[1]),
            "coef_log_n":             float(coefs[2]),
            "intercept":              float(ols3.intercept_),
            "r2":                     float(r2_3),
            "loocv_rmse":             rmse_3,
        },
        "partial_correlation": {
            "spearman_rho": float(rho_part_s), "spearman_p": float(p_part_s),
            "pearson_r":    float(rho_part_p), "pearson_p":  float(p_part_p),
        },
    }
    with open(OUT_DIR / "robust_distance_auc.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"      Results JSON → {OUT_DIR / 'robust_distance_auc.json'}")
    return out


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="data/datasets")
    p.add_argument("--dino_dist", default=DINO_DIST_PATH)
    p.add_argument("--results",   default=RESULTS_PATH)
    p.add_argument("--n_per_ds",  type=int, default=N_PER_DATASET)
    p.add_argument("--skip_a",    action="store_true",
                   help="Skip Part A (cluster explanation) and only run B.")
    p.add_argument("--skip_b",    action="store_true",
                   help="Skip Part B (robust regression) and only run A.")
    return p.parse_args()


def main():
    global N_PER_DATASET
    args = parse_args()
    N_PER_DATASET = args.n_per_ds
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_a:
        part_a_explain_clusters(args)
    if not args.skip_b:
        part_b_robust_regression(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
