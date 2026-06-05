"""Robustness check of the Mantel analysis with deeper, non-cosmetic features.

Background
----------
analyze_distance_relation.py found two cosmetic features (black_border_ratio,
mean_saturation) correlated with DINO centroid distance (Mantel ρ=0.665, 0.549).
test_feature_normalization.py then showed that neutralising those features
DOES NOT shrink the DINO clusters — meaning the Mantel correlations reflect
*association* (those features covary with what DINO sees) rather than
*causation* (DINO doesn't use them directly).

This script asks the next question:
  Do "deep", non-cosmetic features also Mantel-correlate with DINO clustering?
  If YES → DINO is capturing real structural acquisition/anatomy signal that
           even handcrafted texture/frequency descriptors can detect.
  If NO  → DINO encodes something even deeper that no simple descriptor
           captures (and the previous Mantel hits were pure proxies of
           cluster identity, not of anything we can characterise).

Deep features added (all on 256×256 thumbnails for speed):
  • edge_density            — fraction of pixels with Sobel magnitude > thresh
  • local_entropy_mean      — Shannon entropy of grayscale (3×3 local hist)
  • fft_high_freq_ratio     — FFT energy above radial cutoff / total
  • fft_low_freq_ratio      — FFT energy below low cutoff / total
  • rgb_hist_64             — 4×4×4 joint RGB histogram (vector descriptor)
  • lbp_hist_256            — 8-neighbour LBP histogram (vector descriptor)
  • fft_radial_profile_32   — radial-binned FFT magnitude (vector descriptor)

For each scalar feature, per-dataset mean → per-pair |Δz| distance →
Mantel-Spearman vs DINO centroid distance.

For each vector descriptor, per-dataset mean histogram → per-pair distance:
  • RGB / LBP histograms  : Jensen-Shannon divergence
  • FFT radial profile    : cosine distance

Output: figures/deep_features/{deep_features.json,deep_feature_bars.png}
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift
from scipy.stats import spearmanr

from src.datasets import (
    ACRIMADataset, ORIGADataset, LAGDataset, AIROGSLightDataset,
    HarvardGlaucomaDataset, JRAIGSDataset, REFUGE2Dataset, G1020Dataset,
    MultichannelGlaucomaBenchmarkDataset,
)
from src.datasets.RIMONE import RIMONEDataset


SEED = 42
N_PER_DATASET = 100
STAT_THUMBNAIL = 256
NUM_WORKERS = 16
DINO_DIST_PATH = (
    "figures/clustering/vit_small_patch16_dinov3_lvd1689m/centroid_distances.json"
)
OUT_DIR = Path("figures/deep_features")


# ---------------------------------------------------------------------------
# Dataset roster (must match DINO centroid_distances.json keys)
# ---------------------------------------------------------------------------

def dataset_factories(data_dir: str):
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


def get_paths(dataset, n: int, seed: int = SEED) -> list[str]:
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
# Per-image deep features
# ---------------------------------------------------------------------------

SCALAR_FEATURE_NAMES = [
    "edge_density",
    "local_entropy_mean",
    "fft_high_freq_ratio",
    "fft_low_freq_ratio",
]

# Vector descriptor lengths
RGB_BINS = 4                                # 4×4×4 = 64 dims
LBP_DIMS = 256                              # 8-neighbour LBP histogram
FFT_RADIAL_BINS = 32                        # radial profile bins


def _shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _local_entropy_mean(gray_u8: np.ndarray, win: int = 9) -> float:
    """Mean of per-window Shannon entropy over a grid (cheap approximation)."""
    H, W = gray_u8.shape
    step = win
    vals = []
    for y in range(0, H - win, step):
        for x in range(0, W - win, step):
            patch = gray_u8[y:y + win, x:x + win]
            hist, _ = np.histogram(patch, bins=16, range=(0, 256))
            p = hist / max(hist.sum(), 1)
            vals.append(_shannon_entropy(p))
    return float(np.mean(vals)) if vals else 0.0


def _lbp_hist(gray_u8: np.ndarray) -> np.ndarray:
    """Uniform LBP histogram (256 bins, 8 neighbours, P=8, R=1) — cheap impl."""
    g = gray_u8.astype(np.int32)
    H, W = g.shape
    center = g[1:H - 1, 1:W - 1]
    bits = (
        ((g[0:H - 2, 0:W - 2] >= center) << 7) |
        ((g[0:H - 2, 1:W - 1] >= center) << 6) |
        ((g[0:H - 2, 2:W]   >= center) << 5) |
        ((g[1:H - 1, 2:W]   >= center) << 4) |
        ((g[2:H,   2:W]     >= center) << 3) |
        ((g[2:H,   1:W - 1] >= center) << 2) |
        ((g[2:H,   0:W - 2] >= center) << 1) |
        ((g[1:H - 1, 0:W - 2] >= center) << 0)
    ).astype(np.uint8)
    h, _ = np.histogram(bits, bins=LBP_DIMS, range=(0, LBP_DIMS))
    return h / max(h.sum(), 1)


def _fft_radial_profile(gray_u8: np.ndarray, n_bins: int = FFT_RADIAL_BINS) -> tuple[np.ndarray, float, float]:
    """Radial-binned FFT magnitude + (high_freq_ratio, low_freq_ratio)."""
    f = np.abs(fftshift(fft2(gray_u8.astype(np.float32) / 255.0)))
    H, W = f.shape
    cy, cx = H // 2, W // 2
    y, x = np.indices(f.shape)
    r = np.hypot(x - cx, y - cy)
    r_max = r.max()
    bins = np.linspace(0, r_max, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        mask = (r >= bins[b]) & (r < bins[b + 1])
        if mask.any():
            profile[b] = float(f[mask].mean())
    total = max(profile.sum(), 1e-12)
    p = profile / total
    low_cut = max(1, n_bins // 8)
    high_cut = max(1, 3 * n_bins // 4)
    return p, float(p[high_cut:].sum()), float(p[:low_cut].sum())


def _rgb_hist(arr_u8: np.ndarray, bins: int = RGB_BINS) -> np.ndarray:
    h, _ = np.histogramdd(
        arr_u8.reshape(-1, 3), bins=(bins, bins, bins),
        range=((0, 256), (0, 256), (0, 256)),
    )
    h = h.flatten()
    return h / max(h.sum(), 1)


def image_features(path: str):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    ow, oh = img.size
    wt = max(1, ow * STAT_THUMBNAIL // max(ow, oh))
    ht = max(1, oh * STAT_THUMBNAIL // max(ow, oh))
    img_small = img.resize((wt, ht), Image.BILINEAR)
    arr = np.asarray(img_small)  # u8 H×W×3

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(sx, sy)
    edge_density = float((mag > 30).mean())

    ent = _local_entropy_mean(gray)
    fft_prof, hi_ratio, lo_ratio = _fft_radial_profile(gray)
    lbp = _lbp_hist(gray)
    rgb = _rgb_hist(arr)

    return {
        "scalar": {
            "edge_density":        edge_density,
            "local_entropy_mean":  ent,
            "fft_high_freq_ratio": hi_ratio,
            "fft_low_freq_ratio":  lo_ratio,
        },
        "rgb_hist":     rgb,
        "lbp_hist":     lbp,
        "fft_profile":  fft_prof,
    }


def dataset_features(name: str, paths: list[str]) -> dict | None:
    rows = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for r in ex.map(image_features, paths, chunksize=4):
            if r is not None:
                rows.append(r)
    if not rows:
        return None

    scalars = {k: float(np.mean([r["scalar"][k] for r in rows])) for k in SCALAR_FEATURE_NAMES}
    rgb_mean    = np.mean(np.stack([r["rgb_hist"]    for r in rows]), axis=0)
    lbp_mean    = np.mean(np.stack([r["lbp_hist"]    for r in rows]), axis=0)
    fft_mean    = np.mean(np.stack([r["fft_profile"] for r in rows]), axis=0)
    # renormalise after averaging (still distributions)
    rgb_mean /= max(rgb_mean.sum(), 1e-12)
    lbp_mean /= max(lbp_mean.sum(), 1e-12)
    fft_mean /= max(fft_mean.sum(), 1e-12)

    return {
        "scalar":      scalars,
        "rgb_hist":    rgb_mean,
        "lbp_hist":    lbp_mean,
        "fft_profile": fft_mean,
        "n_sampled":   len(rows),
    }


# ---------------------------------------------------------------------------
# Pairwise distances + Mantel
# ---------------------------------------------------------------------------

def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * (_kl(p, m) + _kl(q, m))


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def pairwise(names: list[str], fn, get_vec) -> np.ndarray:
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = fn(get_vec(names[i]), get_vec(names[j]))
            M[i, j] = d; M[j, i] = d
    return M


def upper_tri(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices_from(M, k=1)]


def mantel(D1: np.ndarray, D2: np.ndarray) -> tuple[float, float]:
    rho, p = spearmanr(upper_tri(D1), upper_tri(D2))
    return float(rho), float(p)


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="data/datasets")
    p.add_argument("--dino_dist", default=DINO_DIST_PATH)
    p.add_argument("--n_per_ds",  type=int, default=N_PER_DATASET)
    return p.parse_args()


def main():
    global N_PER_DATASET
    args = parse_args()
    N_PER_DATASET = args.n_per_ds
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting deep features ({N_PER_DATASET} imgs/dataset, thumbnail={STAT_THUMBNAIL}px)")
    factories = dataset_factories(args.data_dir)
    stats: dict[str, dict] = {}
    for name, build in factories.items():
        try:
            ds = build()
            paths = get_paths(ds, N_PER_DATASET)
            r = dataset_features(name, paths)
            if r is None:
                print(f"  {name:<16} no features extracted")
                continue
            stats[name] = r
            s = r["scalar"]
            print(f"  {name:<16} n={r['n_sampled']:>3}  "
                  f"edge={s['edge_density']:.3f}  ent={s['local_entropy_mean']:.2f}  "
                  f"hi={s['fft_high_freq_ratio']:.3f}  lo={s['fft_low_freq_ratio']:.3f}")
        except Exception as e:
            print(f"  {name:<16} SKIPPED ({e})")

    with open(args.dino_dist) as f:
        dino_dict = json.load(f)
    names = [n for n in stats if n in dino_dict]
    print(f"\n{len(names)} datasets shared with DINO centroids: {names}")

    dino_mat = np.array([[dino_dict[a].get(b, np.nan) for b in names] for a in names])

    # ── Scalar features → per-feature Mantel ──────────────────────────────
    scalar_mat = np.array([[stats[n]["scalar"][k] for k in SCALAR_FEATURE_NAMES] for n in names])
    mu = scalar_mat.mean(axis=0); sd = scalar_mat.std(axis=0); sd[sd == 0] = 1.0
    Z = (scalar_mat - mu) / sd

    print("\n── Per-scalar-feature Mantel (DEEP non-cosmetic) ──")
    per_scalar = {}
    for j, fn in enumerate(SCALAR_FEATURE_NAMES):
        D = np.abs(Z[:, j:j+1] - Z[:, j:j+1].T)
        rho, p = mantel(D, dino_mat)
        per_scalar[fn] = {"rho": rho, "p": p}
        print(f"  {fn:<22} rho={rho:+.3f}  p={p:.2g}")

    # Combined Mantel using all 4 deep scalars
    D_all_scalar = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    rho_scalar_combined, p_scalar_combined = mantel(D_all_scalar, dino_mat)
    print(f"  {'[ALL 4 deep scalars]':<22} rho={rho_scalar_combined:+.3f}  p={p_scalar_combined:.2g}")

    # ── Vector descriptors → Mantel ───────────────────────────────────────
    print("\n── Vector descriptor Mantel ──")
    vec_names_fns = [
        ("rgb_hist_JS",     "rgb_hist",    jensen_shannon),
        ("lbp_hist_JS",     "lbp_hist",    jensen_shannon),
        ("fft_profile_cos", "fft_profile", cosine_dist),
    ]
    per_vec = {}
    for label, key, dist_fn in vec_names_fns:
        D = pairwise(names, dist_fn, lambda n_, k=key: stats[n_][k])
        rho, p = mantel(D, dino_mat)
        per_vec[label] = {"rho": rho, "p": p}
        print(f"  {label:<22} rho={rho:+.3f}  p={p:.2g}")

    # ── Compare to cosmetic features (from analyze_distance_relation.py) ──
    cosmetic_path = Path("figures/distance_analysis/cluster_explanation.json")
    cosmetic_summary = {}
    if cosmetic_path.exists():
        with open(cosmetic_path) as f:
            cosmetic = json.load(f)
        cosmetic_summary = cosmetic.get("per_feature_mantel", {})
        cosmetic_summary["_global_8_cosmetic"] = cosmetic.get("global_mantel", {})

    # ── Plot ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    all_rows = []
    for k, v in per_scalar.items():
        all_rows.append((f"deep · {k}", v["rho"], "deep"))
    all_rows.append(("deep · [4 scalars combined]", rho_scalar_combined, "deep"))
    for k, v in per_vec.items():
        all_rows.append((f"deep · {k}", v["rho"], "deep"))
    for k, v in cosmetic_summary.items():
        if k.startswith("_"): continue
        all_rows.append((f"cosmetic · {k}", v["rho"], "cosmetic"))
    if "_global_8_cosmetic" in cosmetic_summary:
        all_rows.append(("cosmetic · [8 features combined]",
                         cosmetic_summary["_global_8_cosmetic"]["rho"], "cosmetic"))

    all_rows.sort(key=lambda x: x[1], reverse=True)
    labels = [r[0] for r in all_rows]
    rhos   = [r[1] for r in all_rows]
    colors = ["#2ca02c" if r[2] == "deep" else "#1f77b4" for r in all_rows]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(labels))))
    bars = ax.barh(labels, rhos, color=colors, edgecolor="black", linewidth=0.4)
    for b, r in zip(bars, rhos):
        ax.text(r + 0.01 if r >= 0 else r - 0.01, b.get_y() + b.get_height()/2,
                f"{r:+.2f}", va="center", ha="left" if r >= 0 else "right",
                fontsize=8)
    ax.axvline(0, color="grey", linewidth=0.7)
    ax.set_xlabel("Mantel Spearman ρ vs DINO centroid distance")
    ax.set_title("Deep features vs cosmetic features — what correlates with DINO clustering?")
    # legend via proxy
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="#2ca02c", label="deep (texture / freq / joint colour)"),
        mpatches.Patch(color="#1f77b4", label="cosmetic (resolution, mean RGB, …)"),
    ], loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "deep_feature_bars.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {out_png}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "n_datasets": len(names),
        "dataset_order": names,
        "per_scalar_mantel": per_scalar,
        "all_deep_scalars_combined": {"rho": rho_scalar_combined, "p": p_scalar_combined},
        "per_vector_mantel": per_vec,
        "cosmetic_reference": cosmetic_summary,
        "scalar_features_per_dataset": {n: stats[n]["scalar"] for n in names},
    }
    with open(OUT_DIR / "deep_features.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON → {OUT_DIR / 'deep_features.json'}")


if __name__ == "__main__":
    main()
