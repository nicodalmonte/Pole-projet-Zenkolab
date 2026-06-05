"""Round 2 of feature neutralisation, targeting the DEEP correlates of DINO
clustering identified by analyze_deep_features.py.

Background
----------
analyze_deep_features.py found that:
  • fft_low_freq_ratio   (ρ=+0.82)
  • fft_profile          (ρ=+0.78)
  • lbp_hist             (ρ=+0.75)
correlate much more strongly with DINO centroid distance than cosmetic
features.  Round 1 (test_feature_normalization.py) showed that cosmetic
preprocessing (border_crop, desat) does NOT shrink the clusters — confirming
the cosmetic features were proxies, not causes.

This round targets deep texture / local-contrast / FoV signatures:
  raw              : control
  clahe            : per-channel CLAHE on luminance → equalises local texture
  hist_equalize    : global histogram equalisation → flattens intensity dist
  disc_crop        : aggressive square crop around the brightest blob
                     → equalises field of view (the only strategy that helped
                     in the previous comparison)
  disc_crop_clahe  : FoV equalisation + texture equalisation (combo)

If NONE of these shrinks the mean pairwise centroid distance significantly,
it is strong evidence that DINO clusters cannot be collapsed by preprocessing
alone — and that further harmonisation requires interventional methods
(style transfer, domain adaptation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

# Re-use dataset roster and infrastructure from round 1
from src.generalization.test_feature_normalization import (
    SEED, BACKBONE, IMG_SIZE, N_PER_DATASET,
    dataset_factories, subsample, extract_features, centroid_matrix,
    build_dino_eval_pipeline,
)


# ---------------------------------------------------------------------------
# Strategy implementations (PIL → PIL)
# ---------------------------------------------------------------------------

def _raw(pil: Image.Image) -> Image.Image:
    return pil


def _clahe(pil: Image.Image) -> Image.Image:
    """CLAHE on L channel of LAB — equalises local contrast / texture."""
    arr = np.array(pil)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def _hist_equalize(pil: Image.Image) -> Image.Image:
    """Global histogram equalisation on luminance — flattens intensity dist."""
    arr = np.array(pil)
    yuv = cv2.cvtColor(arr, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    out = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(out)


def _disc_crop(pil: Image.Image, crop_frac: float = 0.4) -> Image.Image:
    """Crop around the brightest region (optic disc proxy) — equalises FoV."""
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


def _disc_crop_clahe(pil: Image.Image) -> Image.Image:
    return _clahe(_disc_crop(pil))


STRATEGY_FNS = {
    "raw":             _raw,
    "clahe":           _clahe,
    "hist_equalize":   _hist_equalize,
    "disc_crop":       _disc_crop,
    "disc_crop_clahe": _disc_crop_clahe,
}


# ---------------------------------------------------------------------------
# Per-strategy run (lighter version — only matrix + summary needed)
# ---------------------------------------------------------------------------

def run_strategy(strategy_name: str, fn, data_dir: str,
                 n_per_dataset: int, batch_size: int, num_workers: int,
                 model, device, out_dir: Path) -> tuple[np.ndarray, list[str]]:
    print(f"\n--- Strategy: {strategy_name} ---")
    pipeline = build_dino_eval_pipeline(fn)
    factories = dataset_factories(data_dir, pipeline)

    datasets: dict[str, Subset] = {}
    for name, build in factories.items():
        try:
            ds = build()
            datasets[name] = subsample(ds, n_per_dataset)
            print(f"  {name:<18} {len(ds):>6} total → {len(datasets[name])} sampled")
        except Exception as e:
            print(f"  {name:<18} SKIPPED ({e})")

    all_feats, all_labels = [], []
    for name, ds in datasets.items():
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=num_workers > 0)
        feats = extract_features(model, dl, device)
        all_feats.append(feats)
        all_labels.extend([name] * len(feats))
        print(f"    {name:<18} feats {feats.shape}")

    X = np.concatenate(all_feats, axis=0)
    names = list(datasets.keys())
    mat = centroid_matrix(X, all_labels, names)

    np.save(out_dir / f"centroid_{strategy_name}.npy", mat)
    with open(out_dir / f"centroid_{strategy_name}.json", "w") as f:
        json.dump({names[i]: {names[j]: float(mat[i, j]) for j in range(len(names))}
                   for i in range(len(names))}, f, indent=2)

    # Heatmap
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    plt.colorbar(im, ax=ax, label="DINO centroid L2 distance")
    ax.set_title(f"{strategy_name} — centroid distances")
    plt.tight_layout()
    plt.savefig(out_dir / f"centroid_heatmap_{strategy_name}.png", dpi=130, bbox_inches="tight")
    plt.close()
    return mat, names


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(matrices: dict[str, np.ndarray], names: list[str], out_dir: Path) -> dict:
    from scipy.stats import spearmanr

    summary = {}
    iu = np.triu_indices(len(names), k=1)
    base = matrices["raw"][iu]
    for strat, m in matrices.items():
        upper = m[iu]
        rho, p = spearmanr(base, upper)
        summary[strat] = {
            "mean_pairwise":   float(upper.mean()),
            "std_pairwise":    float(upper.std()),
            "max_pairwise":    float(upper.max()),
            "p90_pairwise":    float(np.percentile(upper, 90)),
            "mantel_vs_raw":   {"rho": float(rho), "p": float(p)},
        }

    import matplotlib.pyplot as plt
    strats = list(matrices.keys())
    means = [summary[s]["mean_pairwise"] for s in strats]
    stds  = [summary[s]["std_pairwise"]  for s in strats]
    maxes = [summary[s]["max_pairwise"]  for s in strats]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(strats)); w = 0.27
    ax.bar(x - w, means, w, label="mean",  color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.bar(x,     stds,  w, label="std",   color="#ff7f0e", edgecolor="black", linewidth=0.4)
    ax.bar(x + w, maxes, w, label="max",   color="#d62728", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(strats, rotation=15)
    ax.set_ylabel("Pairwise DINO centroid distance")
    ax.set_title("Round 2 — deep-targeted preprocessing (n=17 datasets)")
    ax.axhline(means[strats.index("raw")], ls=":", color="grey", alpha=0.6)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pairwise_distance_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"dataset_order": names, "per_strategy": summary}, f, indent=2)
    return summary


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default="data/datasets")
    p.add_argument("--out_dir",       default="figures/deep_normalization")
    p.add_argument("--n_per_dataset", type=int, default=N_PER_DATASET)
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--strategies",    default="all",
                   help="Comma-separated subset or 'all'")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nBackbone: {BACKBONE}")
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    to_run = list(STRATEGY_FNS.keys()) if args.strategies == "all" \
             else [s.strip() for s in args.strategies.split(",")]

    matrices: dict[str, np.ndarray] = {}
    names_ref: list[str] | None = None
    for strat in to_run:
        if strat not in STRATEGY_FNS:
            print(f"  Unknown strategy '{strat}', skipping.")
            continue
        mat, names = run_strategy(strat, STRATEGY_FNS[strat],
                                   args.data_dir, args.n_per_dataset,
                                   args.batch_size, args.num_workers,
                                   model, device, out_dir)
        if names_ref is None:
            names_ref = names
        matrices[strat] = mat

    if matrices and names_ref is not None:
        print("\n=== SUMMARY ===")
        summary = summarise(matrices, names_ref, out_dir)
        print(f"\n{'strategy':<22} {'mean':>6}  {'std':>6}  {'max':>6}  {'mantel(raw)':>13}")
        print("-" * 60)
        for s in to_run:
            if s not in summary: continue
            r = summary[s]
            print(f"{s:<22} {r['mean_pairwise']:>6.3f}  {r['std_pairwise']:>6.3f}  "
                  f"{r['max_pairwise']:>6.3f}  {r['mantel_vs_raw']['rho']:>+13.3f}")
        base = summary["raw"]
        print(f"\nΔ vs raw (negative = shrunk):")
        for s in to_run:
            if s not in summary or s == "raw": continue
            r = summary[s]
            d_mean = (r["mean_pairwise"] - base["mean_pairwise"]) / base["mean_pairwise"] * 100
            d_std  = (r["std_pairwise"]  - base["std_pairwise"])  / base["std_pairwise"]  * 100
            d_max  = (r["max_pairwise"]  - base["max_pairwise"])  / base["max_pairwise"]  * 100
            print(f"  {s:<22} mean {d_mean:+6.1f}%  std {d_std:+6.1f}%  max {d_max:+6.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
