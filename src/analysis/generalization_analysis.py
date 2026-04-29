"""Generalization analysis — all papers, all datasets.

Three training strategies:
  - ACRIMA→ORIGA : P1 EfficientNet, P2 MobileNet-ft, P4 MaskRCNN
  - JRAIGS 50k   : P1 EfficientNet (JRAIGS)
  - P5 strategy  : P5 MobNet+GCN (ORIGA + REFUGE2 — current results from ACRIMA→ORIGA run)

Run:
    python -m src.analysis.generalization_analysis
"""

from __future__ import annotations

import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("figures/generalization_analysis")

# ---------------------------------------------------------------------------
# Model registry  (label → {dataset_key: {metric: value}})
# ---------------------------------------------------------------------------
RESULTS: dict[str, dict] = {
    "P1 EfficientNet\nACRIMA→ORIGA": json.load(open(
        "figures/paper1/acrima_origa/test_results_with_train_grouped.json"
    )),
    "P1 EfficientNet\nJRAIGS 50k": json.load(open(
        "figures/paper1/jraigs/test_results_with_train_grouped.json"
    )),
    "P2 MobileNet-ft\nACRIMA→ORIGA": json.load(open(
        "figures/paper2/model_ft/test_results.json"
    )),
    "P4 MaskRCNN\nACRIMA→ORIGA": json.load(open(
        "figures/paper4/acrima_origa/test_results.json"
    )),
    "P5 MobNet+GCN\nACRIMA→ORIGA": json.load(open(
        "figures/paper5/acrima_origa/test_results.json"
    )),
}

MODEL_NAMES = list(RESULTS.keys())
N_MODELS    = len(MODEL_NAMES)

# Training-domain keys per model
TRAIN_DOMAINS: dict[str, set] = {
    "P1 EfficientNet\nACRIMA→ORIGA": {"ACRIMA_test", "ORIGA_test"},
    "P1 EfficientNet\nJRAIGS 50k":   {"JRAIGS_test"},
    "P2 MobileNet-ft\nACRIMA→ORIGA": {"ACRIMA_test", "ORIGA_test"},
    "P4 MaskRCNN\nACRIMA→ORIGA":     {"ACRIMA_test", "ORIGA_test"},
    "P5 MobNet+GCN\nACRIMA→ORIGA":   {"ACRIMA_test", "ORIGA_test"},
}

# Strategy groupings — P5 is kept separate (different architecture & datasets)
STRATEGY_ACRIMA_ORIGA = [
    "P1 EfficientNet\nACRIMA→ORIGA",
    "P2 MobileNet-ft\nACRIMA→ORIGA",
    "P4 MaskRCNN\nACRIMA→ORIGA",
]
STRATEGY_JRAIGS = [
    "P1 EfficientNet\nJRAIGS 50k",
]
STRATEGY_P5 = [
    "P5 MobNet+GCN\nACRIMA→ORIGA",
]

# Display order: training-domain cols first, then external
DATASETS_ORDER = [
    "ACRIMA_test",
    "ORIGA_test",
    "JRAIGS_test",
    "JRAIGS_all",
    "RIM-ONE_all",
    "REFUGE2_train_labeled",
    "LAG_all",
    "G1020_all",
    "Fundus_all",
    "AIROGSLight_all",
]

DATASET_LABELS = {
    "ACRIMA_test":           "ACRIMA\n(train↑)",
    "ORIGA_test":            "ORIGA\n(train↑)",
    "JRAIGS_test":           "JRAIGS\n(train↑)",
    "JRAIGS_all":            "JRAIGS\n(ext.)",
    "RIM-ONE_all":           "RIM-ONE",
    "REFUGE2_train_labeled": "REFUGE2",
    "LAG_all":               "LAG",
    "G1020_all":             "G1020",
    "Fundus_all":            "Fundus",
    "AIROGSLight_all":       "AIROGSLight",
}

COMMON_EXTERNAL = [
    "JRAIGS_all",
    "RIM-ONE_all",
    "REFUGE2_train_labeled",
    "LAG_all",
    "G1020_all",
    "Fundus_all",
    "AIROGSLight_all",
]

N_DS = len(DATASETS_ORDER)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(model: str, ds: str, metric: str) -> float:
    return RESULTS[model].get(ds, {}).get(metric, float("nan"))


def matrix(metric_key: str) -> np.ndarray:
    return np.array([
        [get(m, ds, metric_key) for ds in DATASETS_ORDER]
        for m in MODEL_NAMES
    ])


def train_mean(model: str, metric: str) -> float:
    vals = [get(model, d, metric)
            for d in TRAIN_DOMAINS[model]
            if not np.isnan(get(model, d, metric))]
    return float(np.nanmean(vals)) if vals else float("nan")


def ext_mean(model: str, metric: str) -> float:
    all_ds = set(RESULTS[model].keys())
    ext_ds = all_ds - TRAIN_DOMAINS[model]
    vals   = [get(model, d, metric)
              for d in ext_ds
              if not np.isnan(get(model, d, metric))]
    return float(np.nanmean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Figure 1 — AUC heatmap (models × datasets)
# ---------------------------------------------------------------------------

def fig_auc_heatmap():
    auc = matrix("test_auc")

    fig, ax = plt.subplots(figsize=(14, 4.5))
    masked = np.ma.masked_invalid(auc)
    cmap   = plt.cm.RdYlGn.copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(masked, cmap=cmap, vmin=0.45, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="AUC-ROC", fraction=0.025, pad=0.02)

    ax.set_xticks(range(N_DS))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS_ORDER], fontsize=8.5)
    ax.set_yticks(range(N_MODELS))
    ax.set_yticklabels(MODEL_NAMES, fontsize=8)

    for i in range(N_MODELS):
        for j in range(N_DS):
            v = auc[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=7, color="grey")
                continue
            color = "white" if v < 0.58 or v > 0.90 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold")

    ax.axvline(2.5, color="white", lw=2.5, linestyle="--")
    ax.text(1.0,  -0.75, "Training domains", ha="center", fontsize=8,
            color="navy", transform=ax.transData)
    ax.text(6.5, -0.75, "External / unseen datasets", ha="center", fontsize=8,
            color="darkred", transform=ax.transData)

    ax.set_title(
        "AUC-ROC — all models × all test datasets\n"
        "(grey = not evaluated  |  dashed = training / external boundary)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "1_auc_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: 1_auc_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 2 — Generalization gap
# ---------------------------------------------------------------------------

def fig_generalization_gap():
    tr_aucs  = [train_mean(m, "test_auc") for m in MODEL_NAMES]
    ext_aucs = [ext_mean(m, "test_auc")   for m in MODEL_NAMES]

    x  = np.arange(N_MODELS)
    w  = 0.30
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(x - w/2, tr_aucs,  w, label="Training domain (mean AUC)",
           color="#4CAF50", alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, ext_aucs, w, label="External datasets (mean AUC)",
           color="#F44336", alpha=0.85, edgecolor="white")

    for i, (tr, ex) in enumerate(zip(tr_aucs, ext_aucs)):
        if np.isnan(tr) or np.isnan(ex):
            continue
        gap  = tr - ex
        sign = f"−{gap:.2f}" if gap >= 0 else f"+{abs(gap):.2f}"
        col  = "darkred" if gap >= 0 else "darkgreen"
        ax.annotate(
            sign,
            xy=(x[i] + w/2, ex),
            xytext=(x[i] + w/2, ex + 0.035),
            ha="center", fontsize=8.5, color=col, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
        )

    ax.axhline(0.5, linestyle="--", color="grey", lw=0.8, label="Random (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, fontsize=8)
    ax.set_ylabel("Mean AUC-ROC")
    ax.set_ylim(0.35, 1.05)
    ax.set_title(
        "Generalization gap — mean AUC on training domain vs external datasets",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "2_generalization_gap.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: 2_generalization_gap.png")


# ---------------------------------------------------------------------------
# Figure 3 — Per-dataset AUC variance
# ---------------------------------------------------------------------------

def fig_dataset_variance():
    usable = [ds for ds in DATASETS_ORDER
              if sum(not np.isnan(get(m, ds, "test_auc")) for m in MODEL_NAMES) >= 2]

    means, stds, mins_, maxs_ = [], [], [], []
    for ds in usable:
        vals = [get(m, ds, "test_auc") for m in MODEL_NAMES
                if not np.isnan(get(m, ds, "test_auc"))]
        means.append(np.mean(vals)); stds.append(np.std(vals))
        mins_.append(np.min(vals)); maxs_.append(np.max(vals))

    x = np.arange(len(usable))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, means, 0.55, yerr=stds, color="#2196F3", alpha=0.75, capsize=5,
           label="Mean ± std AUC across models")
    ax.fill_between(x, mins_, maxs_, alpha=0.18, color="#2196F3",
                    label="Min–Max range")
    ax.plot(x, maxs_, "^", color="green",   ms=8, label="Best model AUC")
    ax.plot(x, mins_, "v", color="crimson", ms=8, label="Worst model AUC")

    for i, ds in enumerate(usable):
        if any(ds in TRAIN_DOMAINS[m] for m in MODEL_NAMES):
            ax.annotate("↑train", xy=(i, 0.33), ha="center",
                        fontsize=7, color="navy")

    ax.axhline(0.5, linestyle="--", color="grey", lw=0.8, label="Random (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in usable], fontsize=9)
    ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.28, 1.10)
    ax.set_title(
        "AUC distribution across all models per dataset\n"
        "(high variance = architecture-dependent, not robust)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "3_dataset_auc_variance.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: 3_dataset_auc_variance.png")


# ---------------------------------------------------------------------------
# Figure 4 — AUC vs F1 scatter
# ---------------------------------------------------------------------------

def fig_auc_vs_f1_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, N_DS))
    ds_color = {ds: colors[i] for i, ds in enumerate(DATASETS_ORDER)}

    for ax, ykey, ylabel in [
        (axes[0], "test_f1",  "F1 Score"),
        (axes[1], "test_acc", "Accuracy"),
    ]:
        handles: dict[str, object] = {}
        for m in MODEL_NAMES:
            for ds in DATASETS_ORDER:
                auc_v = get(m, ds, "test_auc")
                y_v   = get(m, ds, ykey)
                if np.isnan(auc_v) or np.isnan(y_v):
                    continue
                in_domain = ds in TRAIN_DOMAINS[m]
                sc = ax.scatter(
                    auc_v, y_v,
                    color=ds_color[ds],
                    marker="*" if in_domain else "o",
                    s=110 if in_domain else 55,
                    alpha=0.78,
                    edgecolors="black" if in_domain else "none",
                    linewidths=0.6,
                )
                lbl = DATASET_LABELS[ds].replace("\n", " ")
                if lbl not in handles:
                    handles[lbl] = sc

        ax.axvline(0.5, color="grey", lw=0.6, linestyle=":")
        ax.axhline(0.5, color="grey", lw=0.6, linestyle=":")
        ax.set_xlabel("AUC-ROC", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0.4, 1.05); ax.set_ylim(-0.05, 1.08)
        ax.set_title(f"AUC vs {ylabel}  (★ = train domain)", fontsize=10)
        ax.grid(alpha=0.22)
        ax.legend(handles=list(handles.values()), labels=list(handles.keys()),
                  fontsize=7, loc="upper left", ncol=2)

    plt.suptitle(
        "High AUC ≠ high F1/Accuracy — especially on imbalanced external sets",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "4_auc_vs_f1_acc_scatter.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: 4_auc_vs_f1_acc_scatter.png")


# ---------------------------------------------------------------------------
# Figure 5 — Sensitivity vs Specificity
# ---------------------------------------------------------------------------

def fig_sens_spec():
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, N_DS))
    ds_color = {ds: colors[i] for i, ds in enumerate(DATASETS_ORDER)}

    handles: dict[str, object] = {}
    for m in MODEL_NAMES:
        for ds in DATASETS_ORDER:
            sens = get(m, ds, "test_sensitivity")
            spec = get(m, ds, "test_specificity")
            if np.isnan(sens) or np.isnan(spec):
                continue
            in_dom = ds in TRAIN_DOMAINS[m]
            sc = ax.scatter(spec, sens,
                            color=ds_color[ds],
                            marker="*" if in_dom else "o",
                            s=110 if in_dom else 55,
                            alpha=0.72,
                            edgecolors="black" if in_dom else "none",
                            linewidths=0.5)
            lbl = DATASET_LABELS[ds].replace("\n", " ")
            if lbl not in handles:
                handles[lbl] = sc

    ax.plot([0, 1], [1, 0], "k--", lw=0.8, alpha=0.35, label="Balanced")
    ax.text(0.03, 0.97, "Predict ALL\nGlaucoma", fontsize=8, color="crimson")
    ax.text(0.83, 0.03, "Predict ALL\nNormal",   fontsize=8, color="darkblue")
    ax.set_xlabel("Specificity", fontsize=11)
    ax.set_ylabel("Sensitivity", fontsize=11)
    ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.1)
    ax.set_title(
        "Sensitivity vs Specificity — all models × all datasets\n"
        "(points near axes = degenerate predictions)",
        fontsize=11,
    )
    ax.legend(handles=list(handles.values()), labels=list(handles.keys()),
              fontsize=8, loc="center right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "5_sensitivity_vs_specificity.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: 5_sensitivity_vs_specificity.png")


# ---------------------------------------------------------------------------
# Figure 6 — AUC + F1 side-by-side heatmap (all models)
# ---------------------------------------------------------------------------

def fig_f1_heatmap():
    auc = matrix("test_auc")
    f1  = matrix("test_f1")
    xlabels = [DATASET_LABELS[d] for d in DATASETS_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(22, 4.5))
    for ax, data, title, vmin, vmax in [
        (axes[0], auc, "AUC-ROC",  0.45, 1.0),
        (axes[1], f1,  "F1 Score", 0.0,  1.0),
    ]:
        masked = np.ma.masked_invalid(data)
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad("lightgrey")
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        ax.set_xticks(range(N_DS)); ax.set_xticklabels(xlabels, fontsize=8.5)
        ax.set_yticks(range(N_MODELS)); ax.set_yticklabels(MODEL_NAMES, fontsize=8)
        ax.axvline(2.5, color="white", lw=2, linestyle="--")
        ax.set_title(title, fontsize=11)
        for i in range(N_MODELS):
            for j in range(N_DS):
                v = data[i, j]
                if np.isnan(v):
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=7, color="grey")
                    continue
                c = "white" if v < 0.25 or v > 0.85 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=c)

    fig.suptitle(
        "AUC vs F1 — models × datasets  "
        "(grey = N/A | models can show decent AUC but near-zero F1 on imbalanced sets)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "6_auc_and_f1_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: 6_auc_and_f1_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 7 — Radar charts (external datasets)
# ---------------------------------------------------------------------------

def fig_radar_external():
    common_ext = [ds for ds in COMMON_EXTERNAL
                  if all(not np.isnan(get(m, ds, "test_auc")) for m in MODEL_NAMES)]

    N      = len(common_ext)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    labels = [DATASET_LABELS[d].replace("\n", " ") for d in common_ext]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                             subplot_kw=dict(polar=True))
    axes = axes.flatten()
    colors = plt.cm.Set2(np.linspace(0, 1, N_MODELS))

    for idx, (m, color) in enumerate(zip(MODEL_NAMES, colors)):
        ax   = axes[idx]
        vals = [get(m, ds, "test_auc") for ds in common_ext] + \
               [get(m, common_ext[0], "test_auc")]
        ax.plot(angles, vals, color=color, lw=2)
        ax.fill(angles, vals, color=color, alpha=0.18)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6)
        ax.axhline(0.5, color="grey", lw=0.7, linestyle="--", alpha=0.5)
        ax.set_title(m, size=9, pad=12, color=color, fontweight="bold")

    for j in range(N_MODELS, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "AUC on common external datasets — one radar per model\n"
        "(inner dashed circle = 0.50 random baseline)",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "7_radar_external_auc.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: 7_radar_external_auc.png")


# ---------------------------------------------------------------------------
# Figure 8 — JRAIGS vs ACRIMA/ORIGA training: direct comparison
# ---------------------------------------------------------------------------

def fig_jraigs_vs_acrima_origa():
    m_small = "P1 EfficientNet\nACRIMA→ORIGA"
    m_large = "P1 EfficientNet\nJRAIGS 50k"

    shared = [ds for ds in DATASETS_ORDER
              if not np.isnan(get(m_small, ds, "test_auc"))
              and not np.isnan(get(m_large, ds, "test_auc"))]

    x = np.arange(len(shared))
    w = 0.28
    metrics = [("test_auc", "AUC-ROC"), ("test_f1", "F1"), ("test_sensitivity", "Sensitivity")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (mkey, mlabel) in zip(axes, metrics):
        vals_small = [get(m_small, ds, mkey) for ds in shared]
        vals_large = [get(m_large, ds, mkey) for ds in shared]

        ax.bar(x - w/2, vals_small, w, label="ACRIMA→ORIGA (~1 k imgs)",
               color="#5C6BC0", alpha=0.85)
        ax.bar(x + w/2, vals_large, w, label="JRAIGS 50k (~50 k imgs)",
               color="#EF6C00", alpha=0.85)

        ax.axhline(0.5, linestyle="--", color="grey", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d].replace("\n", " ") for d in shared],
                           rotation=28, ha="right", fontsize=8)
        ax.set_ylabel(mlabel); ax.set_ylim(0, 1.05)
        ax.set_title(f"EfficientNet-B0: {mlabel}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

        for i, ds in enumerate(shared):
            if ds in TRAIN_DOMAINS[m_small]:
                ax.get_xticklabels()[i].set_color("navy")
            elif ds in TRAIN_DOMAINS[m_large]:
                ax.get_xticklabels()[i].set_color("darkorange")

    plt.suptitle(
        "Same architecture (EfficientNet-B0): small dataset (ACRIMA→ORIGA) vs large (JRAIGS 50k)\n"
        "More training data does not always mean better generalization",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "8_jraigs_vs_acrima_origa.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: 8_jraigs_vs_acrima_origa.png")


# ---------------------------------------------------------------------------
# Figure 9 — Summary table
# ---------------------------------------------------------------------------

def fig_summary_table():
    usable = [ds for ds in DATASETS_ORDER
              if sum(not np.isnan(get(m, ds, "test_auc")) for m in MODEL_NAMES) >= 2]

    fig, ax = plt.subplots(figsize=(15, 0.55 * len(usable) + 2))
    ax.axis("off")

    col_labels = ["Dataset", "Type", "Best AUC", "Worst AUC",
                  "Range", "Mean AUC", "Mean F1", "Verdict"]
    rows = []
    for ds in usable:
        auc_vals = [(get(m, ds, "test_auc"), m) for m in MODEL_NAMES
                    if not np.isnan(get(m, ds, "test_auc"))]
        f1_vals  = [get(m, ds, "test_f1") for m in MODEL_NAMES
                    if not np.isnan(get(m, ds, "test_f1"))]
        best_v, best_m   = max(auc_vals, key=lambda x: x[0])
        worst_v, worst_m = min(auc_vals, key=lambda x: x[0])
        mean_auc = np.mean([v for v, _ in auc_vals])
        mean_f1  = np.mean(f1_vals) if f1_vals else float("nan")
        rng      = best_v - worst_v

        is_train = any(ds in TRAIN_DOMAINS[m] for m in MODEL_NAMES)
        ds_type  = "TRAIN domain" if is_train else "External"

        if mean_auc >= 0.82:
            verdict = "✓ Good"
        elif mean_auc >= 0.68:
            verdict = "~ Moderate"
        else:
            verdict = "✗ Poor"

        rows.append([
            DATASET_LABELS[ds].replace("\n", " "),
            ds_type,
            f"{best_v:.3f}  ({best_m.split(chr(10))[0]})",
            f"{worst_v:.3f}  ({worst_m.split(chr(10))[0]})",
            f"{rng:.3f}",
            f"{mean_auc:.3f}",
            f"{mean_f1:.3f}" if not np.isnan(mean_f1) else "N/A",
            verdict,
        ])

    table = ax.table(cellText=rows, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)

    for i, row in enumerate(rows):
        verdict_cell = table[(i + 1, 7)]
        if "Good"     in row[7]: verdict_cell.set_facecolor("#c8e6c9")
        elif "Moderate" in row[7]: verdict_cell.set_facecolor("#fff9c4")
        else:                      verdict_cell.set_facecolor("#ffcdd2")

        if row[1] == "TRAIN domain":
            for j in range(8):
                table[(i + 1, j)].set_facecolor("#e3f2fd")

    ax.set_title(
        "Generalization summary — all datasets across all models\n"
        "(blue = training domain  |  red = poor  |  yellow = moderate  |  green = good)",
        fontsize=11, pad=14,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "9_summary_table.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: 9_summary_table.png")


# ---------------------------------------------------------------------------
# Figure 10 — In-domain vs Out-of-domain: F1 and AUC, 3 strategies
# ---------------------------------------------------------------------------
# Layout: 2 rows (F1, AUC)  ×  3 columns (ACRIMA→ORIGA | JRAIGS | P5)
# Each panel: grouped bars, one pair per model in the strategy.
#   Green bar  = mean on training domain datasets
#   Red bar    = mean on all other (external) datasets
#   Δ annotated above bars

def _bar_strategy(ax, models: list[str], metric_key: str, metric_label: str,
                  title: str, in_color: str, out_color: str,
                  in_legend: str, out_legend: str):
    short = [m.split("\n")[0] for m in models]
    in_v  = [train_mean(m, metric_key)  for m in models]
    out_v = [ext_mean(m, metric_key) for m in models]

    x = np.arange(len(models))
    w = 0.32

    b_in  = ax.bar(x - w/2, in_v,  w, color=in_color,  alpha=0.85,
                   label=in_legend,  edgecolor="white")
    b_out = ax.bar(x + w/2, out_v, w, color=out_color, alpha=0.85,
                   label=out_legend, edgecolor="white")

    for i, (iv, ov) in enumerate(zip(in_v, out_v)):
        if np.isnan(iv) or np.isnan(ov):
            continue
        gap  = iv - ov
        ypos = max(iv, ov) + 0.04
        ax.text(x[i], ypos, f"Δ={gap:+.2f}", ha="center", fontsize=8,
                color="darkred" if gap > 0 else "darkgreen", fontweight="bold")

    for bars in [b_in, b_out]:
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h) or h < 0.005:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0.5, linestyle="--", color="grey", lw=0.8, alpha=0.6,
               label="Random (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9, rotation=10, ha="right")
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_ylim(0, 1.22)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)


def fig_indomain_vs_outdomain():
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.subplots_adjust(hspace=0.50, wspace=0.30)

    strategies = [
        # (models, title_prefix, in_legend, in_color)
        (STRATEGY_ACRIMA_ORIGA,
         "ACRIMA→ORIGA strategy",
         "In-domain\n(ACRIMA + ORIGA mean)",
         "#2E7D32"),
        (STRATEGY_JRAIGS,
         "JRAIGS strategy",
         "In-domain\n(JRAIGS)",
         "#1565C0"),
        (STRATEGY_P5,
         "P5 strategy\n(MobNet+GCN)",
         "In-domain\n(ACRIMA + ORIGA mean)",
         "#6A1B9A"),
    ]

    for col, (models, title_prefix, in_legend, in_color) in enumerate(strategies):
        for row, (mkey, mlabel) in enumerate([("test_f1",  "F1 Score"),
                                               ("test_auc", "AUC-ROC")]):
            _bar_strategy(
                ax=axes[row, col],
                models=models,
                metric_key=mkey,
                metric_label=mlabel,
                title=f"{title_prefix}\n{mlabel}: in-domain vs out-of-domain",
                in_color=in_color,
                out_color="#C62828",
                in_legend=in_legend,
                out_legend="Out-of-domain\n(all other datasets mean)",
            )

    fig.suptitle(
        "In-domain vs Out-of-domain performance — F1 and AUC per training strategy\n"
        "(Δ = in-domain − out-of-domain: positive → generalization gap / overfitting)",
        fontsize=13,
    )
    fig.savefig(OUT_DIR / "1_indomain_vs_outdomain.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Saved: 1_indomain_vs_outdomain.png")


# ---------------------------------------------------------------------------
# Figures 11a / 11b / 11c — Confusion-matrix-style heatmaps per strategy
# Relative coloring: red = worst value in this panel, green = best
# ---------------------------------------------------------------------------

def _strategy_heatmap(strategy_models: list[str], strategy_name: str, fname: str):
    # All datasets tested by at least one model in this strategy
    tested_ds = [ds for ds in DATASETS_ORDER
                 if any(not np.isnan(get(m, ds, "test_f1"))
                        for m in strategy_models)]
    n_rows = len(strategy_models)
    n_cols = len(tested_ds)
    xlabels = [DATASET_LABELS[d] for d in tested_ds]

    fig_w = max(13, n_cols * 1.6 + 5)
    fig_h = max(4,  n_rows * 1.4 + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig.subplots_adjust(wspace=0.10)

    # Identify in-domain column indices for this strategy
    in_domain_cols = [
        j for j, ds in enumerate(tested_ds)
        if any(ds in TRAIN_DOMAINS[m] for m in strategy_models)
    ]

    for ax, mkey, panel_title in [
        (axes[0], "test_f1",  "F1 Score"),
        (axes[1], "test_auc", "AUC-ROC"),
    ]:
        data = np.array([
            [get(m, ds, mkey) for ds in tested_ds]
            for m in strategy_models
        ])

        # Relative min/max (ignoring NaN): worst = red, best = green
        valid = data[~np.isnan(data)]
        vmin  = float(np.min(valid)) if len(valid) else 0.0
        vmax  = float(np.max(valid)) if len(valid) else 1.0
        if vmax - vmin < 1e-6:
            vmin, vmax = max(0.0, vmin - 0.1), min(1.0, vmax + 0.1)

        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad("lightgrey")
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(panel_title, fontsize=9)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(xlabels, fontsize=8.5, rotation=30, ha="right")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(strategy_models, fontsize=8.5)
        ax.set_title(panel_title, fontsize=12, pad=10)

        # Separator between in-domain and out-of-domain columns
        if in_domain_cols:
            boundary = max(in_domain_cols) + 0.5
            ax.axvline(boundary, color="white", lw=2.5, linestyle="--")
            n_out = n_cols - len(in_domain_cols)
            if n_out > 0:
                mid_in  = np.mean(in_domain_cols)
                mid_out = np.mean([j for j in range(n_cols) if j not in in_domain_cols])
                ax.text(mid_in, -0.75, "In-domain",
                        ha="center", fontsize=8, color="navy",
                        transform=ax.transData)
                ax.text(mid_out, -0.75, "Out-of-domain",
                        ha="center", fontsize=8, color="darkred",
                        transform=ax.transData)

        # Cell annotations
        for i in range(n_rows):
            for j in range(n_cols):
                v = data[i, j]
                if np.isnan(v):
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=8, color="grey")
                    continue
                norm = (v - vmin) / (vmax - vmin + 1e-9)
                txt_color = "black" if 0.20 < norm < 0.80 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, color=txt_color, fontweight="bold")

    fig.suptitle(
        f"{strategy_name}\n"
        "F1 Score (left) and AUC-ROC (right) — relative colour scale: "
        "red = worst, green = best within each panel",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def fig_strategy_heatmap_acrima():
    _strategy_heatmap(
        STRATEGY_ACRIMA_ORIGA,
        "ACRIMA→ORIGA training strategy  (P1 EfficientNet · P2 MobileNet-ft · P4 MaskRCNN)",
        "2a_heatmap_acrima_origa.png",
    )


def fig_strategy_heatmap_jraigs():
    _strategy_heatmap(
        STRATEGY_JRAIGS,
        "JRAIGS 50k training strategy  (P1 EfficientNet)",
        "2b_heatmap_jraigs.png",
    )


def fig_strategy_heatmap_p5():
    _strategy_heatmap(
        STRATEGY_P5,
        "P5 strategy  (MobNet+GCN — ORIGA + REFUGE2)",
        "2c_heatmap_p5.png",
    )


# ---------------------------------------------------------------------------
# Stdout stats
# ---------------------------------------------------------------------------

def print_stats():
    print("\n" + "=" * 72)
    print("  GENERALIZATION ANALYSIS — KEY NUMBERS")
    print("=" * 72)
    for m in MODEL_NAMES:
        tr  = train_mean(m, "test_auc")
        ex  = ext_mean(m, "test_auc")
        f1e = ext_mean(m, "test_f1")
        print(f"\n  {m.replace(chr(10), ' ')}")
        print(f"    Train-domain AUC  : {tr:.3f}")
        print(f"    External mean AUC : {ex:.3f}  (gap = {tr-ex:+.3f})")
        print(f"    External mean F1  : {f1e:.3f}")

    print("\n  Per-dataset (mean ± std AUC across models):")
    for ds in DATASETS_ORDER:
        vals = [get(m, ds, "test_auc") for m in MODEL_NAMES
                if not np.isnan(get(m, ds, "test_auc"))]
        if not vals:
            continue
        is_train = any(ds in TRAIN_DOMAINS[m] for m in MODEL_NAMES)
        tag = " [TRAIN]" if is_train else " [EXT]  "
        print(f"    {DATASET_LABELS[ds].replace(chr(10),' '):<22}{tag}  "
              f"mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
              f"min={np.min(vals):.3f}  max={np.max(vals):.3f}  "
              f"(n={len(vals)})")
    print()


if __name__ == "__main__":
    # Delete all old figures and regenerate only the requested ones
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print_stats()
    # Figure 1 — in-domain vs out-of-domain F1 and AUC (3 strategies)
    fig_indomain_vs_outdomain()
    # Figures 2a/b/c — heatmaps per strategy (red=worst, green=best)
    fig_strategy_heatmap_acrima()
    fig_strategy_heatmap_jraigs()
    fig_strategy_heatmap_p5()
    print(f"\nAll figures saved to: {OUT_DIR}/")
