"""
Generates two figures illustrating generalization challenges:
1. AUC vs feature-space distance from training source (JRAIGS)
2. Effect of training on more clusters on unseen test sets
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Styling ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
GRAY   = "#6B7280"
RED    = "#DC2626"
PURPLE = "#7C3AED"

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Distance vs AUC
# ══════════════════════════════════════════════════════════════════════════════

with open("figures/distance_generalization/results_jraigs.json") as f:
    dist_data = json.load(f)

results = dist_data["results"]
datasets  = [r["dataset"]  for r in results]
distances = [r["distance"] for r in results]
aucs      = [r["test_auc"] for r in results]

# colour by distance zone
def zone_color(d):
    if d < 1.0:  return BLUE
    if d < 2.0:  return ORANGE
    return RED

colors = [zone_color(d) for d in distances]

# linear regression for trend line
slope, intercept, r, p, _ = stats.linregress(distances, aucs)
rho, p_spear = stats.spearmanr(distances, aucs)
x_line = np.linspace(0, max(distances) + 0.2, 200)
y_line = slope * x_line + intercept

fig1, ax1 = plt.subplots(figsize=(9, 5.5))

ax1.scatter(distances, aucs, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.6)
ax1.plot(x_line, y_line, color=GRAY, linewidth=1.4, linestyle="--", label=f"Linear fit  R²={r**2:.2f}")
ax1.axhline(0.5, color=GRAY, linewidth=0.8, linestyle=":", alpha=0.6)

# annotate every point
for ds, d, auc in zip(datasets, distances, aucs):
    label = ds.replace("(train)", "").replace("(test)", "").strip()
    va = "bottom" if auc > 0.7 else "top"
    offset = 0.012 if va == "bottom" else -0.012
    ax1.annotate(label, (d, auc + offset), fontsize=7.5, ha="center",
                 color="#374151", fontweight="normal")

# legend for zones
patches = [
    mpatches.Patch(color=BLUE,   label="Distance < 1  (same cluster)"),
    mpatches.Patch(color=ORANGE, label="Distance 1–2  (adjacent cluster)"),
    mpatches.Patch(color=RED,    label="Distance > 2  (far cluster)"),
]
ax1.legend(handles=patches + [plt.Line2D([0],[0], color=GRAY, ls="--",
           label=f"Linear fit  R²={r**2:.2f}")],
           loc="upper right", fontsize=9, framealpha=0.9)

ax1.set_xlabel("Feature-space distance from training source (JRAIGS)", fontsize=11)
ax1.set_ylabel("AUC on target dataset", fontsize=11)
ax1.set_title(
    f"Generalization degrades with feature-space distance\n"
    f"(Spearman ρ = {rho:.3f},  p = {p_spear:.3f}  |  trained on JRAIGS only)",
    fontsize=12, pad=10
)
ax1.set_xlim(-0.1, max(distances) + 0.3)
ax1.set_ylim(0.35, 1.02)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))

plt.tight_layout()
fig1.savefig("figures/distance_vs_auc.png", dpi=160, bbox_inches="tight")
print("Saved figures/distance_vs_auc.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – More training clusters → better generalisation
# ══════════════════════════════════════════════════════════════════════════════

with open("figures/cluster_gen/results.json") as f:
    cluster_data = json.load(f)

conditions_raw = {
    "A only\n(JRAIGS)":          cluster_data["JRAIGS only (cluster A, far)"],
    "B only\n(ORIGA+LAG)":       cluster_data["ORIGA+LAG (cluster B, far)"],
    "C only\n(ACRIMA+Harvard)":  cluster_data["ACRIMA+Harvard (cluster C, close)"],
    "A+B\n(2 clusters)":         cluster_data["JRAIGS+ORIGA/LAG (A+B, no close data)"],
    "A+B+C\n(3 clusters)":       cluster_data["JRAIGS+ORIGA/LAG+ACRIMA/Harvard (A+B+C)"],
}

test_sets   = ["RIMONE (unified)", "AIRROGS", "Fundus (unified)"]
ts_labels   = ["RIMONE\n(unseen cluster)", "AIRROGS\n(unseen)", "Fundus\n(unseen)"]
ts_colors   = [RED, BLUE, GREEN]

cond_labels = list(conditions_raw.keys())
n_cond      = len(cond_labels)
n_ts        = len(test_sets)
x           = np.arange(n_cond)
bar_w       = 0.22
offsets     = np.array([-1, 0, 1]) * bar_w

fig2, ax2 = plt.subplots(figsize=(10, 5.5))

for i, (ts, ts_label, color) in enumerate(zip(test_sets, ts_labels, ts_colors)):
    auc_vals = [conditions_raw[c][ts]["test_auc"] for c in cond_labels]
    bars = ax2.bar(x + offsets[i], auc_vals, bar_w,
                   color=color, alpha=0.85, label=ts_label,
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, auc_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.008,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=7.5, color="#374151")

# highlight best RIMONE bar
rimone_aucs = [conditions_raw[c]["RIMONE (unified)"]["test_auc"] for c in cond_labels]
best_idx = int(np.argmax(rimone_aucs))
ax2.annotate("Best RIMONE\ngeneralisation",
             xy=(x[best_idx] + offsets[0], rimone_aucs[best_idx] + 0.02),
             xytext=(x[best_idx] + offsets[0] - 0.5, rimone_aucs[best_idx] + 0.09),
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
             fontsize=9, color=RED, fontweight="bold")

ax2.axhline(0.5, color=GRAY, linewidth=0.8, linestyle=":", alpha=0.5, label="Random (AUC=0.5)")
ax2.set_xticks(x)
ax2.set_xticklabels(cond_labels, fontsize=10)
ax2.set_ylim(0.35, 1.05)
ax2.set_ylabel("AUC", fontsize=11)
ax2.set_xlabel("Training condition  (test sets are always held-out)", fontsize=11)
ax2.set_title(
    "Training on more clusters improves generalisation to unseen domains\n"
    "(RIMONE, AIRROGS, and Fundus never seen during training)",
    fontsize=12, pad=10
)
ax2.legend(title="Test set", fontsize=9, title_fontsize=9, loc="lower right", framealpha=0.9)

# shade A+B+C column to draw attention
ax2.axvspan(x[-1] - 0.4, x[-1] + 0.4, color=PURPLE, alpha=0.07, zorder=0)

plt.tight_layout()
fig2.savefig("figures/cluster_generalization.png", dpi=160, bbox_inches="tight")
print("Saved figures/cluster_generalization.png")

plt.show()
print("Done.")
