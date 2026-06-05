"""
Cluster generalization figure: explicit train/test cluster mapping.

Clusters:
  A — JRAIGS (train)  /  AIRROGS (test)
  B — ORIGA + LAG (train)  /  Fundus (test)
  C — ACRIMA + Harvard (train)  /  RIMONE (test)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Palette ────────────────────────────────────────────────────────────────
CA = "#2563EB"   # cluster A — blue
CB = "#EA580C"   # cluster B — orange
CC = "#16A34A"   # cluster C — green
GRAY = "#9CA3AF"

# ── Data ───────────────────────────────────────────────────────────────────
with open("figures/cluster_gen/results.json") as f:
    raw = json.load(f)

# Each condition: (label, clusters_trained, AUC_A_test, AUC_B_test, AUC_C_test)
# Test sets are always: AIRROGS (cluster A), Fundus (cluster B), RIMONE (cluster C)
conditions = [
    ("Train A only\n(JRAIGS)",          {0},    raw["JRAIGS only (cluster A, far)"]             ),
    ("Train B only\n(ORIGA+LAG)",        {1},    raw["ORIGA+LAG (cluster B, far)"]               ),
    ("Train C only\n(ACRIMA+Harvard)",   {2},    raw["ACRIMA+Harvard (cluster C, close)"]        ),
    ("Train A+B",                        {0,1},  raw["JRAIGS+ORIGA/LAG (A+B, no close data)"]   ),
    ("Train A+B+C",                      {0,1,2},raw["JRAIGS+ORIGA/LAG+ACRIMA/Harvard (A+B+C)"]),
]

test_keys    = ["AIRROGS", "Fundus (unified)", "RIMONE (unified)"]
test_labels  = ["AIRROGS\n(cluster A)", "Fundus\n(cluster B)", "RIMONE\n(cluster C)"]
test_colors  = [CA, CB, CC]
test_cluster = [0, 1, 2]   # which cluster index each test set belongs to

n_cond = len(conditions)
n_ts   = len(test_keys)
x      = np.arange(n_cond)
bar_w  = 0.22
offsets = np.array([-1, 0, 1]) * bar_w

# ── Layout: 2 rows, top = cluster legend, bottom = bars ────────────────────
fig = plt.figure(figsize=(12, 8))
gs  = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.35)

ax_top = fig.add_subplot(gs[0])
ax_bar = fig.add_subplot(gs[1])

# ── Top panel: cluster structure ────────────────────────────────────────────
ax_top.set_xlim(0, 1)
ax_top.set_ylim(0, 1)
ax_top.axis("off")

cluster_info = [
    (CA, "Cluster A",  "Train: JRAIGS",          "Test: AIRROGS"),
    (CB, "Cluster B",  "Train: ORIGA + LAG",      "Test: Fundus"),
    (CC, "Cluster C",  "Train: ACRIMA + Harvard", "Test: RIMONE"),
]

for i, (color, title, train_ds, test_ds) in enumerate(cluster_info):
    cx = 0.12 + i * 0.31
    # background box
    box = FancyBboxPatch((cx - 0.11, 0.05), 0.22, 0.88,
                         boxstyle="round,pad=0.02",
                         facecolor=color, alpha=0.10,
                         edgecolor=color, linewidth=2)
    ax_top.add_patch(box)
    ax_top.text(cx, 0.82, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)
    ax_top.text(cx, 0.56, train_ds, ha="center", va="center",
                fontsize=9.5, color="#374151")
    # separator
    ax_top.plot([cx - 0.09, cx + 0.09], [0.43, 0.43],
                color=color, linewidth=0.8, alpha=0.5)
    ax_top.text(cx, 0.28, test_ds, ha="center", va="center",
                fontsize=9.5, color="#374151", style="italic")

# legend for train/test rows
ax_top.text(0.02, 0.56, "Train:", ha="left", va="center", fontsize=9, color=GRAY)
ax_top.text(0.02, 0.28, "Test:", ha="left", va="center",  fontsize=9, color=GRAY, style="italic")
ax_top.text(0.5, 0.98,
            "Each cluster has a dedicated training split and a held-out test split",
            ha="center", va="top", fontsize=10, color="#374151", style="italic")

# ── Bottom panel: bars ───────────────────────────────────────────────────────
for ti, (tkey, tlabel, tcolor, tclus) in enumerate(
        zip(test_keys, test_labels, test_colors, test_cluster)):

    auc_vals = [cond[2][tkey]["test_auc"] for cond in conditions]
    trained_on_cluster = [tclus in cond[1] for cond in conditions]

    bars = ax_bar.bar(x + offsets[ti], auc_vals, bar_w,
                      color=tcolor, alpha=0.82, label=tlabel,
                      edgecolor="white", linewidth=0.6, zorder=3)

    for bar, val, seen in zip(bars, auc_vals, trained_on_cluster):
        bx = bar.get_x() + bar.get_width() / 2
        # value label
        ax_bar.text(bx, bar.get_height() + 0.012,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color="#374151", zorder=4)
        # star when test cluster was seen during training
        if seen:
            ax_bar.text(bx, bar.get_height() - 0.04,
                        "★", ha="center", va="top",
                        fontsize=9, color="white", zorder=5)

ax_bar.axhline(0.5, color=GRAY, linewidth=0.9, linestyle=":", alpha=0.5, zorder=1)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([c[0] for c in conditions], fontsize=10)
ax_bar.set_ylim(0.35, 1.08)
ax_bar.set_ylabel("AUC", fontsize=11)
ax_bar.set_xlabel("Training condition", fontsize=11)
ax_bar.set_title(
    "★ = test cluster was seen during training   "
    "(no star = true zero-shot cluster generalisation)",
    fontsize=10, color="#6B7280", pad=6, style="italic"
)
ax_bar.legend(title="Test set (always held-out split)", fontsize=9,
              title_fontsize=9, loc="lower right", framealpha=0.9)

# shade the A+B+C column
ax_bar.axvspan(x[-1] - 0.42, x[-1] + 0.42,
               color="#7C3AED", alpha=0.06, zorder=0)
ax_bar.text(x[-1], 1.04, "All clusters", ha="center", va="center",
            fontsize=9, color="#7C3AED", fontweight="bold")

# vertical grid
ax_bar.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
ax_bar.set_axisbelow(True)

fig.suptitle("Generalisation across glaucoma imaging clusters\n"
             "Training on more clusters consistently improves zero-shot transfer",
             fontsize=13, fontweight="bold", y=0.995)

plt.savefig("figures/cluster_generalization_v2.png", dpi=160, bbox_inches="tight")
print("Saved figures/cluster_generalization_v2.png")
plt.show()
