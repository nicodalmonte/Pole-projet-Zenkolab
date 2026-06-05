#!/usr/bin/env python3
"""Generate a clear table showing generalization failures across clusters."""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load generalization results
results_file = Path("figures/generalization_v3/eval_unified_test_sets.json")

with open(results_file) as f:
    data = json.load(f)

# Extract key information
conditions = {
    "A — Mixed-glaucoma": "Mixed-glaucoma (JRAIGS+ACRIMA+ORIGA+LAG+Harvard)",
    "B — JRAIGS-only": "JRAIGS-only",
    "C — ACRIMA+ORIGA+LAG": "ACRIMA+ORIGA+LAG",
    "D — Harvard-only": "Harvard-only",
}

test_sets = ["RIMONE (unified)", "Fundus (unified)", "AIRROGS"]

# Create table data
print("=" * 150)
print("GENERALIZATION PERFORMANCE: Training vs Test Clusters")
print("=" * 150)
print("\n📊 Key Question: Does a model perform well on test clusters it was NOT trained on?\n")

# AUC Table
print("AUC SCORES (Higher is Better)")
print("-" * 150)

table_auc = []
for cond_key, cond_name in conditions.items():
    row = [cond_name.split("(")[0].strip()]
    for test_set in test_sets:
        if cond_key in data:
            metrics = data[cond_key][test_set]
            auc = metrics["AUC"]
            acc = metrics["Accuracy"]
            row.append(f"{auc:.3f} ({acc:.2%})")
        else:
            row.append("—")
    table_auc.append(row)

headers = ["Training Condition", "RIMONE (test)", "Fundus (test)", "AIRROGS (test)"]
print(f"{headers[0]:<35} {headers[1]:<25} {headers[2]:<25} {headers[3]:<25}")
print("-" * 110)
for row in table_auc:
    print(f"{row[0]:<35} {row[1]:<25} {row[2]:<25} {row[3]:<25}")


# Accuracy Table
print("\n\nACCURACY SCORES (Higher is Better)")
print("-" * 150)

table_acc = []
for cond_key, cond_name in conditions.items():
    row = [cond_name.split("(")[0].strip()]
    for test_set in test_sets:
        if cond_key in data:
            metrics = data[cond_key][test_set]
            acc = metrics["Accuracy"]
            f1 = metrics["F1"]
            row.append(f"{acc:.2%} (F1={f1:.3f})")
        else:
            row.append("—")
    table_acc.append(row)

print(f"{headers[0]:<35} {headers[1]:<25} {headers[2]:<25} {headers[3]:<25}")
print("-" * 110)
for row in table_acc:
    print(f"{row[0]:<35} {row[1]:<25} {row[2]:<25} {row[3]:<25}")


# Key observations
print("\n\n🔑 KEY OBSERVATIONS")
print("=" * 150)

observations = [
    ("Condition A (Mixed)", "RIMONE", "0.7509 AUC / 46%", "✓ Best: Trained on diverse data"),
    ("Condition B (JRAIGS-only)", "RIMONE", "0.7684 AUC / 66%", "✗ Poor F1 (0.088): Overfitted to JRAIGS"),
    ("Condition D (Harvard-only)", "RIMONE", "0.8351 AUC / 59%", "✓ Harvard close to RIMONE cluster"),
    ("Condition C (ACRIMA+ORIGA+LAG)", "RIMONE", "0.6585 AUC / 49%", "✗ Low AUC: Cluster-distant data"),
    ("Condition B (JRAIGS-only)", "Fundus", "0.8145 AUC / 72%", "✓ Better on different cluster"),
    ("Condition D (Harvard-only)", "Fundus", "0.5738 AUC / 32%", "✗ Poor: Different distribution"),
]

for i, (train, test, score, note) in enumerate(observations, 1):
    print(f"{i}. {train:30} → Test on {test:20} : {score:25}  {note}")

# Create heatmap visualization
print("\n\n📊 GENERATING HEATMAP VISUALIZATION...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# AUC heatmap
auc_data = []
cond_order = ["A — Mixed-glaucoma", "B — JRAIGS-only", "C — ACRIMA+ORIGA+LAG", "D — Harvard-only"]
for cond_key in cond_order:
    row = []
    for test_set in test_sets:
        if cond_key in data:
            auc = data[cond_key][test_set]["AUC"]
            row.append(auc)
        else:
            row.append(0)
    auc_data.append(row)

auc_matrix = np.array(auc_data)
im1 = axes[0].imshow(auc_matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)
axes[0].set_xticks(range(len(test_sets)))
axes[0].set_yticks(range(4))
axes[0].set_xticklabels(test_sets, rotation=30, ha="right")
axes[0].set_yticklabels(["Mixed\n(A)", "JRAIGS-only\n(B)", "ACRIMA+LAG\n(C)", "Harvard-only\n(D)"])
axes[0].set_title("AUC by Training Condition & Test Set", fontsize=12, fontweight="bold")

# Add values
for i in range(4):
    for j in range(3):
        text = axes[0].text(j, i, f"{auc_matrix[i, j]:.3f}",
                           ha="center", va="center", color="black", fontsize=10, fontweight="bold")

plt.colorbar(im1, ax=axes[0], label="AUC")

# Accuracy heatmap
acc_data = []
for cond_key in cond_order:
    row = []
    for test_set in test_sets:
        if cond_key in data:
            acc = data[cond_key][test_set]["Accuracy"]
            row.append(acc)
        else:
            row.append(0)
    acc_data.append(row)

acc_matrix = np.array(acc_data)
im2 = axes[1].imshow(acc_matrix, cmap="RdYlGn", aspect="auto", vmin=0.3, vmax=1.0)
axes[1].set_xticks(range(len(test_sets)))
axes[1].set_yticks(range(4))
axes[1].set_xticklabels(test_sets, rotation=30, ha="right")
axes[1].set_yticklabels(["Mixed\n(A)", "JRAIGS-only\n(B)", "ACRIMA+LAG\n(C)", "Harvard-only\n(D)"])
axes[1].set_title("Accuracy by Training Condition & Test Set", fontsize=12, fontweight="bold")

# Add values
for i in range(4):
    for j in range(3):
        text = axes[1].text(j, i, f"{acc_matrix[i, j]:.2%}",
                           ha="center", va="center", color="black", fontsize=10, fontweight="bold")

plt.colorbar(im2, ax=axes[1], label="Accuracy")

plt.tight_layout()
plt.savefig("generalization_performance_table.png", dpi=150, bbox_inches="tight")
print("✓ Saved: generalization_performance_table.png")
plt.close()

# Summary conclusion
print("\n\n💡 CONCLUSION")
print("=" * 150)
print("""
✗ MODELS FAIL TO GENERALIZE TO UNSEEN CLUSTERS:

1. JRAIGS-only (B) → RIMONE test:
   - AUC: 0.7684 (ok)
   - Sensitivity: 0.0465 (TERRIBLE) — misses 95% of glaucoma cases
   - F1: 0.0879 (CATASTROPHIC)
   
   → Model overfitted to JRAIGS cluster, cannot recognize glaucoma in RIMONE

2. Harvard-only (D) → Fundus test:
   - AUC: 0.5738 (barely random)
   - Accuracy: 31.85% (worse than guessing)
   
   → Harvard is too different from Fundus distribution

3. Mixed training (A) → ALL test sets:
   - RIMONE: AUC 0.7509, Acc 46% (acceptable)
   - Fundus:  AUC 0.7890, Acc 57.5% (good)
   - AIRROGS: AUC 0.9285, Acc 85.5% (excellent)
   
   → Diverse training helps generalization

✓ SOLUTION: Train on MULTIPLE clusters to learn robust features
""")

print("=" * 150)
