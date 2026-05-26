"""
Generate test_metrics.png for all papers using optimal F1 thresholds.
Reads threshold_results.json for each paper and generates visualizations.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_threshold_results(threshold_json_path: str) -> dict:
    """Load threshold optimization results."""
    with open(threshold_json_path, "r") as f:
        return json.load(f)


def generate_test_metrics_plot(results: dict, paper_name: str, out_dir: str):
    """Generate test_metrics.png using metrics at optimal threshold."""
    metrics_names = ["f1", "acc", "sensitivity", "specificity", "auc"]
    datasets = [d for d in results["datasets"].keys()]
    
    # Extract metrics at optimal threshold
    metrics_data = {m: [] for m in metrics_names}
    
    for dataset in datasets:
        ds_data = results["datasets"][dataset]
        opt_metrics = ds_data.get("metrics_at_opt_threshold", {})
        if not opt_metrics:
            # Fall back to metrics_at_0.50 if opt not available
            opt_metrics = ds_data.get("metrics_at_0.50", {})
        
        for m in metrics_names:
            metrics_data[m].append(opt_metrics.get(m, 0.0))
    
    # Create plots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    colors = plt.cm.tab20(np.linspace(0, 1, len(datasets)))
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        ax.bar(range(len(datasets)), metrics_data[metric], color=colors)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} @ Optimal Threshold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    out_path = Path(out_dir) / "test_metrics_optimal_threshold.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    results_dirs = [
        ("paper1", "figures/threshold_opt/acrima_origa"),
        ("paper2", "figures/threshold_opt/acrima_origa"),
        ("paper4", "figures/threshold_opt/acrima_origa"),
        ("paper5", "figures/threshold_opt/acrima_origa"),
    ]
    
    for paper, results_dir in results_dirs:
        threshold_json = Path(results_dir) / "threshold_results.json"
        if not threshold_json.exists():
            print(f"⚠️  {threshold_json} not found, skipping")
            continue
        
        results = load_threshold_results(str(threshold_json))
        if paper not in results:
            print(f"⚠️  {paper} not in results, skipping")
            continue
        
        paper_results = results[paper]
        opt_threshold = paper_results.get("mean_optimal_threshold", 0.5)
        print(f"\n{paper.upper()} - Optimal Threshold: {opt_threshold:.4f}")
        
        generate_test_metrics_plot(paper_results, paper, results_dir)


if __name__ == "__main__":
    main()
