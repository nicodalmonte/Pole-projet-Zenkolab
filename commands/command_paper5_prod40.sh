#!/bin/bash
#
#SBATCH --job-name=paper5_gcn_prod40
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper5_prod40_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper5_prod40_%j.err

## Partition: prod10 | prod20 | prod40 | prod80
#SBATCH --partition=prod40

## GPU: A100 40 GB
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:0:0

# ---------------------------------------------------------------------------
PROJECT=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab
cd "$PROJECT"
mkdir -p logs

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "  Python     : $(uv run python --version)"
echo "  Start time : $(date)"
echo "======================================================"

# Paper 5 — MobileNetV2 + Attention Module (5 blocks) + ResGCN
#
# Strategy (faithful to paper):
#   Datasets  : ORIGA (70/15/15 split) + REFUGE2 train split (full, no holdout)
#               REFUGE2 val/test have no public labels → only "train" used
#               ORIGA train + REFUGE2 full-train combined → single training phase
#   Backbone  : MobileNetV2 pretrained (ImageNet), trained end-to-end
#   Optimizer : Adam lr=0.001, ReduceLROnPlateau(factor=0.5, patience=10)
#   Epochs    : 300 max  (early-stop patience=30 on val_auc)
#   Loss      : Focal loss (gamma=2)
#   Batch     : 64  (A100 40 GB)
#
# Figures generated:
#   figures/paper5_prod40/training_curves.png       ← loss + AUC + LR schedule
#   figures/paper5_prod40/test_metrics.png          ← bar chart by dataset
#   figures/paper5_prod40/test_metrics_by_metric.png← bar chart by metric
#   figures/paper5_prod40/roc_curves.png            ← all ROC curves overlaid
#   figures/paper5_prod40/confusion_matrices.png    ← CM grid
#   figures/paper5_prod40/score_distributions.png   ← P(glaucoma) histograms
#   figures/paper5_prod40/test_results.json
#   figures/paper5_prod40/test_results_by_metric.json
uv run python -m src.train.paper.paper5_mobilenet_gcn \
    --batch_size 64 \
    --max_epochs 300 \
    --lr 1e-3 \
    --focal_gamma 2.0 \
    --num_workers 8 \
    --precision 16-mixed \
    --train_ratio 0.70 \
    --val_ratio   0.15 \
    --ckpt_dir    checkpoints/paper5_prod40 \
    --figures_dir figures/paper5_prod40

echo "======================================================"
echo "  End time : $(date)"
echo "======================================================"
