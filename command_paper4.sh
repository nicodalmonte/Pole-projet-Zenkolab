#!/bin/bash
#
#SBATCH --job-name=paper4_maskrcnn
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper4_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper4_%j.err

#SBATCH --partition=prod10
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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

# Paper 4 — Mask R-CNN ResNet-50 + FPN backbone for binary classification
# Phase 1 : ACRIMA  (backbone frozen, lr=1e-3, 25 epochs)
# Phase 2 : ORIGA   (backbone unfrozen, lr=1e-4, 25 epochs)
# Test    : G1020 + RIM-ONE + REFUGE2 + LAG + JRAIGS + Fundus + AIROGSLight
# Only the best checkpoint is kept (save_top_k=1, save_last=False)
uv run python -m src.train.paper.paper4_maskrcnn \
    --batch_size 32 \
    --max_epochs 25 \
    --lr1 1e-3 \
    --lr2 1e-4 \
    --num_workers 4 \
    --precision 16-mixed

echo "======================================================"
echo "  End time : $(date)"
echo "======================================================"
