#!/bin/bash
#
#SBATCH --job-name=paper2_mobilenet
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper2_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper2_%j.err

## Partition: prod10 | prod20 | prod40 | prod80
#SBATCH --partition=prod10

## GPU
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

# Paper 2 — MobileNet-V2 three-model study
# Runs Model-1, Model-2, Model-FT sequentially.
# Each: Phase 1 (ACRIMA) → Phase 2 (ORIGA fine-tune) → test all datasets.
# Figures saved to figures/paper2/{model}/ + comparison_auc.png
uv run python -m src.train.paper.paper2_mobilenet \
    --batch_size 32 \
    --max_epochs 25 \
    --lr1 1e-3 \
    --lr2 1e-4 \
    --num_workers 4 \
    --precision 16-mixed

echo "======================================================"
echo "  End time : $(date)"
echo "======================================================"
