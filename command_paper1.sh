#!/bin/bash
#
#SBATCH --job-name=paper1_efficientnet
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper1_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper1_%j.err

## Partition: prod10 | prod20 | prod40 | prod80
#SBATCH --partition=prod80

## GPU
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

# Paper 1 — EfficientNet-B0
# Phase 1: ACRIMA (frozen backbone, lr=1e-3, 25 epochs)
# Phase 2: ORIGA fine-tuning (unfrozen, lr=1e-4, 25 epochs)
# Test:    RIM-ONE + REFUGE2 + LAG + JRAIGS + G1020 + Fundus + AIROGSLight
uv run python -m src.train.paper.paper1_efficientnet \
    --batch_size 32 \
    --max_epochs 25 \
    --lr1 1e-3 \
    --lr2 1e-4 \
    --num_workers 4 \
    --precision 16-mixed 

echo "======================================================"
echo "  End time : $(date)"
echo "======================================================"
