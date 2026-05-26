#!/bin/bash
#
#SBATCH --job-name=paper5_mobilenet_gcn
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper5_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper5_%j.err

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

# Paper 5 — MobileNetV2 + Attention Module (5 blocks) + ResGCN
# Phase 1 : ACRIMA  (backbone frozen, focal γ=2, lr=1e-3, 25 epochs)
# Phase 2 : ORIGA   (backbone unfrozen, lr=1e-4, 25 epochs)
# Test    : G1020 + RIM-ONE + REFUGE2 + LAG + JRAIGS + Fundus + AIROGSLight
uv run python -m src.train.paper.paper5_mobilenet_gcn \
    --batch_size 32 \
    --max_epochs 25 \
    --lr1 1e-3 \
    --lr2 1e-4 \
    --focal_gamma 2.0 \
    --num_workers 4 \
    --precision 16-mixed

echo "======================================================"
echo "  End time : $(date)"
echo "======================================================"
