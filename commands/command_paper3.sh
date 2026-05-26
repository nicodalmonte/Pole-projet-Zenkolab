#!/bin/bash
#
#SBATCH --job-name=paper3_yolo11
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper3_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper3_%j.err

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

# Phase 1 already trained — resume from its best.pt checkpoint
# Pass --phase1_ckpt to skip Phase 1 training
uv run python -m src.train.paper.paper3_yolo11 \
    --model yolo11s-cls.pt \
    --batch_size 32 \
    --max_epochs 25 \
    --lr1 1e-3 \
    --lr2 1e-4 \
    --num_workers 4 \
    --precision 16-mixed \
    --phase1_ckpt runs/classify/lightning_logs/paper3_yolo11_phase1/weights/best.pt

echo "======================================================"
echo "  End time : $(date)"
echo "======================================================"
