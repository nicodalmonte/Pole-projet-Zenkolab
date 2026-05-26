#!/bin/bash
#
#SBATCH --job-name=paper2_mobilenet
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper2_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper2_%j.err

## Partition: prod10 | prod20 | prod40 | prod80
#SBATCH --partition=prod40

## GPU
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1

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

# Paper 2 — MobileNet (Esengönül & Cunha, Procedia 2023)
# 4 datasets × 3 model variants = 12 independent training runs
# Split: 72/8/20 (train/val/test) → 80-20 test holdout fidèle au papier
# Preprocessing: resize 224 → grayscale → center crop → CLAHE → 3-ch RGB
# Augmentation: zoom ±3.5%, rotation ±0.025° seulement (pas de flips)
# Datasets: ACRIMA (703), AIROGSLight (5000 cap), Harvard (1544, 3 classes), RIM-ONE (970)
uv run python -m src.train.paper.paper2_mobilenet \
    --batch_size 32 \
    --max_epochs 50 \
    --lr 1e-3 \
    --num_workers 4 \
    --precision 16-mixed

echo "======================================================"
echo "  End time : $(date)"
echo "======================================================"
