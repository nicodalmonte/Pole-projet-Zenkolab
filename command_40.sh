#!/bin/bash
#
#SBATCH --job-name=Zenkolab
#SBATCH --output=/raid/home/students/damon_mat/P-le-Projet-Zenkolab/logs/train_%j.out
#SBATCH --error=/raid/home/students/damon_mat/P-le-Projet-Zenkolab/logs/train_%j.out

## Partition: prod10 | prod20 | prod40 | prod80
#SBATCH --partition=prod40

## GPU: nvidia_a100_1g.10gb | nvidia_a100_3g.40gb | nvidia_a100-sxm4-80gb
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1

## ntasks * cpus-per-task must be in [1 : 4 * nMIG]  (nMIG=4 for prod40)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

## RAM
#SBATCH --mem=32G

## Max wall time
#SBATCH --time=24:0:0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT=/raid/home/students/damon_mat/P-le-Projet-Zenkolab

# Activate virtual environment
#I use uv

# Move to project root so relative paths (datasets/, checkpoints/) work
cd "$PROJECT"

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "  Python     : $(uv run python --version)"
echo "  Start time : $(date)"
echo "======================================================"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
##timm/vit_huge_plus_patch16_dinov3.lvd1689m
##timm/vit_large_patch16_dinov3.lvd1689m
uv run python -m src.train.train_split --batch_size 32 --precision 16-mixed --image_size 448 --backbone vit_large_patch16_dinov3.lvd1689m --class_weights 0.5 15.0 

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  End time   : $(date)"
echo "======================================================"