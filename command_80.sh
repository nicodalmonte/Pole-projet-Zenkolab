#!/bin/bash
#
#SBATCH --job-name=Zenkolab
#SBATCH --output=/raid/home/students/dalmonte_nic/P-le-Projet-Zenkolab/logs/train_%j.out
#SBATCH --error=/raid/home/students/dalmonte_nic/P-le-Projet-Zenkolab/logs/train_%j.out

## Partition: prod10 | prod20 | prod40 | prod80
#SBATCH --partition=prod80
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

## ntasks * cpus-per-task must be in [1 : 4 * nMIG]  (nMIG=4 for prod40)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

## RAM
#SBATCH --mem=64G

## Max wall time
#SBATCH --time=24:0:0

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT=/raid/home/students/dalmonte_nic/P-le-Projet-Zenkolab

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
uv run python src/train/train.py \
  --dataset JRAIGS \
  --backbone vit_large_patch16_dinov3.lvd1689m \
  --image_size 224 \
  --batch_size 32 \
  --precision 16-mixed \
  --max_epochs 50 \
  --lr 1e-4 \
  --unfreeze_backbone_epoch 3
