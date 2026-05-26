#!/bin/bash
#
#SBATCH --job-name=generalisation_exp
#SBATCH --output=/raid/home/students/damon_mat/P-le-Projet-Zenkolab/logs/generalisation_%j.out
#SBATCH --error=/raid/home/students/damon_mat/P-le-Projet-Zenkolab/logs/generalisation_%j.out

## Partition: prod10 | prod20 | prod40 | prod80
#SBATCH --partition=prod80

## GPU: nvidia_a100_1g.10gb | nvidia_a100_3g.40gb | nvidia_a100-sxm4-80gb
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb

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
# Generalisation Experiment
# ---------------------------------------------------------------------------
# Training parameters:
# - Diverse model: samples across datasets excluding REFUGE2 to total 400 images
# - Single model: samples 400 images from JRAIGS
# - Test: REFUGE2 labeled 'train' split (used as held-out test set)
# - Both trained on glaucoma classification task

uv run python -m src.dataset_experiment.generalisation_experience \
  --model_type dinov3 \
  --batch_size 32 \
  --precision 16-mixed \
  --max_epochs 100 \
  --train_images_total 1000 \
  --single_source JRAIGS \
  --single_source_split train \
  --test_source REFUGE2 \
  --test_split train \
  --seed 42
