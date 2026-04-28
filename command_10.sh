#!/bin/bash
#
#SBATCH --job-name=Zenkolab
#SBATCH --output=/raid/home/students/dalmonte_nic/P-le-Projet-Zenkolab/logs/train_%j.out
#SBATCH --error=/raid/home/students/dalmonte_nic/P-le-Projet-Zenkolab/logs/train_%j.out

## Partition: prod10 | prod20 | prod40 | prod80
#SBATCH --partition=prod10

## GPU: nvidia_a100_1g.10gb | nvidia_a100_3g.40gb | nvidia_a100-sxm4-80gb
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1

## ntasks * cpus-per-task must be in [1 : 4 * nMIG]  (nMIG=4 for prod40)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

## RAM
#SBATCH --mem=8G

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
uv run python src/test/test_checkpoint_on_datasets.py \
  --checkpoint checkpoints/version_0/last.ckpt \
  --datasets REFUGE2
