#!/bin/bash
#
#SBATCH --job-name=retfound_glaucoma
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/train_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/train_%j.out

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
set -e

PROJECT=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab

# Activate virtual environment
source "$PROJECT/.venv/bin/activate"

# Move to project root so relative paths (datasets/, checkpoints/) work
cd "$PROJECT"

# Create logs dir if needed
mkdir -p logs checkpoints results

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "  Python     : $(python --version)"
echo "  Start time : $(date)"
echo "======================================================"

# ---------------------------------------------------------------------------
# Install / verify dependencies (fast if already cached by uv)
# ---------------------------------------------------------------------------
uv pip install --quiet timm huggingface_hub scikit-learn matplotlib

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
python -u src/train.py \
    --data_root    datasets/        \
    --epochs       30               \
    --batch_size   32               \
    --lr           1e-4             \
    --weight_decay 0.05             \
    --num_workers  8                \
    --patience     7                \
    --freeze_epochs 3

echo ""
echo "======================================================"
echo "  Training done! Running evaluation on test set …"
echo "======================================================"

# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------
python -u src/evaluate.py \
    --data_root   datasets/                   \
    --checkpoint  checkpoints/best_model.pth  \
    --split       test                        \
    --output_dir  results/

echo ""
echo "End time : $(date)"
echo "======================================================"
