#!/bin/bash
#SBATCH --job-name=threshold_opt_acrima_origa
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/threshold_opt_acrima_origa_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/threshold_opt_acrima_origa_%j.err
#SBATCH --partition=prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:0:0

PROJECT=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab
cd "$PROJECT"
mkdir -p logs

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "  Start time : $(date)"
echo "======================================================"

# ---------------------------------------------------------------------------
# Checkpoints — models trained on ACRIMA (phase1) → ORIGA (phase2)
# Best phase2 checkpoint for each paper before jraigs runs were introduced.
# ---------------------------------------------------------------------------
PAPER1_CKPT="checkpoints/paper1/paper1_phase2/v3/last.ckpt"
PAPER2_CKPT="checkpoints/paper2/model1/paper2_model1_phase2/v2/last.ckpt"
PAPER4_CKPT="checkpoints/paper4/paper4_phase2_v2-epoch=08-val_auc=0.7863.ckpt"
PAPER5_CKPT="checkpoints/paper5/paper5_phase2_v0-epoch=16-val_auc=0.8029.ckpt"

echo ""
echo "===== Threshold optimisation — ACRIMA/ORIGA-trained models ====="
echo "  Paper1 : $PAPER1_CKPT"
echo "  Paper2 : $PAPER2_CKPT"
echo "  Paper4 : $PAPER4_CKPT"
echo "  Paper5 : $PAPER5_CKPT"
echo ""

uv run python -m src.train.paper.threshold_opt_papers \
  --training_set  acrima_origa \
  --paper1_ckpt   "$PAPER1_CKPT" \
  --paper2_ckpt   "$PAPER2_CKPT" \
  --paper4_ckpt   "$PAPER4_CKPT" \
  --paper5_ckpt   "$PAPER5_CKPT" \
  --data_dir      data/datasets \
  --figures_dir   figures \
  --batch_size    64 \
  --num_workers   8

echo ""
echo "======================================================"
echo "  End time : $(date)"
echo "======================================================"
