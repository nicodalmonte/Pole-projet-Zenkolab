#!/bin/bash
#SBATCH --job-name=threshold_opt_jraigs
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/threshold_opt_jraigs_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/threshold_opt_jraigs_%j.err
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
# Checkpoints — models trained on JRAIGS (phase1=jraigs, phase2=jraigs)
# ---------------------------------------------------------------------------
PAPER1_CKPT="checkpoints/paper1/paper1_phase2/v4/last.ckpt"
PAPER2_CKPT="checkpoints/paper2/model1/paper2_model1_phase2/v3/last.ckpt"
PAPER4_CKPT="checkpoints/paper4/paper4_phase2_v3-epoch=16-val_auc=0.9156.ckpt"
PAPER5_CKPT="checkpoints/paper5/paper5_phase2_v2-epoch=07-val_auc=0.8101.ckpt"

echo ""
echo "===== Threshold optimisation — JRAIGS-trained models ====="
echo "  Paper1 : $PAPER1_CKPT"
echo "  Paper2 : $PAPER2_CKPT"
echo "  Paper4 : $PAPER4_CKPT"
echo "  Paper5 : $PAPER5_CKPT"
echo ""

uv run python -m src.train.paper.threshold_opt_papers \
  --training_set  jraigs \
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
