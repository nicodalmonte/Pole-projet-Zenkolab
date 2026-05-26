#!/bin/bash
#
#SBATCH --job-name=paper1_eval3
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper1_eval3_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper1_eval3_%j.err

#SBATCH --partition=prod10
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:0:0

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

# Paper 1 eval-only on 3 useful datasets:
# - ACRIMA_train (training dataset phase 1)
# - ORIGA_train  (training dataset phase 2)
# - REFUGE2_train (requested recalculation)
uv run python -m src.train.paper.paper1_efficientnet \
    --eval_ckpt checkpoints/paper1/paper1_phase2/v0/paper1_phase2_v0-epoch=03-val_auc=0.8046.ckpt \
    --eval_datasets ACRIMA_train ORIGA_train REFUGE2_train \
    --batch_size 32 \
    --num_workers 4 \
    --precision 16-mixed

echo "======================================================"
echo "  End time : $(date)"
echo "======================================================"
