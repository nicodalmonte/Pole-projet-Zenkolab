#!/bin/bash
#SBATCH --job-name=eval_papers_jraigs
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/eval_papers_jraigs_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/eval_papers_jraigs_%j.err
#SBATCH --partition=prod10
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:0:0

PROJECT=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab
cd "$PROJECT"
mkdir -p logs

PAPER1_CKPT="checkpoints/paper1/paper1_phase2/v4/last.ckpt"
PAPER2_MODEL1_CKPT="checkpoints/paper2/model1/paper2_model1_phase2/v3/last.ckpt"
PAPER4_CKPT="checkpoints/paper4/paper4_phase2_v3-epoch=16-val_auc=0.9156.ckpt"
PAPER5_CKPT="checkpoints/paper5/paper5_phase2_v2-epoch=07-val_auc=0.8101.ckpt"

mkdir -p figures/paper1/jraigs figures/paper2/jraigs figures/paper4/jraigs figures/paper5/jraigs

echo "===== Evaluating Paper1 on all datasets ====="
uv run python -m src.train.paper.paper1_efficientnet \
  --eval_ckpt "$PAPER1_CKPT" \
  --phase1_dataset jraigs \
  --phase2_dataset jraigs \
  --figures_dir figures/paper1/jraigs \
  --batch_size 32 \
  --num_workers 4 \
  --precision 16-mixed

echo ""
echo "===== Evaluating Paper2 on all datasets ====="
uv run python -m src.train.paper.paper2_mobilenet \
  --eval_ckpt "$PAPER2_MODEL1_CKPT" \
  --phase1_dataset jraigs \
  --phase2_dataset jraigs \
  --figures_dir figures/paper2/jraigs \
  --batch_size 32 \
  --num_workers 4 \
  --precision 16-mixed

echo ""
echo "===== Evaluating Paper4 on all datasets ====="
uv run python -m src.train.paper.paper4_maskrcnn \
  --eval_ckpt "$PAPER4_CKPT" \
  --phase1_dataset jraigs \
  --phase2_dataset jraigs \
  --figures_dir figures/paper4/jraigs \
  --batch_size 32 \
  --num_workers 4 \
  --precision 16-mixed

echo ""
echo "===== Evaluating Paper5 on all datasets ====="
uv run python -m src.train.paper.paper5_mobilenet_gcn \
  --eval_ckpt "$PAPER5_CKPT" \
  --phase1_dataset jraigs \
  --phase2_dataset jraigs \
  --figures_dir figures/paper5/jraigs \
  --batch_size 32 \
  --num_workers 4 \
  --precision 16-mixed

echo ""
echo "All evaluations completed!"
