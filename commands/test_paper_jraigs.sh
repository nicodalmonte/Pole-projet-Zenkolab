#!/bin/bash
#SBATCH --job-name=test_papers_jraigs
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/test_papers_jraigs_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/test_papers_jraigs_%j.err
#SBATCH --partition=prod10
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:0:0

PROJECT=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab
cd "$PROJECT"
mkdir -p logs

echo "===== Testing Paper1 ====="
uv run python -m src.train.paper.paper1_efficientnet \
  --eval_only \
  --phase1_dataset jraigs \
  --phase2_dataset jraigs \
  --checkpoint_phase2 lightning_logs/paper1_phase2/version_5/checkpoints/*.ckpt 2>/dev/null || \
uv run python -m src.train.paper.paper1_efficientnet \
  --eval_only \
  --phase1_dataset jraigs \
  --phase2_dataset jraigs \
  --checkpoint_phase2 $(find lightning_logs -path "*paper1*phase2*" -name "*.ckpt" | sort | tail -1)

echo ""
echo "===== Testing Paper2 ====="
uv run python -m src.train.paper.paper2_mobilenet \
  --eval_only \
  --phase1_dataset jraigs \
  --phase2_dataset jraigs \
  --checkpoint_model1_phase2 $(find lightning_logs -path "*paper2*model1*phase2*" -name "*.ckpt" | sort | tail -1) \
  --checkpoint_model2_phase2 $(find lightning_logs -path "*paper2*model2*phase2*" -name "*.ckpt" | sort | tail -1) \
  --checkpoint_modelft_phase2 $(find lightning_logs -path "*paper2*model_ft*phase2*" -name "*.ckpt" | sort | tail -1)

echo ""
echo "===== Testing Paper4 ====="
uv run python -m src.train.paper.paper4_maskrcnn \
  --eval_only \
  --phase1_dataset jraigs \
  --phase2_dataset jraigs \
  --checkpoint_phase2 $(find lightning_logs -path "*paper4*phase2*" -name "*.ckpt" | sort | tail -1)

echo ""
echo "===== Testing Paper5 ====="
uv run python -m src.train.paper.paper5_mobilenet_gcn \
  --eval_only \
  --phase1_dataset jraigs \
  --phase2_dataset jraigs \
  --checkpoint_phase2 $(find lightning_logs -path "*paper5*phase2*" -name "*.ckpt" | sort | tail -1)

echo ""
echo "All tests completed!"
