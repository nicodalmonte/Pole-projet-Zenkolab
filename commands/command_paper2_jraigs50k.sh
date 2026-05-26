#!/bin/bash
#SBATCH --job-name=paper2_jraigs50k
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper2_jraigs50k_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/paper2_jraigs50k_%j.err
#SBATCH --partition=prod10
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:0:0

PROJECT=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab
cd "$PROJECT"
mkdir -p logs

uv run python -m src.train.paper.paper2_mobilenet \
  --batch_size 32 \
  --max_epochs 25 \
  --lr1 1e-3 \
  --lr2 1e-4 \
  --num_workers 4 \
  --precision 16-mixed \
  --phase1_dataset jraigs \
  --phase2_dataset jraigs \
  --max_split_samples 50000
