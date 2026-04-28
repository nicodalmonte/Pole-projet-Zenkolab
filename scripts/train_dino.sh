#!/bin/bash
#SBATCH --job-name=dinov3_large
#SBATCH --output=/raid/home/students/dalmonte_nic/P-le-Projet-Zenkolab/logs/dinov3_%j.out
#SBATCH --error=/raid/home/students/dalmonte_nic/P-le-Projet-Zenkolab/logs/dinov3_%j.err
#SBATCH --partition=prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:0:0

set -e
PROJECT=/raid/home/students/dalmonte_nic/P-le-Projet-Zenkolab
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT"
mkdir -p logs checkpoints results

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "Start time : $(date)"

python src/train/train.py \
  --dataset JRAIGS \
  --backbone vit_large_patch16_dinov3.lvd1689m \
  --image_size 896 \
  --batch_size 32 \
  --precision 16-mixed \
  --max_epochs 50 \
  --lr 1e-4 \
  --unfreeze_backbone_epoch 3

echo "End time : $(date)"