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

python -m src.train

echo "End time : $(date)"