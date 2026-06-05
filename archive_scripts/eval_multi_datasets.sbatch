#!/bin/bash
#
#SBATCH --job-name=eval_multi_datasets
#SBATCH --output=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab/outputs/eval_multi_datasets_%j.out
#SBATCH --error=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab/outputs/eval_multi_datasets_%j.err
#SBATCH --partition=prod10
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_DIR=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/.venv/lib/python3.12/site-packages:$PROJECT_DIR/.venv/lib64/python3.12/site-packages:${PYTHONPATH:-}"

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPU(s)     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Start time : $(date)"
echo "  CWD        : $(pwd)"
echo "======================================================"

mkdir -p "$PROJECT_DIR/outputs"

# Run evaluation on all requested datasets
/usr/bin/python3 eval_multi_datasets.py \
    --model_path outputs/origa_resnet/best_model_resnet50.pth \
    --batch_size 64 \
    --num_workers 8 \
    --image_size 224

echo "======================================================"
echo "  End time : $(date)"
echo "======================================================"
