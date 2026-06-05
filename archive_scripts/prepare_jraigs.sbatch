#!/bin/bash
#
#SBATCH --job-name=prepare_jraigs
#SBATCH --output=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab/outputs/prepare_jraigs_%j.out
#SBATCH --error=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab/outputs/prepare_jraigs_%j.err
#SBATCH --partition=prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_DIR=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab
cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv/bin/activate"

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPU(s)     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Start time : $(date)"
echo "  CWD        : $(pwd)"
echo "======================================================"

python prepare_jraigs.py

echo "End time : $(date)"
