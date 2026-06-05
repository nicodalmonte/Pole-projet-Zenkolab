#!/bin/bash
#
#SBATCH --job-name=add_blur_metric
#SBATCH --output=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab/outputs/add_blur_metric_%j.out
#SBATCH --error=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab/outputs/add_blur_metric_%j.err
#SBATCH --partition=prod40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_DIR=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab
cd "$PROJECT_DIR"

source "$PROJECT_DIR/.venv/bin/activate"

mkdir -p "$PROJECT_DIR/outputs"

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  Start time : $(date)"
echo "  CWD        : $(pwd)"
echo "======================================================"

python add_blur_metric.py \
  --input-csv datasets/jraigs_prepared/labels.csv \
  --output-csv datasets/jraigs_prepared/labels.csv \
  --blur-col blur_score \
  --thumb-size 64

echo "End time : $(date)"
