#!/bin/bash
#
#SBATCH --job-name=train_origa
#SBATCH --output=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab/outputs/train_origa_%j.out
#SBATCH --error=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab/outputs/train_origa_%j.err
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

mkdir -p "$PROJECT_DIR/outputs"

# ORIGA dataset (split train/val stored in labels_train_val.csv: split A/B)
ORIGA_DIR="$PROJECT_DIR/datasets/sshikamaru__glaucoma-detection/ORIGA/ORIGA"
ORIGA_CSV="$ORIGA_DIR/labels_train_val.csv"
ORIGA_IMAGES="$ORIGA_DIR/Images"

if [[ ! -f "$ORIGA_CSV" ]]; then
	echo "ERROR: Missing ORIGA CSV: $ORIGA_CSV"
	exit 1
fi

if [[ ! -d "$ORIGA_IMAGES" ]]; then
	echo "ERROR: Missing ORIGA images dir: $ORIGA_IMAGES"
	exit 1
fi

export ORIGA_DATA_DIR="$ORIGA_DIR"
export ORIGA_CSV_PATH="$ORIGA_CSV"
export ORIGA_IMAGES_DIR="$ORIGA_IMAGES"
export ORIGA_TRAIN_SPLIT="A"
export ORIGA_VAL_SPLIT="B"
export PYTHONUNBUFFERED=1

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPU(s)     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Start time : $(date)"
echo "  CWD        : $(pwd)"
echo "  ORIGA CSV  : $ORIGA_CSV_PATH"
echo "  ORIGA IMG  : $ORIGA_IMAGES_DIR"
echo "  Splits     : train=$ORIGA_TRAIN_SPLIT, val=$ORIGA_VAL_SPLIT"
echo "======================================================"

srun python train_origa.py

echo "End time : $(date)"
