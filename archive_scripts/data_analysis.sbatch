#!/bin/bash
#
#SBATCH --job-name=retfound_glaucoma
#SBATCH --output=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab/data_analysis%j.out
#SBATCH --error=/raid/home/students/hawky_luc/P-le-Projet-Zenkolab/outputs/eda/data_analysis%j.out
#SBATCH --partition=prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:0:0

source /raid/home/students/hawky_luc/P-le-Projet-Zenkolab/.venv/bin/activate

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "  Start time : $(date)"
echo "======================================================"

# Délai pour éviter une race condition si plusieurs jobs démarrent simultanément
# (MLflow tente de créer la DB en même temps → conflit SQLite)

python data_analysis.py

echo "End time : $(date)"