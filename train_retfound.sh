#!/bin/bash
#
#SBATCH --job-name=retfound_glaucoma
#SBATCH --output=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/retfound_%j.out
#SBATCH --error=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab/logs/retfound_%j.out
#SBATCH --partition=prod40
#SBATCH --gres=gpu:nvidia_a100_3g.40gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:0:0

set -e
PROJECT=/raid/home/students/goldrajch_dav/projectEYE/P-le-Projet-Zenkolab
source "$PROJECT/.venv/bin/activate"
cd "$PROJECT"
mkdir -p logs checkpoints results

echo "======================================================"
echo "  Model      : RETFound MAE (retinal pretrained)"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "  Start time : $(date)"
echo "======================================================"

uv pip install --quiet timm huggingface_hub scikit-learn matplotlib pandas

# Délai pour éviter une race condition si plusieurs jobs démarrent simultanément
# (MLflow tente de créer la DB en même temps → conflit SQLite)
sleep 10

python -u src/train.py \
    --model        retfound       \
    --data_root    datasets/      \
    --test_dir     datasets/REFUGE2/train  \
    --epochs       30             \
    --batch_size   32             \
    --lr           1e-4           \
    --weight_decay 0.05           \
    --num_workers  8              \
    --patience     7              \
    --freeze_epochs 3             \
    --mlflow                      \
    --mlflow_exp   retfound-vs-vit

echo "End time : $(date)"
