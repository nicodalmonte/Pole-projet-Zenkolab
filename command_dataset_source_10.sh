#!/bin/bash
#
#SBATCH --job-name=dataset_source_cnn
#SBATCH --output=/raid/home/students/damon_mat/P-le-Projet-Zenkolab/logs/dataset_source_%j.out
#SBATCH --error=/raid/home/students/damon_mat/P-le-Projet-Zenkolab/logs/dataset_source_%j.out

## Partition: prod10 | prod20 | prod40 | prod80
#SBATCH --partition=prod10

## GPU: nvidia_a100_1g.10gb | nvidia_a100_3g.40gb | nvidia_a100-sxm4-80gb
#SBATCH --gres=gpu:nvidia_a100_1g.10gb:1

## ntasks * cpus-per-task must be in [1 : 4 * nMIG]
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

## RAM
#SBATCH --mem=12G

## Max wall time
#SBATCH --time=24:0:0

PROJECT=/raid/home/students/damon_mat/P-le-Projet-Zenkolab
cd "$PROJECT"

echo "======================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "  Python     : $(uv run python --version)"
echo "  Start time : $(date)"
echo "======================================================"

uv run python -m src.dataset_experiment.train_source_classifier \
  --data_dir data/datasets \
  --batch_size 64 \
  --num_workers 4 \
  --image_size 224 \
  --largest_source_max_ratio 2.0 \
  --max_epochs 30 \
  --precision 16-mixed \
  --devices 1 \
  --experiment_name dataset_source_cnn \
  --checkpoint_dir checkpoints/dataset_source_cnn/job_${SLURM_JOB_ID}
