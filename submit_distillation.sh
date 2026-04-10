#!/bin/bash
# submit_distillation.sh
#
# Submits:
#   1. Three teacher fine-tuning jobs (in parallel)
#   2. Distillation student job (after all teachers succeed)
#
# Usage: bash submit_distillation.sh

set -e

cd "$(dirname "$0")"

echo "Submitting teacher fine-tuning jobs..."

JID_EVA=$(sbatch --parsable run_teacher_eva02.batch)
JID_LARGE=$(sbatch --parsable run_teacher_dinov3_large.batch)
JID_HUGE=$(sbatch --parsable run_teacher_dinov3_huge.batch)

echo "  Teacher EVA02-Large    : job $JID_EVA"
echo "  Teacher DINOv3-Large   : job $JID_LARGE"
echo "  Teacher DINOv3-Huge+   : job $JID_HUGE"

echo ""
echo "Submitting distillation student (depends on $JID_EVA:$JID_LARGE:$JID_HUGE)..."

JID_STUDENT=$(sbatch --parsable \
    --dependency=afterok:${JID_EVA}:${JID_LARGE}:${JID_HUGE} \
    run_distillation_student.batch)

echo "  Distillation student   : job $JID_STUDENT"
echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/teacher_eva02_${JID_EVA}.out"
echo "  tail -f logs/distill_student_${JID_STUDENT}.out"
