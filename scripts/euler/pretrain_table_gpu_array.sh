#!/bin/bash
#SBATCH --job-name=mlfcs-pretrain
#SBATCH --account=ls_math
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --array=0-3%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
PROJECT_CODE_DIR="${PROJECT_CODE_DIR:-${HOME}/projects/mlfcs-gapa}"
cd "${PROJECT_CODE_DIR}"
source .venv/bin/activate
source scripts/euler/wandb_env.sh

MODELS=("FC-LOB" "Conv-LOB" "DeepLOB" "Attn-LOB")
MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"
EVENTS="${EVENTS:-6000}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-101}"
RUN_ROOT="${RUN_ROOT:-/cluster/work/math/piroth/mlfcs-gapa/runs/pretrain-table-gpu/${SLURM_ARRAY_JOB_ID}}"

echo "model=${MODEL_NAME}"
echo "gpu_concurrency_cap=4"
echo "events=${EVENTS} epochs=${EPOCHS} batch_size=${BATCH_SIZE} seed=${SEED}"

mlfcs-gapa pretrain-synthetic \
  --model-name "${MODEL_NAME}" \
  --output-dir "${RUN_ROOT}/${MODEL_NAME}" \
  --events "${EVENTS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --seed "${SEED}" \
  "${WANDB_ARGS[@]}"
