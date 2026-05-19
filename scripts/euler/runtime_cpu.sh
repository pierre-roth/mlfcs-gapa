#!/bin/bash
#SBATCH --job-name=mlfcs-runtime
#SBATCH --account=ls_math
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
cd "${SCRATCH:?}/mlfcs-gapa"
source .venv/bin/activate

RUN_ROOT="${RUN_ROOT:-${SCRATCH}/mlfcs-gapa/runs/runtime-cpu/${SLURM_JOB_ID}}"
EVENTS="${EVENTS:-1000}"
EPISODE_EVENTS="${EPISODE_EVENTS:-500}"
TRAIN_TIMESTEPS="${TRAIN_TIMESTEPS:-256}"
SEED="${SEED:-101}"

mlfcs-gapa benchmark-runtime-synthetic \
  --output-path "${RUN_ROOT}/runtime_metrics.csv" \
  --events "${EVENTS}" \
  --episode-events "${EPISODE_EVENTS}" \
  --train-timesteps "${TRAIN_TIMESTEPS}" \
  --device cpu \
  --seed "${SEED}"
