#!/bin/bash
#SBATCH --job-name=mlfcs-lat-base
#SBATCH --account=ls_math
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
PROJECT_CODE_DIR="${PROJECT_CODE_DIR:-${HOME}/projects/mlfcs-gapa}"
cd "${PROJECT_CODE_DIR}"
source .venv/bin/activate
source scripts/euler/wandb_env.sh

RUN_ROOT="${RUN_ROOT:-/cluster/work/math/piroth/mlfcs-gapa/runs/synthetic-latency-full/${SLURM_JOB_ID}}"
EVENTS="${EVENTS:-6000}"
EPISODE_EVENTS="${EPISODE_EVENTS:-2000}"
DAYS="${DAYS:-1}"
TABULAR_EPISODES="${TABULAR_EPISODES:-10}"
SEED="${SEED:-101}"
LATENCIES="${LATENCIES:-1,5,10,20,50,100}"

echo "latency_baselines_cpu=true"
echo "latencies=${LATENCIES}"
echo "events=${EVENTS} episode_events=${EPISODE_EVENTS} days=${DAYS}"
echo "tabular_episodes=${TABULAR_EPISODES} seed=${SEED}"
echo "run_root=${RUN_ROOT}"

mlfcs-gapa run-synthetic-latency-baselines \
  --output-dir "${RUN_ROOT}" \
  --latencies "${LATENCIES}" \
  --days "${DAYS}" \
  --events-per-day "${EVENTS}" \
  --episode-events "${EPISODE_EVENTS}" \
  --tabular-episodes "${TABULAR_EPISODES}" \
  --seed "${SEED}" \
  "${WANDB_ARGS[@]}"
