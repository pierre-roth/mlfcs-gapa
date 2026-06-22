#!/bin/bash
#SBATCH --job-name=mlfcs-as-paper-diag
#SBATCH --account=ls_math
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-26%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
PROJECT_CODE_DIR="${PROJECT_CODE_DIR:-${HOME}/projects/mlfcs-gapa}"
cd "${PROJECT_CODE_DIR}"
source .venv/bin/activate

RUN_ROOT="${RUN_ROOT:-/cluster/work/math/piroth/mlfcs-gapa/runs/extensions/as-paper-strengthening/${SLURM_ARRAY_JOB_ID:-manual}/diagnostics}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/cluster/work/math/piroth/mlfcs-gapa/runs/extensions/as-matched-400k/4094973}"
LATENCIES="${LATENCIES:-1,5,10,20,50}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-101}"

mkdir -p "${RUN_ROOT}"

echo "run_root=${RUN_ROOT}"
echo "checkpoint_root=${CHECKPOINT_ROOT}"
echo "task=${SLURM_ARRAY_TASK_ID:-0}"
echo "latencies=${LATENCIES} device=${DEVICE} seed=${SEED}"
echo "node=${SLURMD_NODENAME:-unknown}"

PYTHONUNBUFFERED=1 python -m mlfcs_gapa.extensions.as_paper_diagnostics \
  --run-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --output-dir "${RUN_ROOT}" \
  --latencies "${LATENCIES}" \
  --seed "${SEED}" \
  --device "${DEVICE}"

echo "diagnostic artifacts for task ${SLURM_ARRAY_TASK_ID:-0}:"
find "${RUN_ROOT}" -mindepth 4 -maxdepth 4 \( -name latency_metrics.csv -o -name teacher_diagnostics.csv \) -print | sort
