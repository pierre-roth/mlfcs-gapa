#!/bin/bash
#SBATCH --job-name=mlfcs-as32
#SBATCH --account=ls_math
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --array=0-31%8
#SBATCH --exclude=eu-lo-g2-022,eu-lo-g3-017
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
PROJECT_CODE_DIR="${PROJECT_CODE_DIR:-${HOME}/projects/mlfcs-gapa}"
cd "${PROJECT_CODE_DIR}"
source .venv/bin/activate

RUN_ROOT="${RUN_ROOT:-/cluster/work/math/piroth/mlfcs-gapa/runs/extensions/as-guided-32/${SLURM_ARRAY_JOB_ID:-manual}}"
CHECKPOINT="${CHECKPOINT:-/cluster/work/math/piroth/mlfcs-gapa/runs/full-replication/3272020/table_i_pretraining/attn_lob_pretrain_model.pt}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
AGENT_SEEDS="${AGENT_SEEDS:-3}"
N_ENVS="${N_ENVS:-8}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${RUN_ROOT}"

echo "run_root=${RUN_ROOT}"
echo "task=${SLURM_ARRAY_TASK_ID:-0}"
echo "timesteps=${TOTAL_TIMESTEPS} agent_seeds=${AGENT_SEEDS} n_envs=${N_ENVS} device=${DEVICE}"
echo "checkpoint=${CHECKPOINT}"

PYTHONUNBUFFERED=1 python -m mlfcs_gapa.extensions.as_guided_sweep \
  --run-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --output-dir "${RUN_ROOT}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --agent-seeds "${AGENT_SEEDS}" \
  --n-envs "${N_ENVS}" \
  --encoder-checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}"

echo "sweep artifacts:"
find "${RUN_ROOT}" -mindepth 2 -maxdepth 2 -name extension_metrics.csv -print | sort
