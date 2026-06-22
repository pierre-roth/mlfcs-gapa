#!/bin/bash
#SBATCH --job-name=mlfcs-as-guided
#SBATCH --account=ls_math
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --array=0-7%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
PROJECT_CODE_DIR="${PROJECT_CODE_DIR:-${HOME}/projects/mlfcs-gapa}"
cd "${PROJECT_CODE_DIR}"
source .venv/bin/activate

RUN_ROOT="${RUN_ROOT:-/cluster/work/math/piroth/mlfcs-gapa/runs/extensions/as-guided/${SLURM_ARRAY_JOB_ID:-manual}}"
mkdir -p "${RUN_ROOT}"

CHECKPOINT="${CHECKPOINT:-/cluster/work/math/piroth/mlfcs-gapa/runs/full-replication/3272020/table_i_pretraining/attn_lob_pretrain_model.pt}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
AGENT_SEEDS="${AGENT_SEEDS:-3}"
BC_SAMPLES="${BC_SAMPLES:-20000}"
BC_EPOCHS="${BC_EPOCHS:-3}"
N_ENVS="${N_ENVS:-8}"
DEVICE="${DEVICE:-cuda}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) VARIANT="paper_cppo"; LABEL="paper_cppo_rerun"; EXTRA=() ;;
  1) VARIANT="bc_warm_start"; LABEL="bc_as_e3_n20k"; EXTRA=("--bc-samples" "${BC_SAMPLES}" "--bc-epochs" "${BC_EPOCHS}") ;;
  2) VARIANT="soft_as"; LABEL="soft_as_0p01"; EXTRA=("--soft-penalty" "0.01") ;;
  3) VARIANT="soft_as"; LABEL="soft_as_0p10"; EXTRA=("--soft-penalty" "0.10") ;;
  4) VARIANT="soft_as"; LABEL="soft_as_1p00"; EXTRA=("--soft-penalty" "1.00") ;;
  5) VARIANT="hard_as"; LABEL="hard_as_w0p05"; EXTRA=("--hard-window-bias" "0.05" "--hard-window-spread" "0.05") ;;
  6) VARIANT="hard_as"; LABEL="hard_as_w0p10"; EXTRA=("--hard-window-bias" "0.10" "--hard-window-spread" "0.10") ;;
  7) VARIANT="hard_as"; LABEL="hard_as_w0p20"; EXTRA=("--hard-window-bias" "0.20" "--hard-window-spread" "0.20") ;;
  *) echo "unexpected array task id ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

echo "run_root=${RUN_ROOT}"
echo "task=${SLURM_ARRAY_TASK_ID:-0} variant=${VARIANT} label=${LABEL}"
echo "timesteps=${TOTAL_TIMESTEPS} agent_seeds=${AGENT_SEEDS} n_envs=${N_ENVS} device=${DEVICE}"
echo "checkpoint=${CHECKPOINT}"

python -m mlfcs_gapa.extensions.as_guided_panel \
  --output-dir "${RUN_ROOT}" \
  --variant "${VARIANT}" \
  --label "${LABEL}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --agent-seeds "${AGENT_SEEDS}" \
  --n-envs "${N_ENVS}" \
  --encoder-checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  "${EXTRA[@]}"

echo "extension artifacts for ${LABEL}:"
find "${RUN_ROOT}/${LABEL}" -maxdepth 2 -type f | sort
