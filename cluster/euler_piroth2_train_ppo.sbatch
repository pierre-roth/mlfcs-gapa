#!/bin/bash
#SBATCH --job-name=piroth2-ppo
#SBATCH --account=ls_math
#SBATCH --partition=gpu.24h
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --tmp=8G
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export REPO_ROOT
source "${REPO_ROOT}/cluster/euler_common.sh"

export EULER_RUNNER_SCRIPT="${EULER_RUNNER_SCRIPT:-cluster/euler_run_piroth2.py}"
export DEVICE="${DEVICE:-cuda}"
export MODE="${MODE:-medium}"
export SYMBOL="${SYMBOL:-000001}"
export RUN_NAME="${RUN_NAME:-piroth2_replication}"
export DATA_DIR="${DATA_DIR:-/cluster/work/math/${USER}/mlfcs-gapa/data/${RUN_NAME}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/project/math/${USER}/mlfcs-gapa/artifacts_piroth2}"
mkdir -p "${DATA_DIR}" "${OUTPUT_ROOT}"

euler_setup_job
extra_args=()
if [[ -n "${CHECKPOINT:-}" ]]; then
    extra_args+=(--checkpoint "${CHECKPOINT}")
fi
euler_run_kind train-ppo "${extra_args[@]}"
