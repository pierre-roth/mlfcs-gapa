#!/bin/bash
#SBATCH --job-name=piroth-sweep
#SBATCH --account=ls_math
#SBATCH --partition=gpu.24h
#SBATCH --time=12:00:00
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

export EULER_RUNNER_SCRIPT="${EULER_RUNNER_SCRIPT:-cluster/euler_run_piroth.py}"
export DEVICE="${DEVICE:-cuda}"
export MODE="${MODE:-medium}"
export SYMBOLS="${SYMBOLS:-000001}"
export RUN_NAME="${RUN_NAME:-piroth_sweep}"
export SWEEP_NAME="${SWEEP_NAME:-piroth_sweep}"
export DATA_DIR="${DATA_DIR:-/cluster/work/math/${USER}/mlfcs-gapa/data/${RUN_NAME}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/project/math/${USER}/mlfcs-gapa/artifacts_piroth}"
mkdir -p "${DATA_DIR}" "${OUTPUT_ROOT}"

euler_setup_job
euler_run_kind sweep
