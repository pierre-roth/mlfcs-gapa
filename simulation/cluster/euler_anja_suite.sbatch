#!/bin/bash
#SBATCH --job-name=anja-suite
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

export EULER_RUNNER_SCRIPT="${EULER_RUNNER_SCRIPT:-cluster/euler_run_anja_simulations.py}"
export DEVICE="${DEVICE:-cuda}"
export MODE="${MODE:-full}"
export SYMBOLS="${SYMBOLS:-000001}"
export RUN_NAME="${RUN_NAME:-anja_suite}"
export DATA_DIR="${DATA_DIR:-/cluster/scratch/${USER}/data/${RUN_NAME}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/scratch/${USER}/artifacts_anja}"
mkdir -p "${DATA_DIR}" "${OUTPUT_ROOT}"

euler_setup_job
euler_run_kind suite
