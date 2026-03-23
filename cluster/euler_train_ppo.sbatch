#!/bin/bash
#SBATCH --job-name=lobmm-train
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
# shellcheck source=cluster/euler_common.sh
source "${REPO_ROOT}/cluster/euler_common.sh"

export DEVICE="${DEVICE:-cuda}"
export ALGORITHM="${ALGORITHM:-ppo}"
export STATE_MODE="${STATE_MODE:-full}"
euler_setup_job
euler_run_kind train
