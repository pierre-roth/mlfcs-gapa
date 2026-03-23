#!/bin/bash
#SBATCH --job-name=lobmm-pretrain
#SBATCH --account=ls_math
#SBATCH --partition=gpu.24h
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --tmp=8G
#SBATCH --signal=B:USR1@300
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export REPO_ROOT
# shellcheck source=cluster/euler_common.sh
source "${REPO_ROOT}/cluster/euler_common.sh"

forward_usr1() {
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        scancel --signal=USR1 "${SLURM_JOB_ID}.0" >/dev/null 2>&1 || true
    fi
}

trap forward_usr1 USR1

export DEVICE="${DEVICE:-cuda}"
euler_setup_job
euler_run_kind pretrain
