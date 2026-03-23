#!/bin/bash
#SBATCH --job-name=lobmm-report
#SBATCH --account=ls_math
#SBATCH --partition=normal.24h
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=4G
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
export REPO_ROOT
# shellcheck source=cluster/euler_common.sh
source "${REPO_ROOT}/cluster/euler_common.sh"

export DEVICE="${DEVICE:-cpu}"
euler_setup_job
euler_run_kind report
