#!/bin/bash
#SBATCH --job-name=lobmm-evaluate
#SBATCH --account=ls_math
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=6G
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster/euler_common.sh
source "${SCRIPT_DIR}/euler_common.sh"

export DEVICE="${DEVICE:-cpu}"
euler_setup_job
euler_run_kind evaluate
