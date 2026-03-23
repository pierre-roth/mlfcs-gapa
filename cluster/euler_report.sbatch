#!/bin/bash
#SBATCH --job-name=lobmm-report
#SBATCH --account=ls_math
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=4G
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster/euler_common.sh
source "${SCRIPT_DIR}/euler_common.sh"

export DEVICE="${DEVICE:-cpu}"
euler_setup_job
euler_run_kind report
