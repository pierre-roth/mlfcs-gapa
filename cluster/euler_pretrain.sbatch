#!/bin/bash
#SBATCH --job-name=lobmm-pretrain
#SBATCH --account=ls_math
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --tmp=8G
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster/euler_common.sh
source "${SCRIPT_DIR}/euler_common.sh"

export DEVICE="${DEVICE:-cuda}"
euler_setup_job
euler_run_kind pretrain
