#!/bin/bash
#SBATCH --job-name=lobmm-suite
#SBATCH --account=ls_math
#SBATCH --time=18:00:00
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
export RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
export RUN_MAIN_AGENTS="${RUN_MAIN_AGENTS:-1}"
export RUN_RL_BASELINES="${RUN_RL_BASELINES:-0}"
export RUN_NON_RL_BASELINES="${RUN_NON_RL_BASELINES:-1}"
export RUN_ABLATIONS="${RUN_ABLATIONS:-0}"
export RUN_LATENCY="${RUN_LATENCY:-0}"
export RUN_REPORT="${RUN_REPORT:-1}"
euler_setup_job
euler_run_kind suite
