#!/bin/bash
#SBATCH --job-name=mlfcs-hard-phases
#SBATCH --account=ls_math
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load stack/2024-06 python/3.12.8
cd "${HOME}/projects/mlfcs-gapa"
source .venv/bin/activate

PYTHONUNBUFFERED=1 python scripts/euler/debug_hard_as_phases.py \
  --window "${WINDOW:-0.10}" \
  --device "${DEVICE:-cuda}" \
  --timesteps "${TIMESTEPS:-512}" \
  --n-envs "${N_ENVS:-2}" \
  --events-per-day "${EVENTS_PER_DAY:-2000}" \
  --train-days "${TRAIN_DAYS:-2}"
