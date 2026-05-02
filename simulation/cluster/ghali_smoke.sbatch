#!/bin/bash
#SBATCH --job-name=anja-smoke
#SBATCH --account=3dv
#SBATCH --constraint=2080ti
#SBATCH --time=00:30:00
#SBATCH --output=/home/gberbich/simulation/cluster/logs/ghali_smoke_%j.out
#SBATCH --error=/home/gberbich/simulation/cluster/logs/ghali_smoke_%j.err

set -euo pipefail

cd /home/gberbich/simulation
mkdir -p cluster/logs

source /work/scratch/gberbich/mlfcs/.venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/gberbich/simulation:${PYTHONPATH:-}"

python run.py suite \
  --set mode=smoke \
  --set run_name=ghali_cal_smoke \
  --set output_root=/work/scratch/gberbich/mlfcs_anja/artifacts \
  --set data_dir=/work/scratch/gberbich/mlfcs_anja/data/ghali_cal_smoke \
  --set symbols='["000001"]' \
  --set device=cuda
