#!/bin/bash
#SBATCH --job-name=anja-cal-full
#SBATCH --account=3dv
#SBATCH --constraint=2080ti
#SBATCH --time=12:00:00
#SBATCH --output=/home/gberbich/simulation/cluster/logs/ghali_calibrated_full_signal_%j.out
#SBATCH --error=/home/gberbich/simulation/cluster/logs/ghali_calibrated_full_signal_%j.err

set -euo pipefail

cd /home/gberbich/simulation
mkdir -p cluster/logs

source /work/scratch/gberbich/mlfcs/.venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/gberbich/simulation:${PYTHONPATH:-}"

python run.py suite \
  --set mode=full \
  --set run_name=ghali_calibrated_full_signal \
  --set output_root=/work/scratch/gberbich/mlfcs_anja/artifacts \
  --set data_dir=/work/scratch/gberbich/mlfcs_anja/data/ghali_calibrated_full \
  --set symbols='["000001"]' \
  --set device=cuda \
  --set pretrain_label_source=signal \
  --set pretrain_signal_threshold=0.1 \
  --set backbone_trainable=false
