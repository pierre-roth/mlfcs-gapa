#!/bin/bash
#SBATCH --job-name=diag_label
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=normal.4h
#SBATCH --output=logs/diag_label-%J.out
#SBATCH --error=logs/diag_label-%J.err

cd /cluster/home/apetric/mlfcs-gapa/simulation
export RUN_NAME="${RUN_NAME:-diag_label}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/scratch/apetric/artifacts_anja}"
export DATA_DIR="${DATA_DIR:-/cluster/scratch/apetric/data/anja_full_normfix_20260423_000941}"
source cluster/euler_common.sh
euler_setup_job

python3 <<'EOF'
import os
from anja_simulations.data import load_day, _paper_labels
from anja_simulations.config import GenerateConfig
import numpy as np

print("=" * 70)
print("DIAGNOSTIC: Label Class Distribution at Various Alpha Values")
print("=" * 70)

cfg = GenerateConfig(mode="medium")
cfg.apply_mode_defaults()
# Override data_dir from env after apply_mode_defaults (which may reset it)
if "DATA_DIR" in os.environ:
    cfg.data_dir = os.environ["DATA_DIR"]
print(f"Using data_dir: {cfg.data_dir}")

print("\nLoading 20191101 training data...")
day = load_day("000001", "20191101", cfg)

print("\nLabel class distribution at various pretrain_alpha thresholds:")
print("(Target: ~33% each for balanced 3-class problem)\n")

results = []
for alpha in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
    labels = _paper_labels(day.midprice, horizon=10, threshold=alpha)
    # Filter NaN (at window boundaries) and cast to int64 for bincount
    valid = labels[~np.isnan(labels)].astype(np.int64)
    counts = np.bincount(valid, minlength=3)
    total = len(valid)
    pct = 100 * counts / max(total, 1)
    line = f"alpha={alpha:0.0e}: up={pct[0]:5.1f}% flat={pct[1]:5.1f}% down={pct[2]:5.1f}% (counts: {counts.tolist()}, n_valid={total})"
    print(line)
    results.append((alpha, pct[0], pct[1], pct[2]))

print("\n" + "=" * 70)
print("ANALYSIS: Deviation from balanced 33/33/33 distribution")
print("=" * 70 + "\n")
for alpha, up, flat, down in results:
    imbalance = max(abs(up - 33.3), abs(flat - 33.3), abs(down - 33.3))
    quality = "GOOD" if imbalance < 5 else ("OK" if imbalance < 10 else "IMBALANCED")
    print(f"alpha={alpha:0.0e}: max deviation = {imbalance:5.1f}%  [{quality}]")

print("\nRECOMMENDATION:")
best_alpha = min(results, key=lambda x: max(abs(x[1]-33.3), abs(x[2]-33.3), abs(x[3]-33.3)))
best_imbalance = max(abs(best_alpha[1]-33.3), abs(best_alpha[2]-33.3), abs(best_alpha[3]-33.3))
print(f"  Use PRETRAIN_ALPHA={best_alpha[0]:.0e} (imbalance={best_imbalance:.1f}%)")
EOF
