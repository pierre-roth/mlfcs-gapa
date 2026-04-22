#!/bin/bash
#SBATCH --job-name=diag_features
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=normal.4h

cd /cluster/home/apetric/mlfcs-gapa/simulation
export RUN_NAME="${RUN_NAME:-diag_features}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/cluster/scratch/apetric/artifacts_anja}"
export DATA_DIR="${DATA_DIR:-/cluster/scratch/apetric/data/anja_norm_fix_overfit_medium_with_leak_hawkes}"
source cluster/euler_common.sh
euler_setup_job

python3 <<'EOF'
"""Diagnose per-day feature distribution drift in dynamic features."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from anja_simulations.data import load_day
from anja_simulations.config import GenerateConfig

cfg = GenerateConfig(mode="medium")
cfg.apply_mode_defaults()
# Override data_dir from env (apply_mode_defaults may have reset it)
cfg.data_dir = os.environ.get("DATA_DIR", cfg.data_dir)
print(f"Using data_dir: {cfg.data_dir}")

# Medium mode: 8 days total. Pick representative days across splits.
days_to_check = [
    ("train_d0", "20191101"),
    ("train_d4", "20191107"),
    ("val",      "20191108"),
    ("test_d0",  "20191111"),
    ("test_d1",  "20191112"),
]

print("=" * 80)
print("DYNAMIC FEATURE DRIFT DIAGNOSTIC")
print("=" * 80)

feature_names = [
    "RV_300s", "RV_600s", "RV_1800s",
    "RSI_300s", "RSI_600s", "RSI_1800s",
    "OSI_mkt_vol_10s", "OSI_mkt_n_10s", "OSI_lim_vol_10s", "OSI_lim_n_10s", "OSI_wd_vol_10s", "OSI_wd_n_10s",
    "OSI_mkt_vol_60s", "OSI_mkt_n_60s", "OSI_lim_vol_60s", "OSI_lim_n_60s", "OSI_wd_vol_60s", "OSI_wd_n_60s",
    "OSI_mkt_vol_300s", "OSI_mkt_n_300s", "OSI_lim_vol_300s", "OSI_lim_n_300s", "OSI_wd_vol_300s", "OSI_wd_n_300s",
]

stats_by_day = {}
for label, day in days_to_check:
    try:
        d = load_day("000001", day, cfg)
    except Exception as e:
        print(f"{label} ({day}): FAILED to load: {e}")
        continue
    dyn = d.dynamic  # shape (n_events, 24)
    event_count = len(d.midprice)
    stats_by_day[label] = {
        "day": day,
        "events": event_count,
        "means": dyn.mean(axis=0),
        "stds": dyn.std(axis=0),
        "mins": dyn.min(axis=0),
        "maxs": dyn.max(axis=0),
    }
    print(f"\n{label} ({day}): {event_count} events")

# Print RV comparison (most important)
print("\n" + "=" * 80)
print("RV (Realized Volatility) — check if scale varies with event density:")
print("=" * 80)
print(f"{'day':<12} {'events':>8} {'RV_300s_mean':>14} {'RV_300s_max':>14} {'RV_1800s_mean':>14}")
for label, s in stats_by_day.items():
    print(f"{label:<12} {s['events']:>8} {s['means'][0]:>14.6e} {s['maxs'][0]:>14.6e} {s['means'][2]:>14.6e}")

# Print RSI comparison (should be stable)
print("\n" + "=" * 80)
print("RSI — should be bounded [0,1] and stable across days:")
print("=" * 80)
print(f"{'day':<12} {'RSI_300s_mean':>14} {'RSI_300s_std':>14} {'RSI_1800s_mean':>14}")
for label, s in stats_by_day.items():
    print(f"{label:<12} {s['means'][3]:>14.4f} {s['stds'][3]:>14.4f} {s['means'][5]:>14.4f}")

# Print OSI comparison (should be bounded [-1,+1])
print("\n" + "=" * 80)
print("OSI market volume imbalance — should be stable [-1,+1]:")
print("=" * 80)
print(f"{'day':<12} {'OSI_10s_mean':>14} {'OSI_10s_std':>14} {'OSI_300s_mean':>14}")
for label, s in stats_by_day.items():
    print(f"{label:<12} {s['means'][6]:>14.4f} {s['stds'][6]:>14.4f} {s['means'][18]:>14.4f}")

# Compute coefficient of variation across days for each feature
print("\n" + "=" * 80)
print("FEATURE DRIFT: Coefficient of variation of (day_mean) across days")
print("(Higher = more drift = more per-day artifact)")
print("=" * 80)
if len(stats_by_day) < 2:
    print(f"Not enough days loaded successfully (got {len(stats_by_day)}). Skipping drift analysis.")
else:
    all_means = np.array([s["means"] for s in stats_by_day.values()])
    grand_mean = all_means.mean(axis=0)
    grand_std = all_means.std(axis=0)
    cv = np.where(np.abs(grand_mean) > 1e-8, grand_std / np.abs(grand_mean), 0.0)
    for i, name in enumerate(feature_names):
        flag = "CRITICAL" if cv[i] > 0.5 else ("HIGH" if cv[i] > 0.2 else "OK")
        print(f"  {name:<30} cv={cv[i]:>8.3f} [{flag}]")

EOF
