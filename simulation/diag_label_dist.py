#!/usr/bin/env python
"""Diagnostic: measure label class distribution at various pretrain_alpha values."""

from pathlib import Path
import numpy as np
from anja_simulations.data import load_day, _paper_labels
from anja_simulations.config import GenerateConfig

config = GenerateConfig(mode='medium')
config.apply_mode_defaults()

# Load one training day
print('Loading 20191101 training data...')
day_data = load_day(config, '000001', '20191101')
prices = day_data.prices

# Test label distribution at various alpha values
print('\nLabel class distribution at various pretrain_alpha thresholds:')
print('(Target: ~33% each for balanced 3-class problem)\n')

results = []
for alpha in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
    labels = _paper_labels(prices['mid'], horizon=10, threshold=alpha)
    counts = np.bincount(labels)
    pct = 100 * counts / len(labels)
    line = f'alpha={alpha:0.0e}: up={pct[0]:5.1f}% flat={pct[1]:5.1f}% down={pct[2]:5.1f}%  (counts: {counts})'
    print(line)
    results.append((alpha, pct[0], pct[1], pct[2]))

# Find closest to balanced (33/33/33)
print('\n--- Analysis ---')
for alpha, up, flat, down in results:
    imbalance = max(abs(up - 33.3), abs(flat - 33.3), abs(down - 33.3))
    print(f'alpha={alpha:0.0e}: max deviation from 33.3% = {imbalance:.1f}%')
