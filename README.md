# mlfcs-gapa

Code accompanying the AS-regularized market-making project. The repository contains
a synthetic-data replication of the Attn-LOB/C-PPO market-making pipeline and a
separate extension package for Avellaneda--Stoikov (AS) guided policies.

## Setup

Requires Python 3.12 and `uv`.

```bash
uv sync
uv run pytest
```

## Layout

- `src/mlfcs_gapa/data/` - synthetic LOB generation and feature preparation.
- `src/mlfcs_gapa/env/`, `src/mlfcs_gapa/training/`, `src/mlfcs_gapa/models/` -
  replication environments, baselines, PPO, DQN, and Attn-LOB components.
- `src/mlfcs_gapa/extensions/` - AS behavioural cloning, soft regularization,
  hard action-window constraints, and matched extension sweeps.
- `scripts/euler/` - ETH Euler smoke tests and GPU array jobs.
- `docs/` - replication and cluster notes.
- `tests/` - unit and smoke tests.

The original Shenzhen LOB data are proprietary and are not redistributed here.
Results should be read as a controlled synthetic replication and extension, not
as the original paper's proprietary-data numbers.
