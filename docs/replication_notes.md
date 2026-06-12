# Replication Notes

Current status: 2026-06-11.

## Scope

The project goal is a straight replication of the paper's method on synthetic
data. The only planned substitute is the data source: the proprietary Shenzhen
market data used by the paper is unavailable, so this repository uses
paper-shaped synthetic limit-order-book replay data.

Do not add extra research variants, new objectives, or convenience deviations
unless they are explicitly marked as future work outside the core replication.

## Paper Constants Kept Fixed

The paper-facing constants live in `src/mlfcs_gapa/paper/constants.py` and are
covered by `tests/test_paper_constants.py`.

- Three paper stock codes: `000001`, `000858`, `002415`.
- November 2019 trading calendar with 21 trading days.
- Train/test split: first 10 trading days for training, remaining 11 for held-out
  evaluation.
- LOB state: 10 levels, 4 features per level, 50-event window.
- Mid-price label horizon: 10 events; label threshold: `alpha = 1e-5`.
- Episode length: 2,000 events.
- Minimum trade unit: 100 shares; inventory cap: 10 lots.
- Continuous quote action: two values in `[0, 1]` mapped to reservation-price
  bias and spread with `max_bias = 0.05`, `max_spread = 0.1`.
- Reward terms follow the paper's dampened PnL, trading PnL, and inventory
  penalty structure.

## Current Replication Pipeline

`mlfcs-gapa run-full-synthetic-replication` is the authoritative synthetic
workflow. It generates a synthetic train/test panel and emits the surfaces used
by the paper:

- Table I: FC-LOB, Conv-LOB, DeepLOB, and Attn-LOB pretraining.
- Table II: C-PPO, D-DQN, Inv-RL, LOB-RL, AS, Random, and Fixed baselines.
- Figure 2: latency sweep. Two figures are written: the full method grid and
  `figure_2_latency_paper.png` restricted to the paper's five panels
  (C-PPO, D-DQN, AS, Random, Fixed; Fixed is the level-1 strategy).
- Table III: runtime benchmark rows.
- Table IV: C-PPO and D-DQN ablations.
- Figure 3: attention from the trained C-PPO encoder on two stable and two
  rapidly changing held-out windows, selected by in-window mid-price
  volatility to mirror the paper's (a)/(b) split.
- Figure 4: held-out decision trace from C-PPO.

The supervised Attn-LOB checkpoint is reused by the RL agents where the paper
uses the pretrained LOB state encoder. Table II and IV evaluation uses held-out
synthetic days, not the training episode stream.

## Synthetic Data Substitute

The synthetic generator is expected to preserve the shape of the paper problem:
canonical LOB columns, stable intraday windows, order-book levels, event-time
windows, mid-price movement labels, and a learnable pressure signal.

Synthetic results should not be compared to the paper's numeric profit or
classification values as if they were the original market data. The correct
claim is methodological replication under a controlled data substitute.

## Known Ambiguities

Some details are not recoverable from the paper and public demo code alone:

- The numeric source data behind Figure 2 is not published.
- The original exchange data and private training checkpoints are unavailable.
- The public demo code and paper prose disagree in places, including parts of
  the pretraining architecture descriptions.
- The paper says D-DQN has 8 discrete actions; the implementation follows that
  8-action interpretation.
- The paper equations use a continuous action in `[0, 1]`; the implementation
  keeps that range instead of normalizing to `[-1, 1]`.

The paper does not state RL optimization hyperparameters beyond the learning
rate, so the full replication uses standard, scale-aware choices: PPO collects
rollouts from 8 parallel environments (256 steps each, minibatch = 1/4
rollout, lr 1e-4, gamma 0.99); D-DQN at scale uses a 100k-transition buffer,
batch 64, one gradient update every 4 environment steps, and a 2,000-step
target sync. Smoke-scale runs fall back to proportionally smaller values.

These are treated as reconstruction limits, not opportunities to add new
behavior.

## Tests

The current test suite has 18 small files. It is worth
keeping for now because each file protects a separate replication surface:

- Constants and paper assumptions.
- Synthetic data schema and generation.
- LOB feature construction and pretraining labels.
- Action and reward equations.
- Discrete, Gymnasium, replay, and tabular RL environments.
- Attn-LOB and pretraining model contracts.
- PPO, D-DQN, and pretraining training smoke coverage.
- Report and figure artifact generation.

The tests should be revisited only after the next Euler run shows which paths are
still exercised. Deleting them before that would remove useful protection around
the recent paper-alignment changes.

## Tracking

Weights & Biases tracking is available at the experiment-entrypoint level, not
inside the paper environment/model logic. This keeps tracking separate from the
replication method.

Defaults:

- entity/team: `piroth-ethz`
- project: `mm-drl-lob`

Use `--wandb` on supported CLI commands to log configuration, final metrics, and
selected artifacts. On Euler, the job scripts accept `WANDB_ENABLED=true` and use
the same defaults.

## Running

For a small local check, use targeted tests only. Do not run the full training
pipeline locally unless intentionally doing compute work.

`scripts/euler/` contains exactly four scripts:

- `setup_venv.sh` — one-time environment setup.
- `wandb_env.sh` — sourced by job scripts; builds W&B CLI flags from
  `WANDB_ENABLED` and friends.
- `test_pipeline_cpu.sh` — small CPU job: unit tests plus a miniature
  `run-full-synthetic-replication` through every stage, with artifact
  assertions. Run this before the full job after any code change.
- `full_replication_gpu.sh` — the entire replication as one GPU job.

```bash
cd "$HOME/projects/mlfcs-gapa"
bash scripts/euler/setup_venv.sh                       # once
sbatch scripts/euler/test_pipeline_cpu.sh              # ~30 min sanity check
WANDB_ENABLED=true sbatch scripts/euler/full_replication_gpu.sh
```

The full job runs `mlfcs-gapa run-full-synthetic-replication` with
paper-faithful constants; the synthetic-data calibration knobs are
environment variables in the script. Defaults: 10,000 events/day (~7,600
stable-window events per stock/day), 200,000 agent timesteps (~2.6 passes
over the train panel), 5 pretraining epochs, PPO collecting from 8 parallel
environments. Conv-LOB's 1024-event windows are capped in-code at 20,000
events because materializing them over the full panel costs >10 GB. The job
is sized well inside its 24-hour limit.

On Euler, code should live under:

```bash
$HOME/projects/mlfcs-gapa
```

Durable large data is kept under:

```bash
/cluster/work/math/piroth/mlfcs-gapa
```

Use `docs/euler_cluster_notes.md` for the current storage policy and cleanup
state.
