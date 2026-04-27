# piroth2 Simulator

This branch is focused on generating synthetic limit-order-book data that is usable for a paper-faithful market-making replication. The data must be interesting at the paper's 2000-event episode scale before PPO/DQN results are meaningful.

## Design Goals

- Generate event-by-event top-10 LOB data in the same file format expected by the paper pipeline.
- Make 2000-event windows meaningfully dynamic, because the paper treats 2000 events as roughly 3-5 minutes of market activity.
- Keep the simulator agent-based and interpretable:
  - competing market makers
  - liquidity providers
  - noise takers
  - informed takers
- Keep generation on demand.
  - Days are generated deterministically from `(seed, symbol, day)`.
  - Nothing requires pre-generating the full dataset.
- Keep the paper-facing export clean:
  - `ask.csv`, `bid.csv`, `price.csv`, `trades.csv`, `msg.csv`
  - `msg.csv` uses aggregate market/limit/withdraw buy/sell columns required by the paper's Order Strength Index.
  - `event_log.csv` and `latent.csv` are simulator diagnostics, not paper inputs.

## Main Components

- [piroth/orderbook.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/orderbook.py)
  - explicit FIFO order queues by price level
  - market-order matching
  - random cancellation
  - top-of-book extraction
- [piroth/simulator.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/simulator.py)
  - latent fair-value process
  - event scheduler
  - agent-type behaviors
  - exportable synthetic day object
- [piroth/baselines.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/baselines.py)
  - fixed quoting baseline
  - Avellaneda-Stoikov baseline
  - paper-style replay/backtest against synthetic events
- [piroth/paper_features.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/paper_features.py)
  - paper LOB normalization
  - pretraining labels
  - realized volatility, RSI, OSI dynamic state
  - agent inventory/time state
- [piroth/paper_env.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/paper_env.py)
  - paper-style latency, trade-timestamp stepping, inventory cap, matching, reward, forced close
- [piroth/models.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/models.py)
  - PyTorch Attn-LOB, pretraining head, C-PPO actor-critic, D-DQN
- [piroth/paper_experiments.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/paper_experiments.py)
  - paper baseline suite
  - latency suite
  - ablation suite
  - full paper suite
- [piroth/visualizer.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/visualizer.py)
  - HTML report for synthetic data quality and baseline artifacts
- [piroth/diagnostics.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/diagnostics.py)
  - generates train/test synthetic days
  - calibrates AS on the train split
  - evaluates AS and fixed baselines on the test split
  - exports summary files and plots
- [piroth/plots.py](/Users/piroth/Documents/projects/mlfcs-gapa/piroth/plots.py)
  - daily midprice overview
  - random 2000-event windows
  - LOB depth heatmap
  - LOB snapshots

## Event Logic

Each event updates the market through:

1. latent fair-value evolution
   - small noise
   - persistent regime drift
   - persistent metaorders
   - occasional shocks
2. one sampled event type
   - noise market order
   - informed market order
   - liquidity add
   - cancel
   - competing market-maker refresh
3. replenishment
   - if the book becomes too sparse or the spread becomes too wide

The key mechanism for meaningful 2000-event windows is the interaction between:
- latent fair-value drift
- informed order flow
- market-maker refresh around fair value
- cancellations and replenishment near the touch

The fair value is now anchored and bounded around the symbol's reference price. This deliberately avoids pathological full-day drift while still allowing local metaorder trends, shocks, queue pressure, and adverse-selection episodes.

## Export Format

For each generated day, the simulator can write:

- `ask.csv`
- `bid.csv`
- `price.csv`
- `trades.csv`
- `msg.csv`
- `latent.csv`
- `event_log.csv`

The first five are the compatibility targets for the paper-style market-making pipeline.

`msg.csv` columns:

- `market_buy_volume`, `market_buy_n`
- `market_sell_volume`, `market_sell_n`
- `limit_buy_volume`, `limit_buy_n`
- `limit_sell_volume`, `limit_sell_n`
- `withdraw_buy_volume`, `withdraw_buy_n`
- `withdraw_sell_volume`, `withdraw_sell_n`

## Visualizer

The diagnostics job writes:

```text
${OUTPUT_ROOT}/${RUN_NAME}/visual_report/index.html
```

The report covers:

- daily midprice vs latent fair value
- sampled 2000-event window moves
- spread and event-return distributions
- rolling order-flow imbalance
- LOB depth heatmaps and snapshots
- synthetic fill-probability curves by quote distance
- paper baseline tables when available

## Cluster Workflow

All execution should happen through the cluster dispatcher:

```bash
RUN_NAME=piroth2_main SYMBOL=000001 MODE=medium cluster/submit_piroth2.sh pipeline
cluster/submit_piroth2.sh diagnostics
KIND=paper-baselines cluster/submit_piroth2.sh evaluate
VALIDATION_SYMBOLS=000001,000858,002415 VALIDATION_SEEDS=7,11,17,23 cluster/submit_piroth2.sh validate-data
KIND=latency-suite cluster/submit_piroth2.sh evaluate
KIND=ablation-suite cluster/submit_piroth2.sh suite
cluster/submit_piroth2.sh suite
```

The `paper-suite` job generates synthetic train/test days, exports inspection days, evaluates paper baselines, pretrains Attn-LOB, trains/evaluates C-PPO and D-DQN, and writes the visual report.

The wrapper supports resource overrides with stage prefixes, for example `PRETRAIN_TIME`, `TRAIN_PPO_GPUS`, `TRAIN_DQN_MEM_PER_CPU`, `EVALUATE_PARTITION`, and `SUITE_TIME`.

## Multi-Seed Validation

The validation job writes:

- `index.html`
- `synthetic_validation_cases.csv`
- `synthetic_validation_by_symbol.csv`
- `synthetic_validation_summary.json`
- selected per-case visual reports under `cases/<symbol>_seed<seed>/visual_report/index.html`

The quality gate requires no flags, score at least 90, plausible spread and
trade-density ranges, non-flat 2000-event windows, persistent order flow, and a
fill curve that decays with quote distance.

## Data Quality

Diagnostics include `synthetic_quality` in `summary.json` and the visual report. It scores:

- trade density
- spread distribution
- event return volatility
- 2000-event window movement
- market-order-flow persistence
- top-of-book depth imbalance variation

Current multi-seed validation on Euler, generated with `MODE=smoke`, three
symbols, four seeds per symbol, two days per case, and
`VALIDATION_EVENTS_PER_DAY=12000`:

```text
run_name: piroth2_validation_final_20260424_181659
cases: 12
pass_rate: 100%
score_mean: 98.71
score_min: 93.78
flags_total: 0
```

## What Is Not Implemented Yet

- explicit impact of the agent under test on future synthetic order flow
- a full ABIDES-style kernel

The current implementation uses ABIDES as architectural inspiration, not as an embedded dependency.
