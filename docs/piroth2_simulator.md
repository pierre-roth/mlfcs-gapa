# piroth2 Simulator

This branch is focused only on generating synthetic limit-order-book data that is usable for the paper-style market-making project.

## Design Goals

- Generate event-by-event top-10 LOB data in the same file format expected by the eventual RL pipeline.
- Make 2000-event windows meaningfully dynamic, because the paper treats 2000 events as roughly 3-5 minutes of market activity.
- Keep the simulator agent-based and interpretable:
  - competing market makers
  - liquidity providers
  - noise takers
  - informed takers
- Keep generation on demand.
  - Days are generated deterministically from `(seed, symbol, day)`.
  - Nothing requires pre-generating the full dataset.
- Leave room for future extensions such as explicit market impact.

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

## Export Format

For each generated day, the simulator can write:

- `ask.csv`
- `bid.csv`
- `price.csv`
- `trades.csv`
- `msg.csv`
- `latent.csv`

The first five are the compatibility targets for the eventual market-making pipeline.

## What Is Not Implemented Yet

- RL training on this branch
- Attn-LOB preprocessing and labeling
- explicit impact of the agent under test on future synthetic order flow
- a full ABIDES-style kernel

The current implementation uses ABIDES as architectural inspiration, not as an embedded dependency.
