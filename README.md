# Market Making with Deep Reinforcement Learning from Limit Order Books

This repository is an attempt at a recreation and possible extention of the work, "Market Making with Deep Reinforcement Learning from Limit Order Books", which was accepted by IJCNN'23.

Team members: 
- **G**hali Berbich
- **A**nja Petric
- **P**ierre Roth
- **A**mine Benjelloun Touimi

## Environment

This project is `uv`-first.

Set up the local environment with:

```bash
uv sync --python 3.12
```

Run project or preprocessing commands with `uv run`, for example:

```bash
uv run python -m preprocessing.databento convert \
  --raw-root data/raw \
  --output-root data/processed \
  --symbol AAPL
```

Run the PyTorch replication pipeline with:

```bash
uv run python -m lobmm.run_suite --mode smoke
```

Useful entrypoints:

```bash
uv run python -m lobmm.pretrain --mode smoke
uv run python -m lobmm.train_rl --mode smoke --algorithm ppo
uv run python -m lobmm.evaluate --mode smoke
uv run python -m lobmm.report --mode smoke
```

## Data Layout

The repository uses the following data directories:

- `data/raw`: raw downloaded market data
- `data/processed`: converted project inputs used by the training code
- `data/validation`: validation and analysis outputs
- `data/sample`: a tracked sample dataset for testing without the full raw data

The active replication code reads directly from `data/processed/<symbol>/<day>/`
using the Databento-derived `ask.csv`, `bid.csv`, `price.csv`, `msg.csv`, and
`trades.csv` files.

See [preprocessing/README.md](/Users/piroth/Documents/projects/mlfcs-gapa/preprocessing/README.md) for Databento conversion and validation commands.
