# Market Making with Deep Reinforcement Learning from Limit Order Books

This repository is for the demonstration of our work, "Market Making with Deep Reinforcement Learning from Limit Order Books",  which has been accepted by IJCNN'23.

## Environment

This project is `uv`-first.

Set up the local environment with:

```bash
uv sync
```

Run project or preprocessing commands with `uv run`, for example:

```bash
uv run python -m preprocessing.databento convert \
  --raw-root data/raw \
  --output-root data/processed \
  --symbol AAPL
```

## Data Layout

The repository uses the following data directories:

- `data/raw`: raw downloaded market data
- `data/processed`: converted project inputs used by the training code
- `data/validation`: validation and analysis outputs
- `data/sample`: a tracked sample dataset for testing without the full raw data

See [preprocessing/README.md](/Users/piroth/Documents/projects/mlfcs-gapa/preprocessing/README.md) for Databento conversion and validation commands.
