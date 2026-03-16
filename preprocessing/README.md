# Databento Preprocessing

This folder contains standalone tooling for converting raw Databento
downloads into the `ask.csv`, `bid.csv`, `price.csv`, `msg.csv`, and
`trades.csv` files expected by the project.

The preprocessing pipeline is intentionally separate from the core
training code so raw-data handling, validation, and conversion logic can
evolve independently.

Typical usage:

```bash
uv sync
uv run python -m preprocessing.databento decompress --raw-root data/raw
uv run python -m preprocessing.databento convert --raw-root data/raw --output-root data/processed --symbol GOOGL
uv run python -m preprocessing.databento validate --raw-root data/raw --processed-root data/processed --validation-root data/validation --symbol GOOGL
```
