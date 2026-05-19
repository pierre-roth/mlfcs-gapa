"""Read and write canonical LOB datasets."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from mlfcs_gapa.data.schema import LobDataset


def write_lob_dataset(dataset: LobDataset, root: Path) -> Path:
    day_dir = root / dataset.stock / dataset.day
    day_dir.mkdir(parents=True, exist_ok=True)
    dataset.orderbook.write_parquet(day_dir / "orderbook.parquet")
    dataset.messages.write_parquet(day_dir / "messages.parquet")
    dataset.trades.write_parquet(day_dir / "trades.parquet")
    return day_dir


def read_lob_dataset(root: Path, stock: str, day: str) -> LobDataset:
    day_dir = root / stock / day
    return LobDataset(
        stock=stock,
        day=day,
        orderbook=pl.read_parquet(day_dir / "orderbook.parquet"),
        messages=pl.read_parquet(day_dir / "messages.parquet"),
        trades=pl.read_parquet(day_dir / "trades.parquet"),
    )
