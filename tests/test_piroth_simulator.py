from __future__ import annotations

from pathlib import Path

import pandas as pd

from piroth.config import GenerateConfig
from piroth.simulator import generate_dataset


def test_simulator_is_deterministic(tmp_path: Path) -> None:
    cfg_a = GenerateConfig(mode="smoke", data_dir=str(tmp_path / "a"), symbols=["000001"], seed=5).apply_mode_defaults()
    cfg_b = GenerateConfig(mode="smoke", data_dir=str(tmp_path / "b"), symbols=["000001"], seed=5).apply_mode_defaults()
    generate_dataset(cfg_a)
    generate_dataset(cfg_b)
    price_a = pd.read_csv(tmp_path / "a" / "000001" / "20191101" / "price.csv")
    price_b = pd.read_csv(tmp_path / "b" / "000001" / "20191101" / "price.csv")
    assert price_a.equals(price_b)


def test_agent_based_book_is_ordered_and_emits_metadata(tmp_path: Path) -> None:
    cfg = GenerateConfig(mode="smoke", data_dir=str(tmp_path / "sim"), symbols=["000001"], seed=13).apply_mode_defaults()
    generate_dataset(cfg)
    root = tmp_path / "sim" / "000001" / "20191101"
    ask = pd.read_csv(root / "ask.csv")
    bid = pd.read_csv(root / "bid.csv")
    latent = pd.read_csv(root / "latent.csv")
    trades = pd.read_csv(root / "trades.csv")
    assert (ask["ask1_price"] > bid["bid1_price"]).all()
    assert (ask["ask1_volume"] >= 0).all()
    assert (bid["bid1_volume"] >= 0).all()
    assert {"event_actor", "maker_agent", "trade_count", "best_bid_depth", "best_ask_depth"}.issubset(latent.columns)
    assert {"taker_agent", "maker_agent", "maker_order_id", "queue_ahead"}.issubset(trades.columns)
