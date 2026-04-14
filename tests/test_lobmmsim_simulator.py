from __future__ import annotations

from pathlib import Path

import pandas as pd

from lobmmsim.config import GenerateConfig
from lobmmsim.simulator import generate_dataset


def test_simulator_is_deterministic_and_book_is_valid(tmp_path: Path) -> None:
    data_dir_1 = tmp_path / "sim1"
    data_dir_2 = tmp_path / "sim2"
    cfg1 = GenerateConfig(mode="smoke", data_dir=str(data_dir_1), symbols=["000001"], seed=17).apply_mode_defaults()
    cfg2 = GenerateConfig(mode="smoke", data_dir=str(data_dir_2), symbols=["000001"], seed=17).apply_mode_defaults()
    generate_dataset(cfg1)
    generate_dataset(cfg2)
    day = "20191101"
    files = ["ask.csv", "bid.csv", "price.csv", "msg.csv", "trades.csv", "latent.csv"]
    for name in files:
        assert (data_dir_1 / "000001" / day / name).read_bytes() == (data_dir_2 / "000001" / day / name).read_bytes()

    ask = pd.read_csv(data_dir_1 / "000001" / day / "ask.csv")
    bid = pd.read_csv(data_dir_1 / "000001" / day / "bid.csv")
    price = pd.read_csv(data_dir_1 / "000001" / day / "price.csv")
    spread_ticks = ((price["ask1_price"] - price["bid1_price"]) / cfg1.tick_size).round().astype(int)
    assert (spread_ticks.isin([1, 2])).mean() > 0.9
    for level in range(1, 11):
        assert (ask[f"ask{level}_volume"] > 0).all()
        assert (bid[f"bid{level}_volume"] > 0).all()
        if level < 10:
            assert (ask[f"ask{level}_price"] < ask[f"ask{level + 1}_price"]).all()
            assert (bid[f"bid{level}_price"] > bid[f"bid{level + 1}_price"]).all()


def test_default_event_rate_implies_paper_like_episode_duration() -> None:
    cfg = GenerateConfig().apply_mode_defaults()
    session_seconds = 4 * 60 * 60
    seconds_per_episode = cfg.episode_length * session_seconds / cfg.events_per_day["000001"]
    assert 180.0 <= seconds_per_episode <= 300.0
