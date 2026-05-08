from __future__ import annotations

from pathlib import Path

import pandas as pd

from piroth.config import DiagnosticsConfig
from piroth.real_data import RealMarketDataLoader, load_market_days


def test_real_loader_normalizes_paper_day(tmp_path: Path) -> None:
    _write_real_day(tmp_path, "AAPL", "20260302")
    config = DiagnosticsConfig(
        mode="full",
        data_source="real",
        real_data_root=str(tmp_path),
        real_start_time="10:00:00",
        real_end_time="10:01:00",
        symbol="AAPL",
        train_days=0,
        test_days=1,
        events_per_day_override=2,
        real_chunk_size=2,
        real_build_depth_cube=True,
    )

    day = load_market_days(config, "test")[0]

    assert day.symbol == "AAPL"
    assert day.day == "20260302"
    assert len(day.price) == 2
    assert set(["best_bid", "best_ask", "spread", "spread_ticks", "microprice", "return_bp"]).issubset(day.price.columns)
    assert set(["signed_size", "taker_agent", "maker_agent_id", "maker_agent", "maker_order_id", "queue_ahead"]).issubset(day.trades.columns)
    assert day.depth_cube.shape[0] == 2


def test_real_loader_uses_sorted_train_test_split(tmp_path: Path) -> None:
    for day in ["20260303", "20260301", "20260302"]:
        _write_real_day(tmp_path, "GOOGL", day)
    config = DiagnosticsConfig(
        mode="full",
        data_source="real",
        real_data_root=str(tmp_path),
        symbol="GOOGL",
        train_days=2,
        test_days=1,
        events_per_day_override=1,
    )
    loader = RealMarketDataLoader(config)

    assert loader.train_day_names() == ["20260301", "20260302"]
    assert loader.test_day_names() == ["20260303"]


def _write_real_day(root: Path, symbol: str, day: str) -> None:
    day_root = root / symbol / day
    day_root.mkdir(parents=True)
    timestamps = pd.to_datetime(
        [
            f"{day[:4]}-{day[4:6]}-{day[6:]} 10:00:00.000000001",
            f"{day[:4]}-{day[4:6]}-{day[6:]} 10:00:00.000000002",
            f"{day[:4]}-{day[4:6]}-{day[6:]} 10:00:00.000000003",
        ]
    )
    ask = pd.DataFrame({"timestamp": timestamps})
    bid = pd.DataFrame({"timestamp": timestamps})
    for level in range(1, 11):
        ask[f"ask{level}_price"] = 100.00 + level * 0.01
        ask[f"ask{level}_volume"] = 100 + level
        bid[f"bid{level}_price"] = 100.00 - level * 0.01
        bid[f"bid{level}_volume"] = 120 + level
    price = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ask1_price": ask["ask1_price"],
            "bid1_price": bid["bid1_price"],
            "midprice": 100.00,
        }
    )
    msg = pd.DataFrame(
        {
            "timestamp": timestamps,
            "market_buy_volume": [0, 100, 0],
            "market_buy_n": [0, 1, 0],
            "market_sell_volume": [0, 0, 100],
            "market_sell_n": [0, 0, 1],
            "limit_buy_volume": [10, 0, 0],
            "limit_buy_n": [1, 0, 0],
            "limit_sell_volume": [0, 10, 0],
            "limit_sell_n": [0, 1, 0],
            "withdraw_buy_volume": [0, 0, 5],
            "withdraw_buy_n": [0, 0, 1],
            "withdraw_sell_volume": [5, 0, 0],
            "withdraw_sell_n": [1, 0, 0],
        }
    )
    trades = pd.DataFrame(
        {
            "timestamp": timestamps[1:],
            "price": [99.99, 100.01],
            "size": [100, 100],
            "aggressor_side": ["A", "B"],
        }
    )
    ask.to_csv(day_root / "ask.csv", index=False)
    bid.to_csv(day_root / "bid.csv", index=False)
    price.to_csv(day_root / "price.csv", index=False)
    msg.to_csv(day_root / "msg.csv", index=False)
    trades.to_csv(day_root / "trades.csv", index=False)
