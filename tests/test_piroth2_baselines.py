from __future__ import annotations

import numpy as np
import pandas as pd

from piroth.baselines import calibrate_avellaneda_stoikov
from piroth.config import DiagnosticsConfig
from piroth.paper_evaluation import evaluate_paper_baselines
from piroth.simulator import SyntheticDay


def test_as_calibration_returns_positive_kappa() -> None:
    config = DiagnosticsConfig(mode="smoke", symbol="000001", seed=3, as_fill_horizon_events=4, as_max_distance_ticks=3)
    calibration = calibrate_avellaneda_stoikov([_calibration_day()], config)

    assert calibration.kappa > 0
    assert calibration.sigma2_event > 0


def test_fast_paper_baseline_evaluator_writes_all_policies(tmp_path) -> None:
    config = DiagnosticsConfig(
        mode="smoke",
        symbol="000001",
        seed=3,
        lookback=2,
        latency=0,
        episode_length=20,
        stable_windows=["10:00:00-10:02:00"],
        max_eval_episodes_per_day=1,
        as_fill_horizon_events=4,
        as_max_distance_ticks=3,
    )

    summary = evaluate_paper_baselines([_calibration_day()], [_calibration_day()], config, tmp_path)

    assert set(summary) == {"AS", "Fixed_1", "Fixed_2", "Fixed_3", "Random"}
    assert (tmp_path / "paper_baseline_episodes.csv").exists()
    assert (tmp_path / "as_calibration.json").exists()


def _calibration_day() -> SyntheticDay:
    timestamps = pd.date_range("2019-11-01 10:00:00", periods=80, freq="s")
    mid = 12.50 + 0.03 * np.sin(np.linspace(0.0, 8.0, len(timestamps))) + np.linspace(0.0, 0.02, len(timestamps))
    ask = pd.DataFrame([_lob_row(ts, "ask", price + 0.01, 1500) for ts, price in zip(timestamps, mid, strict=True)])
    bid = pd.DataFrame([_lob_row(ts, "bid", price - 0.01, 1500) for ts, price in zip(timestamps, mid, strict=True)])
    price = pd.DataFrame(
        {
            "timestamp": timestamps,
            "midprice": mid,
            "ask1_price": ask["ask1_price"],
            "bid1_price": bid["bid1_price"],
            "spread_ticks": 2,
            "return_bp": np.r_[0.0, 10_000.0 * np.diff(mid) / mid[:-1]],
        }
    )
    trade_idx = np.arange(5, len(timestamps), 7)
    trades = pd.DataFrame(
        {
            "timestamp": timestamps[trade_idx],
            "price": np.where(trade_idx % 2 == 0, ask.loc[trade_idx, "ask1_price"], bid.loc[trade_idx, "bid1_price"]),
            "size": 100,
            "aggressor_side": np.where(trade_idx % 2 == 0, "B", "A"),
        }
    )
    msg = pd.DataFrame({"timestamp": timestamps})
    latent = pd.DataFrame({"timestamp": timestamps, "fair_value": mid, "event_kind": "test"})
    return SyntheticDay(
        symbol="000001",
        day="20191101",
        ask=ask,
        bid=bid,
        price=price,
        trades=trades,
        msg=msg,
        event_log=pd.DataFrame(),
        latent=latent,
        depth_cube=np.zeros((len(timestamps), 31), dtype=np.float32),
    )


def _lob_row(timestamp: pd.Timestamp, side: str, start_price: float, volume: int) -> dict[str, float | int | pd.Timestamp]:
    row: dict[str, float | int | pd.Timestamp] = {"timestamp": timestamp}
    sign = 1 if side == "ask" else -1
    for level in range(1, 11):
        row[f"{side}{level}_price"] = round(start_price + sign * (level - 1) * 0.01, 2)
        row[f"{side}{level}_volume"] = volume
    return row
