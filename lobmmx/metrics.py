from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class EpisodeResult:
    symbol: str
    day: str
    method: str
    episode_index: int
    pnl: float
    nd_pnl: float
    pnl_map: float
    profit_ratio: float
    avg_position: float
    avg_abs_position: float
    avg_spread: float
    turnover: float
    reward: float
    trades: int
    latency: int
    fill_rate: float = 0.0
    avg_bias_bps: float = 0.0
    avg_ask_distance_bps: float = 0.0
    avg_bid_distance_bps: float = 0.0
    avg_spread_bps: float = 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def sharpe(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float(arr.mean() / std)
