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
    """Per-episode Sharpe; not annualized."""
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float(arr.mean() / std)


_TRADING_DAYS_PER_YEAR = 252


def sharpe_annualized_episodes(values: list[float], episodes_per_day: float) -> float:
    """Annualize a per-episode Sharpe using the observed episodes per day."""
    if len(values) < 2 or episodes_per_day <= 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(episodes_per_day * _TRADING_DAYS_PER_YEAR))


def sharpe_daily(pnls: list[float], days: list[str]) -> float:
    """Annualized Sharpe from daily aggregated PnL."""
    if len(pnls) < 2 or len(pnls) != len(days):
        return 0.0
    daily_pnl: dict[str, float] = {}
    for day, pnl in zip(days, pnls, strict=True):
        daily_pnl[str(day)] = daily_pnl.get(str(day), 0.0) + float(pnl)
    if len(daily_pnl) < 2:
        return 0.0
    arr = np.asarray(list(daily_pnl.values()), dtype=np.float64)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(_TRADING_DAYS_PER_YEAR))
