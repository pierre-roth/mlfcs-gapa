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
    """Per-episode Sharpe (NOT annualized). Initial way of computing it in this paper"""
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float(arr.mean() / std)

# Annualized Sharpe helpers
_TRADING_DAYS_PER_YEAR = 252
 
 
def sharpe_annualized_episodes(
    values: list[float],
    episodes_per_day: float,
) -> float:
    """Annualized Sharpe from per-episode PnLs. Scales by sqrt(episodes_per_day * 252) so the number is comparable
    across different episode lengths and test-set sizes.
    """
    if len(values) < 2 or episodes_per_day <= 0:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    per_episode_sharpe = float(arr.mean() / std)
    return per_episode_sharpe * np.sqrt(episodes_per_day * _TRADING_DAYS_PER_YEAR)
 
def sharpe_daily(
    pnls: list[float],
    days: list[str],
) -> float:
    """Annualized Sharpe from daily aggregated PnLs. (industry standard)
 
    Groups episode PnLs by day, sums within each day, then computes
    annualized Sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(252).
    """
    if len(pnls) < 2:
        return 0.0
    import pandas as pd  # local import to keep module lightweight
    df = pd.DataFrame({"pnl": pnls, "day": days})
    daily = df.groupby("day")["pnl"].sum()
    if len(daily) < 2:
        return 0.0
    std = float(daily.std(ddof=1))
    if std == 0:
        return 0.0
    return float(daily.mean() / std * np.sqrt(_TRADING_DAYS_PER_YEAR))
 
