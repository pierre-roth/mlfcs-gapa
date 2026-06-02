"""Synthetic event-level LOB data.

This module is intentionally separate from the paper implementation. It exists
to create paper-shaped data when the paper's proprietary exchange data is
unavailable.
The generated data must pass through the same schema as any real-data adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from mlfcs_gapa.data.schema import LobDataset
from mlfcs_gapa.paper.constants import PAPER


@dataclass(frozen=True)
class SyntheticLobConfig:
    stock: str = "000001"
    day: str = "2019-11-01"
    n_events: int = PAPER.episode_events * 3
    levels: int = PAPER.lob_levels
    tick_size: float = 0.01
    base_price: float = 16.45
    seed: int = 1
    stable_regime_probability: float = 0.72
    trend_regime_probability: float = 0.18
    volatile_regime_probability: float = 0.10
    market_event_probability: float = 0.35


def generate_synthetic_lob_day(config: SyntheticLobConfig) -> LobDataset:
    """Generate one stock/day of event-level LOB data.

    The generator is built to mimic the information available in the paper:
    10-level LOB snapshots, message aggregates for OSI, and event-level trade
    extrema used for historical fills. It is not used by the paper logic except
    through the canonical schema.
    """

    rng = np.random.default_rng(config.seed)
    timestamps = _generate_timestamps(config.day, config.n_events)
    center_ticks = _generate_mid_ticks(config, rng)
    spread_ticks = rng.choice([1, 2, 3], size=config.n_events, p=[0.76, 0.20, 0.04])

    orderbook = _build_orderbook(config, timestamps, center_ticks, spread_ticks, rng)
    messages = _build_messages(timestamps, center_ticks, rng, config.market_event_probability)
    trades = _build_trade_summaries(config, timestamps, center_ticks, spread_ticks, messages, rng)

    return LobDataset(
        stock=config.stock,
        day=config.day,
        orderbook=orderbook,
        messages=messages,
        trades=trades,
    )


def _generate_timestamps(day: str, n_events: int) -> list[datetime]:
    morning_start = datetime.fromisoformat(f"{day} 09:30:00")
    morning_end = datetime.fromisoformat(f"{day} 11:30:00")
    afternoon_start = datetime.fromisoformat(f"{day} 13:00:00")
    afternoon_end = datetime.fromisoformat(f"{day} 14:57:00")

    morning_seconds = int((morning_end - morning_start).total_seconds())
    afternoon_seconds = int((afternoon_end - afternoon_start).total_seconds())
    total_seconds = morning_seconds + afternoon_seconds

    n_morning = int(round(n_events * morning_seconds / total_seconds))
    n_afternoon = n_events - n_morning

    return [
        *_linspace_datetimes(morning_start, morning_end, n_morning),
        *_linspace_datetimes(afternoon_start, afternoon_end, n_afternoon),
    ]


def _linspace_datetimes(start: datetime, end: datetime, n: int) -> list[datetime]:
    if n <= 1:
        return [start]
    total_us = int((end - start).total_seconds() * 1_000_000)
    offsets = np.linspace(0, total_us, n, endpoint=False, dtype=np.int64)
    return [start + timedelta(microseconds=int(offset)) for offset in offsets]


def _generate_mid_ticks(config: SyntheticLobConfig, rng: np.random.Generator) -> np.ndarray:
    n = config.n_events
    base_tick = int(round(config.base_price / config.tick_size))
    ticks = np.empty(n, dtype=np.int64)
    ticks[0] = base_tick

    regimes = rng.choice(
        ["stable", "trend", "volatile"],
        size=n,
        p=[
            config.stable_regime_probability,
            config.trend_regime_probability,
            config.volatile_regime_probability,
        ],
    )
    trend_sign = rng.choice([-1, 1])
    trend_clock = 0

    for i in range(1, n):
        if trend_clock <= 0 and regimes[i] == "trend":
            trend_sign = rng.choice([-1, 1])
            trend_clock = int(rng.integers(80, 420))
        trend_clock = max(0, trend_clock - 1)

        if regimes[i] == "stable":
            step = rng.choice([-1, 0, 1], p=[0.08, 0.84, 0.08])
        elif regimes[i] == "trend":
            step = rng.choice([trend_sign, 0, -trend_sign], p=[0.55, 0.35, 0.10])
        else:
            step = rng.choice([-2, -1, 0, 1, 2], p=[0.12, 0.24, 0.28, 0.24, 0.12])
        ticks[i] = max(1, ticks[i - 1] + int(step))

    return ticks


def _build_orderbook(
    config: SyntheticLobConfig,
    timestamps: list[datetime],
    center_ticks: np.ndarray,
    spread_ticks: np.ndarray,
    rng: np.random.Generator,
) -> pl.DataFrame:
    data: dict[str, object] = {"timestamp": timestamps}

    bid1_ticks = center_ticks - (spread_ticks // 2)
    ask1_ticks = bid1_ticks + spread_ticks

    for level in range(1, config.levels + 1):
        distance = level - 1
        depth_mean = 900 + 190 * level
        depth_noise = rng.lognormal(mean=0.0, sigma=0.35, size=len(timestamps))
        ask_volume = _round_lot(depth_mean * depth_noise, rng)
        bid_volume = _round_lot(
            depth_mean * rng.lognormal(mean=0.0, sigma=0.35, size=len(timestamps)), rng
        )

        data[f"ask{level}_price"] = (ask1_ticks + distance) * config.tick_size
        data[f"ask{level}_volume"] = ask_volume
        data[f"bid{level}_price"] = (bid1_ticks - distance) * config.tick_size
        data[f"bid{level}_volume"] = bid_volume

    return pl.DataFrame(data)


def _round_lot(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    jitter = rng.integers(-2, 3, size=values.shape) * PAPER.minimum_trade_unit
    lots = np.maximum(
        PAPER.minimum_trade_unit,
        np.round(values / PAPER.minimum_trade_unit) * PAPER.minimum_trade_unit,
    )
    return np.maximum(PAPER.minimum_trade_unit, lots + jitter).astype(np.int64)


def _build_messages(
    timestamps: list[datetime],
    center_ticks: np.ndarray,
    rng: np.random.Generator,
    market_event_probability: float,
) -> pl.DataFrame:
    price_move = np.diff(center_ticks, prepend=center_ticks[0])
    buy_pressure = np.clip(0.5 + 0.18 * np.sign(price_move), 0.1, 0.9)
    has_market = rng.random(len(timestamps)) < market_event_probability

    market_n = rng.poisson(1.2, len(timestamps)) * has_market
    market_buy_n = rng.binomial(market_n, buy_pressure)
    market_sell_n = market_n - market_buy_n

    limit_n = rng.poisson(3.5, len(timestamps))
    limit_buy_n = rng.binomial(limit_n, np.clip(0.52 - 0.12 * np.sign(price_move), 0.15, 0.85))
    limit_sell_n = limit_n - limit_buy_n

    withdraw_n = rng.poisson(2.1, len(timestamps))
    withdraw_buy_n = rng.binomial(
        withdraw_n, np.clip(0.48 + 0.10 * np.sign(price_move), 0.15, 0.85)
    )
    withdraw_sell_n = withdraw_n - withdraw_buy_n

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "market_buy_volume": _order_volume(market_buy_n, rng),
            "market_buy_n": market_buy_n,
            "market_sell_volume": _order_volume(market_sell_n, rng),
            "market_sell_n": market_sell_n,
            "limit_buy_volume": _order_volume(limit_buy_n, rng),
            "limit_buy_n": limit_buy_n,
            "limit_sell_volume": _order_volume(limit_sell_n, rng),
            "limit_sell_n": limit_sell_n,
            "withdraw_buy_volume": _order_volume(withdraw_buy_n, rng),
            "withdraw_buy_n": withdraw_buy_n,
            "withdraw_sell_volume": _order_volume(withdraw_sell_n, rng),
            "withdraw_sell_n": withdraw_sell_n,
        }
    )


def _order_volume(counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lots_per_order = rng.integers(1, 8, size=counts.shape)
    return counts.astype(np.int64) * lots_per_order.astype(np.int64) * PAPER.minimum_trade_unit


def _build_trade_summaries(
    config: SyntheticLobConfig,
    timestamps: list[datetime],
    center_ticks: np.ndarray,
    spread_ticks: np.ndarray,
    messages: pl.DataFrame,
    rng: np.random.Generator,
) -> pl.DataFrame:
    bid1_ticks = center_ticks - (spread_ticks // 2)
    ask1_ticks = bid1_ticks + spread_ticks

    market_buy_volume = messages["market_buy_volume"].to_numpy()
    market_sell_volume = messages["market_sell_volume"].to_numpy()
    has_buy = market_buy_volume > 0
    has_sell = market_sell_volume > 0

    buy_sweep = rng.binomial(2, 0.12, len(timestamps))
    sell_sweep = rng.binomial(2, 0.12, len(timestamps))

    trade_max_ticks = np.where(has_buy, ask1_ticks + buy_sweep, center_ticks)
    trade_min_ticks = np.where(has_sell, bid1_ticks - sell_sweep, center_ticks)

    trade_max_volume = np.where(has_buy, np.maximum(PAPER.minimum_trade_unit, market_buy_volume), 0)
    trade_min_volume = np.where(
        has_sell, np.maximum(PAPER.minimum_trade_unit, market_sell_volume), 0
    )

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "trade_price_min": trade_min_ticks * config.tick_size,
            "trade_price_min_volume": trade_min_volume.astype(np.int64),
            "trade_price_max": trade_max_ticks * config.tick_size,
            "trade_price_max_volume": trade_max_volume.astype(np.int64),
            "trade_volume_total": (trade_max_volume + trade_min_volume).astype(np.int64),
        }
    )
