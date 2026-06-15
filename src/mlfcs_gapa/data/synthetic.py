"""Synthetic event-level LOB data.

This module is intentionally separate from the paper implementation. It exists
to create paper-shaped data when the paper's proprietary exchange data is
unavailable. The generated data must pass through the same schema as any
real-data adapter.

The market model is calibrated so that passive market making behaves like it
does on real limit-order-book data (and in the paper's Table II):

- The mid price is a latent *fair value* (a slow tick-level random walk with
  occasional trend and volatile regime segments) plus a *transient bounce*
  component that mean-reverts within a few events. The bounce makes
  event-level returns negatively autocorrelated (bid-ask bounce), which is
  the economic source of market-making profit.
- Market-order flow is mostly uninformed: order direction is only weakly
  coupled to the pressure factor that drives fair-value moves, so passive
  fills suffer mild - not fatal - adverse selection.
- Spread, bounce amplitude, and fair-step size scale with the stock's price
  level, so the three paper stocks have distinct microstructures.
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
    stable_regime_probability: float = 0.78
    trend_regime_probability: float = 0.12
    volatile_regime_probability: float = 0.10
    market_event_probability: float = 0.08

    @property
    def mean_spread_ticks(self) -> int:
        """Typical quoted spread in ticks, scaled with the price level."""

        return max(1, round(self.base_price * 0.0006 / self.tick_size))

    @property
    def bounce_amplitude_ticks(self) -> int:
        """Maximum transient mid displacement from fair value."""

        return max(1, self.mean_spread_ticks // 2)

    @property
    def fair_step_ticks(self) -> int:
        """Tick size of one permanent fair-value step."""

        return max(1, round(self.mean_spread_ticks / 2))


@dataclass(frozen=True)
class _MarketPaths:
    center_ticks: np.ndarray
    spread_ticks: np.ndarray
    pressure: np.ndarray
    market_buy_volume: np.ndarray
    market_buy_n: np.ndarray
    market_sell_volume: np.ndarray
    market_sell_n: np.ndarray
    trade_max_ticks: np.ndarray
    trade_max_volume: np.ndarray
    trade_min_ticks: np.ndarray
    trade_min_volume: np.ndarray


def generate_synthetic_lob_day(config: SyntheticLobConfig) -> LobDataset:
    """Generate one stock/day of event-level LOB data.

    The generator mimics the information available in the paper: 10-level LOB
    snapshots, message aggregates for OSI, and event-level trade extrema used
    for historical fills. It is not used by the paper logic except through the
    canonical schema.
    """

    rng = np.random.default_rng(config.seed)
    timestamps = _generate_timestamps(config.day, config.n_events)
    paths = _generate_market_paths(config, rng)

    orderbook = _build_orderbook(
        config, timestamps, paths.center_ticks, paths.spread_ticks, paths.pressure, rng
    )
    messages = _build_messages(timestamps, paths, rng)
    trades = _build_trade_summaries(config, timestamps, paths)

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


# Regime-dependent dynamics. Time shares follow the configured regime
# probabilities through duration-weighted segment sampling.
_REGIMES = ("stable", "trend", "volatile")
_REGIME_MEAN_DURATION = {"stable": 800.0, "trend": 300.0, "volatile": 190.0}
_REGIME_DURATION_RANGE = {"stable": (400, 1200), "trend": (150, 450), "volatile": (80, 300)}
_FAIR_STEP_PROBABILITY = {"stable": 0.012, "trend": 0.05, "volatile": 0.10}
_ARRIVAL_MULTIPLIER = {"stable": 1.0, "trend": 1.6, "volatile": 2.2}
_PRESSURE_DYNAMICS = {
    "stable": (0.92, 0.12),
    "trend": (0.97, 0.15),
    "volatile": (0.85, 0.30),
}
_SWEEP_PROBABILITY = {"stable": 0.04, "trend": 0.06, "volatile": 0.10}
_BOUNCE_DECAY_PROBABILITY = 0.25
_BOUNCE_JITTER_PROBABILITY = 0.02
_SPREAD_CHANGE_PROBABILITY = 0.05


def _generate_market_paths(config: SyntheticLobConfig, rng: np.random.Generator) -> _MarketPaths:
    n = config.n_events
    regimes, trend_signs = _segment_regimes(config, rng)

    base_tick = int(round(config.base_price / config.tick_size))
    step_ticks = config.fair_step_ticks
    bounce_amp = config.bounce_amplitude_ticks
    impact_ticks = max(1, bounce_amp // 2)
    spread_support, spread_weights = _spread_distribution(config)

    # Pre-drawn randomness keeps the per-event loop cheap.
    u_step = rng.random(n)
    u_dir = rng.random(n)
    u_big = rng.random(n)
    u_arrival = rng.random(n)
    u_side = rng.random(n)
    u_sweep = rng.random(n)
    u_decay = rng.random(n)
    u_jitter = rng.random(n)
    u_jitter_dir = rng.random(n)
    u_spread = rng.random(n)
    pressure_shocks = rng.standard_normal(n)
    order_counts = 1 + rng.poisson(0.5, n)
    order_lots = rng.integers(1, 3, n)
    spread_draws = rng.choice(spread_support, size=n, p=spread_weights)

    center_ticks = np.empty(n, dtype=np.int64)
    spread_ticks = np.empty(n, dtype=np.int64)
    pressure = np.empty(n, dtype=np.float64)
    market_buy_volume = np.zeros(n, dtype=np.int64)
    market_buy_n = np.zeros(n, dtype=np.int64)
    market_sell_volume = np.zeros(n, dtype=np.int64)
    market_sell_n = np.zeros(n, dtype=np.int64)
    trade_max_ticks = np.empty(n, dtype=np.int64)
    trade_max_volume = np.zeros(n, dtype=np.int64)
    trade_min_ticks = np.empty(n, dtype=np.int64)
    trade_min_volume = np.zeros(n, dtype=np.int64)

    fair_tick = base_tick
    bounce = 0
    raw_pressure = 0.0
    spread = int(spread_draws[0])

    for i in range(n):
        regime = regimes[i]
        trend_sign = trend_signs[i]

        persistence, shock_scale = _PRESSURE_DYNAMICS[regime]
        drift = 0.05 * trend_sign if regime == "trend" else 0.0
        raw_pressure = persistence * raw_pressure + drift + shock_scale * pressure_shocks[i]
        squashed = float(np.tanh(raw_pressure))
        pressure[i] = squashed

        # Permanent fair-value step: mostly noise, mildly pressure-informed,
        # directional inside trend segments.
        if u_step[i] < _FAIR_STEP_PROBABILITY[regime]:
            p_up = 0.5 + 0.25 * squashed
            if regime == "trend":
                p_up += 0.22 * trend_sign
            p_up = min(0.95, max(0.05, p_up))
            step = step_ticks if u_dir[i] < p_up else -step_ticks
            if regime == "volatile" and u_big[i] < 0.30:
                step *= 2
            fair_tick = max(2 * spread + 1, fair_tick + step)

        if u_spread[i] < _SPREAD_CHANGE_PROBABILITY:
            spread = int(spread_draws[i])

        # Market orders execute against the pre-impact book; their transient
        # one-tick impact moves the snapshot recorded after the event.
        pre_center = fair_tick + bounce
        pre_bid1 = pre_center - (spread // 2)
        pre_ask1 = pre_bid1 + spread

        side = 0
        if u_arrival[i] < config.market_event_probability * _ARRIVAL_MULTIPLIER[regime]:
            p_buy = 0.5 + 0.12 * squashed
            if regime == "trend":
                p_buy += 0.10 * trend_sign
            p_buy = min(0.90, max(0.10, p_buy))
            side = 1 if u_side[i] < p_buy else -1
            volume = int(order_counts[i] * order_lots[i]) * PAPER.minimum_trade_unit
            sweep = 1 if u_sweep[i] < _SWEEP_PROBABILITY[regime] else 0
            if side == 1:
                market_buy_volume[i] = volume
                market_buy_n[i] = int(order_counts[i])
                trade_max_ticks[i] = pre_ask1 + sweep
                trade_max_volume[i] = volume
                trade_min_ticks[i] = pre_center
                bounce = min(bounce_amp, bounce + impact_ticks)
            else:
                market_sell_volume[i] = volume
                market_sell_n[i] = int(order_counts[i])
                trade_min_ticks[i] = pre_bid1 - sweep
                trade_min_volume[i] = volume
                trade_max_ticks[i] = pre_center
                bounce = max(-bounce_amp, bounce - impact_ticks)
        else:
            trade_max_ticks[i] = pre_center
            trade_min_ticks[i] = pre_center
            if bounce != 0 and u_decay[i] < _BOUNCE_DECAY_PROBABILITY:
                decay = 1 + abs(bounce) // 2
                bounce -= decay if bounce > 0 else -decay
            elif u_jitter[i] < _BOUNCE_JITTER_PROBABILITY:
                p_up = 0.5 + 0.20 * squashed
                jitter = 1 if u_jitter_dir[i] < p_up else -1
                bounce = int(np.clip(bounce + jitter, -bounce_amp, bounce_amp))

        center_ticks[i] = fair_tick + bounce
        spread_ticks[i] = spread

    return _MarketPaths(
        center_ticks=center_ticks,
        spread_ticks=spread_ticks,
        pressure=pressure,
        market_buy_volume=market_buy_volume,
        market_buy_n=market_buy_n,
        market_sell_volume=market_sell_volume,
        market_sell_n=market_sell_n,
        trade_max_ticks=trade_max_ticks,
        trade_max_volume=trade_max_volume,
        trade_min_ticks=trade_min_ticks,
        trade_min_volume=trade_min_volume,
    )


def _segment_regimes(
    config: SyntheticLobConfig, rng: np.random.Generator
) -> tuple[list[str], np.ndarray]:
    """Sample regime segments whose time shares match the configured mix."""

    probabilities = {
        "stable": config.stable_regime_probability,
        "trend": config.trend_regime_probability,
        "volatile": config.volatile_regime_probability,
    }
    weights = np.array(
        [probabilities[regime] / _REGIME_MEAN_DURATION[regime] for regime in _REGIMES]
    )
    weights = weights / weights.sum()

    regimes: list[str] = []
    trend_signs = np.zeros(config.n_events, dtype=np.float64)
    position = 0
    while position < config.n_events:
        regime = str(rng.choice(list(_REGIMES), p=weights))
        low, high = _REGIME_DURATION_RANGE[regime]
        duration = int(rng.integers(low, high + 1))
        duration = min(duration, config.n_events - position)
        regimes.extend([regime] * duration)
        if regime == "trend":
            trend_signs[position : position + duration] = float(rng.choice([-1.0, 1.0]))
        position += duration
    return regimes, trend_signs


def _spread_distribution(config: SyntheticLobConfig) -> tuple[np.ndarray, np.ndarray]:
    mean = config.mean_spread_ticks
    support_weights = {
        max(1, mean - 1): 0.15,
        mean: 0.50,
        mean + 1: 0.22,
        mean + 2: 0.09,
        mean + 3: 0.04,
    }
    support = np.array(sorted(support_weights), dtype=np.int64)
    weights = np.array([support_weights[ticks] for ticks in support], dtype=np.float64)
    return support, weights / weights.sum()


def _build_orderbook(
    config: SyntheticLobConfig,
    timestamps: list[datetime],
    center_ticks: np.ndarray,
    spread_ticks: np.ndarray,
    pressure: np.ndarray,
    rng: np.random.Generator,
) -> pl.DataFrame:
    data: dict[str, object] = {"timestamp": timestamps}

    bid1_ticks = center_ticks - (spread_ticks // 2)
    ask1_ticks = bid1_ticks + spread_ticks

    for level in range(1, config.levels + 1):
        distance = level - 1
        depth_mean = 900 + 190 * level
        level_decay = np.exp(-(level - 1) / 5.0)
        imbalance_scale = 0.45 * level_decay * pressure
        ask_noise = rng.lognormal(mean=0.0, sigma=0.30, size=len(timestamps))
        bid_noise = rng.lognormal(mean=0.0, sigma=0.30, size=len(timestamps))
        ask_volume = _round_lot(depth_mean * np.exp(-imbalance_scale) * ask_noise, rng)
        bid_volume = _round_lot(
            depth_mean * np.exp(imbalance_scale) * bid_noise,
            rng,
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
    paths: _MarketPaths,
    rng: np.random.Generator,
) -> pl.DataFrame:
    pressure = paths.pressure

    limit_n = rng.poisson(3.5, len(timestamps))
    limit_buy_n = rng.binomial(limit_n, np.clip(0.50 + 0.18 * pressure, 0.12, 0.88))
    limit_sell_n = limit_n - limit_buy_n

    withdraw_n = rng.poisson(2.1, len(timestamps))
    withdraw_buy_n = rng.binomial(withdraw_n, np.clip(0.50 - 0.16 * pressure, 0.12, 0.88))
    withdraw_sell_n = withdraw_n - withdraw_buy_n

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "market_buy_volume": paths.market_buy_volume,
            "market_buy_n": paths.market_buy_n,
            "market_sell_volume": paths.market_sell_volume,
            "market_sell_n": paths.market_sell_n,
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
    paths: _MarketPaths,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "trade_price_min": paths.trade_min_ticks * config.tick_size,
            "trade_price_min_volume": paths.trade_min_volume,
            "trade_price_max": paths.trade_max_ticks * config.tick_size,
            "trade_price_max_volume": paths.trade_max_volume,
            "trade_volume_total": (paths.trade_max_volume + paths.trade_min_volume),
        }
    )
