from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .data import DayData
from .utils import price_legal_check


@dataclass
class QuoteDecision:
    ask_price: float
    ask_volume: float
    bid_price: float
    bid_volume: float
    spread: float


@dataclass
class ASCalibration:
    gamma: float
    kappa: float
    step_variance: float
    base_spread: float


class FixedLevelPolicy:
    def __init__(self, config: ExperimentConfig, level: int) -> None:
        self.config = config
        self.level = level
        self.name = f"Fixed_{level}"

    def act(self, day: DayData, idx: int, inventory: float, step_cursor: int, total_steps: int) -> QuoteDecision:
        ask = float(day.ask1[idx]) + (self.level - 1) * self.config.tick_size
        bid = float(day.bid1[idx]) - (self.level - 1) * self.config.tick_size
        ask, bid = price_legal_check(ask, bid, self.config.tick_size)
        return QuoteDecision(ask, -self.config.trade_unit, bid, self.config.trade_unit, ask - bid)


class AvellanedaStoikovPolicy:
    name = "AS"

    def __init__(self, config: ExperimentConfig, calibration: ASCalibration) -> None:
        self.config = config
        self.calibration = calibration

    def act(self, day: DayData, idx: int, inventory: float, step_cursor: int, total_steps: int) -> QuoteDecision:
        mid = float(day.midprice[idx])
        remaining_steps = max(total_steps - step_cursor, 1)
        variance_to_horizon = max(self.calibration.step_variance * remaining_steps, self.config.tick_size**2)
        inventory_units = inventory / max(self.config.trade_unit, 1)
        reservation = mid - inventory_units * self.calibration.gamma * variance_to_horizon
        spread = (
            self.calibration.gamma * variance_to_horizon
            + 2.0 / self.calibration.gamma * np.log1p(self.calibration.gamma / self.calibration.kappa)
        )
        spread = max(spread, self.calibration.base_spread)
        spread = float(np.clip(spread, self.config.tick_size, self.config.max_spread))
        ask, bid = price_legal_check(reservation + spread / 2.0, reservation - spread / 2.0, self.config.tick_size)
        return QuoteDecision(ask, -self.config.trade_unit, bid, self.config.trade_unit, ask - bid)


def _fill_probability_at_distance(days: list[DayData], config: ExperimentConfig, ticks: int) -> float:
    hits = 0.0
    total = 0.0
    offset = ticks * config.tick_size
    for day in days:
        valid = day.valid_label_indices(config.lookback, config.pretrain_horizon)
        for idx in valid:
            trades = day.trades_by_index.get(int(idx))
            total += 2.0
            if trades is None or trades.price.size == 0:
                continue
            ask_price = float(day.ask1[idx]) + offset
            bid_price = float(day.bid1[idx]) - offset
            if np.any((trades.aggressor_side == "B") & (trades.price >= ask_price)):
                hits += 1.0
            if np.any((trades.aggressor_side == "A") & (trades.price <= bid_price)):
                hits += 1.0
    return hits / max(total, 1.0)


def calibrate_avellaneda_stoikov(days: list[DayData], config: ExperimentConfig) -> ASCalibration:
    diffs = []
    for day in days:
        valid = day.valid_label_indices(config.lookback, config.pretrain_horizon)
        if valid.size < 2:
            continue
        mid = day.midprice[valid].astype(np.float64)
        diffs.append(np.diff(mid))
    if diffs:
        step_variance = float(np.var(np.concatenate(diffs)))
    else:
        step_variance = float(config.tick_size**2)
    step_variance = max(step_variance, (0.15 * config.tick_size) ** 2)

    max_ticks = max(2, min(6, int(round(config.max_spread / config.tick_size))))
    distances = []
    probs = []
    for ticks in range(0, max_ticks + 1):
        prob = _fill_probability_at_distance(days, config, ticks)
        if prob > 0:
            distances.append(ticks * config.tick_size)
            probs.append(prob)
    if len(probs) >= 2:
        slope, intercept = np.polyfit(np.asarray(distances, dtype=np.float64), np.log(np.asarray(probs, dtype=np.float64)), 1)
        kappa = float(max(-slope, 2.0 / max(config.max_spread, config.tick_size)))
        fill_at_touch = float(np.exp(intercept))
    else:
        kappa = float(2.0 / max(2.5 * config.tick_size, config.max_spread * 0.5))
        fill_at_touch = 0.01

    horizon_variance = step_variance * max(config.episode_length, 1)
    target_inventory_skew = 1.5 * config.tick_size
    gamma = target_inventory_skew / max(config.max_inventory_units * horizon_variance, config.tick_size**2)
    gamma = float(np.clip(gamma, 0.005, 0.1))

    base_spread = max(config.tick_size, min(config.max_spread, 2.0 / max(kappa, 1e-6)))
    if fill_at_touch < 1e-4:
        base_spread = min(config.max_spread, max(base_spread, 2.0 * config.tick_size))

    return ASCalibration(
        gamma=gamma,
        kappa=kappa,
        step_variance=step_variance,
        base_spread=float(base_spread),
    )
