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


class BaselinePolicy:
    name: str

    def act(self, day: DayData, idx: int, inventory: float, step_cursor: int, total_steps: int) -> QuoteDecision:
        raise NotImplementedError


class RandomPolicy(BaselinePolicy):
    name = "Random"

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def act(self, day: DayData, idx: int, inventory: float, step_cursor: int, total_steps: int) -> QuoteDecision:
        offset = int(np.random.randint(0, 5))
        tick = self.config.tick_size
        ask = float(day.ask1[idx]) + offset * tick
        bid = float(day.bid1[idx]) - offset * tick
        ask, bid = price_legal_check(ask, bid, tick)
        return QuoteDecision(ask, -self.config.trade_unit, bid, self.config.trade_unit, ask - bid)


class FixedLevelPolicy(BaselinePolicy):
    def __init__(self, config: ExperimentConfig, level: int) -> None:
        self.config = config
        self.level = level
        self.name = f"Fixed_{level}"

    def act(self, day: DayData, idx: int, inventory: float, step_cursor: int, total_steps: int) -> QuoteDecision:
        tick = self.config.tick_size
        ask = float(day.ask1[idx]) + (self.level - 1) * tick
        bid = float(day.bid1[idx]) - (self.level - 1) * tick
        ask, bid = price_legal_check(ask, bid, tick)
        return QuoteDecision(ask, -self.config.trade_unit, bid, self.config.trade_unit, ask - bid)


class AvellanedaStoikovPolicy(BaselinePolicy):
    name = "AS"

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def act(self, day: DayData, idx: int, inventory: float, step_cursor: int, total_steps: int) -> QuoteDecision:
        gamma = self.config.as_gamma
        kappa = self.config.as_kappa
        time_left = max(total_steps - step_cursor, 1) / max(total_steps, 1)
        vol = float(day.dynamic[idx, 0] / 1e4)
        sigma2 = max(vol**2, 1e-8)
        reservation = float(day.midprice[idx]) - inventory * gamma * sigma2 * time_left
        spread = gamma * sigma2 * time_left + (2.0 / max(gamma, 1e-8)) * np.log(1.0 + gamma / max(kappa, 1e-8))
        spread = max(spread, self.config.tick_size)
        ask, bid = price_legal_check(reservation + spread / 2.0, reservation - spread / 2.0, self.config.tick_size)
        return QuoteDecision(ask, -self.config.trade_unit, bid, self.config.trade_unit, ask - bid)
