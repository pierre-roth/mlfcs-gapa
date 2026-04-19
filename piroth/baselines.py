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

    def __init__(self, config: ExperimentConfig, gamma: float = 0.1, kappa: float = 1.5) -> None:
        self.config = config
        self.gamma = gamma
        self.kappa = kappa

    def act(self, day: DayData, idx: int, inventory: float, step_cursor: int, total_steps: int) -> QuoteDecision:
        mid = float(day.midprice[idx])
        sigma = max(float(day.dynamic[idx, 0]) / 1e4, 1e-6)
        remaining = max(total_steps - step_cursor, 1) / max(total_steps, 1)
        reservation = mid - inventory / max(self.config.trade_unit, 1) * self.gamma * sigma * sigma * remaining
        spread = self.gamma * sigma * sigma * remaining + 2.0 / self.gamma * np.log1p(self.gamma / self.kappa)
        spread = float(np.clip(spread, self.config.tick_size, self.config.max_spread))
        ask, bid = price_legal_check(reservation + spread / 2.0, reservation - spread / 2.0, self.config.tick_size)
        return QuoteDecision(ask, -self.config.trade_unit, bid, self.config.trade_unit, ask - bid)

