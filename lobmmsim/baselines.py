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


class FixedLevelPolicy(BaselinePolicy):
    def __init__(self, config: ExperimentConfig, level: int = 1) -> None:
        self.config = config
        self.level = level
        self.name = f"Fixed_{level}"

    def act(self, day: DayData, idx: int, inventory: float, step_cursor: int, total_steps: int) -> QuoteDecision:
        tick = self.config.tick_size
        ask = float(day.ask1[idx]) + (self.level - 1) * tick
        bid = float(day.bid1[idx]) - (self.level - 1) * tick
        ask, bid = price_legal_check(ask, bid, tick)
        return QuoteDecision(ask, -self.config.trade_unit, bid, self.config.trade_unit, ask - bid)


class OracleAlphaPolicy(BaselinePolicy):
    name = "OracleAlpha"

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def act(self, day: DayData, idx: int, inventory: float, step_cursor: int, total_steps: int) -> QuoteDecision:
        latent = day.latent.iloc[idx]
        mid = float(day.midprice[idx])
        fair_value = float(latent["efficient_price"])
        delta = float(np.clip(fair_value - mid, -self.config.max_bias, self.config.max_bias))
        reservation = mid + delta - np.sign(inventory) * min(abs(delta), self.config.max_bias * 0.25)
        spread = max(self.config.tick_size, float(day.spread[idx]))
        ask, bid = price_legal_check(reservation + spread / 2.0, reservation - spread / 2.0, self.config.tick_size)
        return QuoteDecision(ask, -self.config.trade_unit, bid, self.config.trade_unit, ask - bid)
