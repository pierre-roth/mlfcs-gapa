"""Paper-faithful action-space transformations."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import numpy as np

from mlfcs_gapa.paper.constants import PAPER


@dataclass(frozen=True)
class Quote:
    ask_price: float
    ask_volume: int
    bid_price: float
    bid_volume: int
    reservation_price: float
    spread: float


def legalize_quote_prices(
    ask_price: float, bid_price: float, tick_size: float = 0.01
) -> tuple[float, float]:
    """Round ask up and bid down to legal cent ticks."""

    scale = 1.0 / tick_size
    return ceil(ask_price * scale) / scale, floor(bid_price * scale) / scale


def continuous_action_to_quote(
    action: np.ndarray,
    mid_price: float,
    inventory: int,
    *,
    tick_size: float = 0.01,
    max_bias: float = PAPER.max_bias,
    max_spread: float = PAPER.max_spread,
    trade_unit: int = PAPER.minimum_trade_unit,
) -> Quote:
    """Convert the paper's continuous action `(A1, A2)` to bid/ask quotes.

    Paper equations:

    - `delta = A1 * max_bias`
    - `reservation = mid_price - sign(inventory) * delta`
    - `spread = A2 * max_spread`
    - `ask,bid = reservation +/- spread / 2`
    """

    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,):
        raise ValueError(f"continuous action must have shape (2,), got {action.shape}")
    if not np.all((0.0 <= action) & (action <= 1.0)):
        raise ValueError("paper-faithful continuous actions must be in [0, 1]")

    delta = float(action[0] * max_bias)
    spread = float(action[1] * max_spread)
    reservation = float(mid_price - np.sign(inventory) * delta)
    ask_price, bid_price = legalize_quote_prices(
        reservation + spread / 2.0,
        reservation - spread / 2.0,
        tick_size=tick_size,
    )

    return Quote(
        ask_price=ask_price,
        ask_volume=-trade_unit,
        bid_price=bid_price,
        bid_volume=trade_unit,
        reservation_price=reservation,
        spread=spread,
    )
