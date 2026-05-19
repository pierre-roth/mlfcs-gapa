"""Reward functions from the paper."""

from __future__ import annotations

from dataclasses import dataclass

from mlfcs_gapa.paper.constants import PAPER


@dataclass(frozen=True)
class RewardBreakdown:
    dampened_pnl: float
    trading_pnl: float
    inventory_penalty: float
    reward: float


def dampened_pnl(delta_pnl: float, eta: float = PAPER.eta_dampened_pnl) -> float:
    """Equation 13: `DP_t = DeltaPnL_t - max(0, eta * DeltaPnL_t)`."""

    return float(delta_pnl - max(0.0, eta * delta_pnl))


def trading_pnl(mid_price: float, trade_price: float, trade_volume: int) -> float:
    """Equation 14: `TP_t = X_v * (mid_price - trade_price)`.

    Buy volume is positive and sell volume is negative. This rewards buying
    below mid and selling above mid.
    """

    return float(trade_volume * (mid_price - trade_price))


def inventory_penalty(inventory: int, zeta: float = PAPER.zeta_inventory_penalty) -> float:
    """Equation 15 with inventory measured in minimum-trade-unit lots.

    The paper writes `zeta * Inv_t^2`, while the demo code normalizes inventory
    by the trade unit. We use lot-normalized inventory so the paper's stated
    `zeta = 0.01` is numerically meaningful with 100-share trade units.
    """

    inventory_lots = inventory / PAPER.minimum_trade_unit
    return float(zeta * inventory_lots * inventory_lots)


def hybrid_reward(
    delta_pnl: float,
    mid_price: float,
    trade_price: float,
    trade_volume: int,
    inventory: int,
    *,
    eta: float = PAPER.eta_dampened_pnl,
    zeta: float = PAPER.zeta_inventory_penalty,
) -> RewardBreakdown:
    """Equation 16: `R_t = DP_t + TP_t - IP_t`."""

    dp = dampened_pnl(delta_pnl, eta)
    tp = trading_pnl(mid_price, trade_price, trade_volume)
    ip = inventory_penalty(inventory, zeta)
    return RewardBreakdown(
        dampened_pnl=dp,
        trading_pnl=tp,
        inventory_penalty=ip,
        reward=dp + tp - ip,
    )
