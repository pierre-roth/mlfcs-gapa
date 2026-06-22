"""AS-guided quoting utilities for extension experiments.

This module intentionally depends on the paper replication primitives but does
not modify them. It translates Avellaneda-Stoikov quotes into the continuous
paper action coordinates so extension agents can imitate or stay close to AS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from mlfcs_gapa.env.baselines import AvellanedaStoikovStrategy, estimate_episode_volatility
from mlfcs_gapa.env.replay import Account, HistoricalReplay
from mlfcs_gapa.paper.constants import PAPER
from mlfcs_gapa.data.schema import LobDataset


ASGuidanceMode = Literal["none", "soft", "hard"]
ASBaseReward = Literal["paper_hybrid", "profit"]
ASPenaltyNorm = Literal["l2", "l1", "huber", "adaptive_l2"]
ASPenaltySpace = Literal["action", "quote"]
ASPenaltySchedule = Literal["constant", "episode_decay", "episode_warmup"]


@dataclass(frozen=True)
class ASGuidanceConfig:
    """Configuration for keeping learned quotes close to AS behavior.

    Values are expressed in the paper action space `[0, 1]^2`, where action
    component 0 is inventory bias and component 1 is quoted spread.
    """

    mode: ASGuidanceMode = "none"
    soft_penalty: float = 0.0
    hard_window_bias: float = 0.10
    hard_window_spread: float = 0.10
    max_bias: float = PAPER.max_bias
    max_spread: float = PAPER.max_spread
    base_reward: ASBaseReward = "paper_hybrid"
    bias_weight: float = 1.0
    spread_weight: float = 1.0
    penalty_norm: ASPenaltyNorm = "l2"
    penalty_space: ASPenaltySpace = "action"
    soft_penalty_end: float | None = None
    penalty_schedule: ASPenaltySchedule = "constant"
    huber_delta: float = 0.10
    adaptive_target: float = 0.15


def make_as_strategy(
    dataset: LobDataset,
    *,
    episode_events: int = PAPER.episode_events,
    gamma: float = 1.0,
    kappa: float = 100.0,
) -> AvellanedaStoikovStrategy:
    """Build the AS teacher calibrated to the dataset's episode volatility."""

    sigma = max(estimate_episode_volatility(dataset, episode_events), 1e-6)
    return AvellanedaStoikovStrategy(sigma=sigma, gamma=gamma, kappa=kappa)


def as_teacher_action(
    strategy: AvellanedaStoikovStrategy,
    replay: HistoricalReplay,
    account: Account,
    decision_index: int,
    episode_progress: float,
    *,
    max_bias: float = PAPER.max_bias,
    max_spread: float = PAPER.max_spread,
) -> np.ndarray:
    """Return the AS teacher quote as a paper continuous action in `[0, 1]^2`."""

    mid_price = replay.mid_price(decision_index)
    quote = strategy.quote(replay, account, decision_index, episode_progress)
    return quote_to_paper_action(
        mid_price=mid_price,
        inventory=account.inventory,
        reservation_price=quote.reservation_price,
        spread=quote.spread,
        max_bias=max_bias,
        max_spread=max_spread,
    )


def quote_to_paper_action(
    *,
    mid_price: float,
    inventory: int,
    reservation_price: float,
    spread: float,
    max_bias: float = PAPER.max_bias,
    max_spread: float = PAPER.max_spread,
) -> np.ndarray:
    """Invert the paper action-to-quote equations as closely as possible.

    For zero inventory, the paper's bias action has no effect because
    `sign(inventory) == 0`; in that state the teacher bias is set to zero.
    """

    if max_bias <= 0.0:
        raise ValueError("max_bias must be positive")
    if max_spread <= 0.0:
        raise ValueError("max_spread must be positive")

    sign = np.sign(inventory)
    if sign > 0:
        delta = mid_price - reservation_price
    elif sign < 0:
        delta = reservation_price - mid_price
    else:
        delta = 0.0

    action_bias = np.clip(delta / max_bias, 0.0, 1.0)
    action_spread = np.clip(spread / max_spread, 0.0, 1.0)
    return np.asarray([action_bias, action_spread], dtype=np.float32)


def paper_action_to_env_action(paper_action: np.ndarray, *, normalize_actions: bool) -> np.ndarray:
    """Convert paper `[0, 1]` actions to the environment action space."""

    action = np.asarray(paper_action, dtype=np.float32)
    if normalize_actions:
        return (2.0 * action - 1.0).astype(np.float32)
    return action


def env_action_to_paper_action(action: np.ndarray, *, normalize_actions: bool) -> np.ndarray:
    """Convert environment actions to paper `[0, 1]` coordinates."""

    action = np.asarray(action, dtype=np.float64)
    if normalize_actions:
        action = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
    return np.clip(action, 0.0, 1.0).astype(np.float32)


def apply_hard_as_window(
    paper_action: np.ndarray,
    teacher_action: np.ndarray,
    *,
    hard_window_bias: float,
    hard_window_spread: float,
) -> np.ndarray:
    """Clip a learner action into a rectangular window around AS."""

    action = np.asarray(paper_action, dtype=np.float32)
    teacher = np.asarray(teacher_action, dtype=np.float32)
    lower = teacher - np.asarray([hard_window_bias, hard_window_spread], dtype=np.float32)
    upper = teacher + np.asarray([hard_window_bias, hard_window_spread], dtype=np.float32)
    return np.clip(action, lower, upper).clip(0.0, 1.0).astype(np.float32)


def as_divergence_penalty(
    paper_action: np.ndarray,
    teacher_action: np.ndarray,
    *,
    soft_penalty: float,
    bias_weight: float = 1.0,
    spread_weight: float = 1.0,
    penalty_norm: ASPenaltyNorm = "l2",
    huber_delta: float = 0.10,
    adaptive_target: float = 0.15,
) -> float:
    """Penalty for deviating from AS in paper action coordinates."""

    if soft_penalty <= 0.0:
        return 0.0
    weights = np.asarray([bias_weight, spread_weight], dtype=np.float64)
    diff = (
        np.asarray(paper_action, dtype=np.float64)
        - np.asarray(teacher_action, dtype=np.float64)
    ) * weights
    return _scaled_penalty(
        diff,
        soft_penalty=soft_penalty,
        penalty_norm=penalty_norm,
        huber_delta=huber_delta,
        adaptive_target=adaptive_target,
    )


def quote_divergence_penalty(
    quote_prices: np.ndarray,
    teacher_quote_prices: np.ndarray,
    *,
    soft_penalty: float,
    scale: float,
    penalty_norm: ASPenaltyNorm = "l2",
    huber_delta: float = 0.10,
    adaptive_target: float = 0.15,
) -> float:
    """Penalty for deviating from AS bid/ask prices."""

    if soft_penalty <= 0.0:
        return 0.0
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    diff = (
        np.asarray(quote_prices, dtype=np.float64)
        - np.asarray(teacher_quote_prices, dtype=np.float64)
    ) / scale
    return _scaled_penalty(
        diff,
        soft_penalty=soft_penalty,
        penalty_norm=penalty_norm,
        huber_delta=huber_delta,
        adaptive_target=adaptive_target,
    )


def _scaled_penalty(
    diff: np.ndarray,
    *,
    soft_penalty: float,
    penalty_norm: ASPenaltyNorm,
    huber_delta: float,
    adaptive_target: float,
) -> float:
    if penalty_norm == "l2":
        value = float(np.dot(diff, diff))
        scale = soft_penalty
    elif penalty_norm == "l1":
        value = float(np.abs(diff).sum())
        scale = soft_penalty
    elif penalty_norm == "huber":
        delta = max(float(huber_delta), 1e-8)
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        value = float((0.5 * quadratic * quadratic + delta * linear).sum())
        scale = soft_penalty
    elif penalty_norm == "adaptive_l2":
        norm = float(np.sqrt(np.dot(diff, diff)))
        target = max(float(adaptive_target), 1e-8)
        value = float(np.dot(diff, diff))
        scale = soft_penalty * float(np.clip(norm / target, 0.25, 4.0))
    else:
        raise ValueError("unknown AS penalty norm")
    return float(scale * value)
