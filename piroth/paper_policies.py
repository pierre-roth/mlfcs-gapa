from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np

from .baselines import AvellanedaStoikovCalibration
from .paper_env import PaperAction, PaperTradingEnv


class FixedLevelPaperPolicy:
    uses_state = False

    def __init__(self, level: int = 1) -> None:
        self.level = level
        self.name = f"Fixed_{level}"

    def act(self, state, env: PaperTradingEnv) -> PaperAction:
        quote_idx = max(env.event_idx - env.config.latency, 0)
        ask = float(env.day.ask.iloc[quote_idx][f"ask{self.level}_price"])
        bid = float(env.day.bid.iloc[quote_idx][f"bid{self.level}_price"])
        lot = env.config.trade_unit
        return PaperAction(ask_price=ask, ask_volume=-lot, bid_price=bid, bid_volume=lot)


class RandomLevelPaperPolicy:
    name = "Random"
    uses_state = False

    def act(self, state, env: PaperTradingEnv) -> PaperAction:
        quote_idx = max(env.event_idx - env.config.latency, 0)
        key = (env.day.day, env.episode_index, env.event_idx, env.rng_seed)
        digest = hashlib.blake2b(json.dumps(key).encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "big", signed=False) % (2**32)
        rng = np.random.default_rng(seed)
        ask_level = int(rng.integers(1, 6))
        bid_level = int(rng.integers(1, 6))
        ask = float(env.day.ask.iloc[quote_idx][f"ask{ask_level}_price"])
        bid = float(env.day.bid.iloc[quote_idx][f"bid{bid_level}_price"])
        lot = env.config.trade_unit
        return PaperAction(ask_price=ask, ask_volume=-lot, bid_price=bid, bid_volume=lot)


class AvellanedaStoikovPaperPolicy:
    name = "AS"
    uses_state = False

    def __init__(self, calibration: AvellanedaStoikovCalibration) -> None:
        self.calibration = calibration

    def act(self, state, env: PaperTradingEnv) -> PaperAction:
        quote_idx = max(env.event_idx - env.config.latency, 0)
        mid = float(env.day.price.iloc[quote_idx]["midprice"])
        tau = max(env.episode_stop - env.event_idx, 1)
        gamma = env.config.as_gamma
        sigma2 = self.calibration.sigma2_event
        kappa = max(self.calibration.kappa, 1e-6)
        inventory_units = env.inventory / max(env.config.trade_unit, 1)
        reservation = mid - inventory_units * gamma * sigma2 * tau * mid
        total_spread_ticks = gamma * sigma2 * tau + (2.0 / gamma) * np.log1p(gamma / kappa)
        half_spread = max(env.config.symbol_spec.tick_size, 0.5 * total_spread_ticks * env.config.symbol_spec.tick_size)
        ask = _round_up(reservation + half_spread, env.config.symbol_spec.tick_size)
        bid = _round_down(reservation - half_spread, env.config.symbol_spec.tick_size)
        if ask <= bid:
            ask = bid + env.config.symbol_spec.tick_size
        lot = env.config.trade_unit
        return PaperAction(ask_price=ask, ask_volume=-lot, bid_price=bid, bid_volume=lot)


class ContinuousActionPolicy:
    name = "ContinuousAction"

    def __init__(self, action: np.ndarray | list[float]) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def act(self, state, env: PaperTradingEnv) -> PaperAction:
        quote_idx = max(env.event_idx - env.config.latency, 0)
        mid = float(env.day.price.iloc[quote_idx]["midprice"])
        raw_bias = float(np.clip(self.action[0], -1.0, 1.0))
        raw_spread = float(np.clip(self.action[1], -1.0, 1.0))
        if env.config.continuous_action_mode == "author":
            action_bias = (raw_bias + 1.0) / 2.0
            action_spread = (raw_spread + 1.0) / 2.0
            bias = action_bias * env.config.max_bias
            spread = action_spread * env.config.max_spread
            if env.inventory > 0:
                reservation = mid - bias
            elif env.inventory < 0:
                reservation = mid + bias
            else:
                reservation = mid
        elif env.config.continuous_action_mode == "author_raw":
            bias = raw_bias * env.config.max_bias
            spread = raw_spread * env.config.max_spread
            if env.inventory > 0:
                reservation = mid - bias
            elif env.inventory < 0:
                reservation = mid + bias
            else:
                reservation = mid
        elif env.config.continuous_action_mode == "bounded":
            bias = raw_bias * env.config.max_bias
            min_spread = env.config.symbol_spec.tick_size
            max_spread = max(env.config.max_spread, min_spread)
            spread = min_spread + ((raw_spread + 1.0) / 2.0) * (max_spread - min_spread)
            if env.inventory > 0:
                reservation = mid - abs(bias)
            elif env.inventory < 0:
                reservation = mid + abs(bias)
            else:
                reservation = mid + bias
        else:
            raise ValueError(f"Unknown continuous_action_mode: {env.config.continuous_action_mode}")
        ask = _round_up(reservation + spread / 2.0, env.config.symbol_spec.tick_size)
        bid = _round_down(reservation - spread / 2.0, env.config.symbol_spec.tick_size)
        if ask <= bid:
            ask = bid + env.config.symbol_spec.tick_size
        lot = env.config.trade_unit
        return PaperAction(ask_price=ask, ask_volume=-lot, bid_price=bid, bid_volume=lot)


class DiscreteActionPolicy:
    name = "DiscreteAction"

    def __init__(self, action: int) -> None:
        self.action = int(action)

    def act(self, state, env: PaperTradingEnv) -> PaperAction:
        quote_idx = max(env.event_idx - env.config.latency, 0)
        ask1 = float(env.day.price.iloc[quote_idx]["ask1_price"])
        bid1 = float(env.day.price.iloc[quote_idx]["bid1_price"])
        tick = env.config.symbol_spec.tick_size
        lot = env.config.trade_unit
        action = self.action
        if action == 7:
            if env.inventory < 0:
                return PaperAction(ask_price=0.0, ask_volume=0, bid_price=np.inf, bid_volume=-env.inventory)
            if env.inventory > 0:
                return PaperAction(ask_price=tick, ask_volume=-env.inventory, bid_price=0.0, bid_volume=0)
            return PaperAction(ask_price=0.0, ask_volume=0, bid_price=0.0, bid_volume=0)
        ask_offset, bid_offset = _dqn_discrete_offsets(env.config.dqn_discrete_offset_pairs).get(action, (0, 0))
        ask_volume = -lot
        bid_volume = lot
        inventory_limit = env.config.max_inventory_units * lot
        if env.inventory < -inventory_limit:
            ask_volume = 0
        elif env.inventory > inventory_limit:
            bid_volume = 0
        return PaperAction(
            ask_price=ask1 + ask_offset * tick,
            ask_volume=ask_volume,
            bid_price=bid1 - bid_offset * tick,
            bid_volume=bid_volume,
        )


@lru_cache(maxsize=32)
def _dqn_discrete_offsets(raw_offsets: str) -> dict[int, tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw_pair in raw_offsets.split(","):
        item = raw_pair.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid dqn_discrete_offset_pairs item {item!r}; expected ask_ticks:bid_ticks")
        raw_ask, raw_bid = item.split(":", 1)
        ask_offset = int(raw_ask)
        bid_offset = int(raw_bid)
        if ask_offset < 0 or bid_offset < 0:
            raise ValueError("dqn_discrete_offset_pairs offsets must be non-negative")
        pairs.append((ask_offset, bid_offset))
    if len(pairs) != 7:
        raise ValueError("dqn_discrete_offset_pairs must define exactly 7 quote actions")
    return {action: pair for action, pair in enumerate(pairs)}


def _round_up(value: float, tick_size: float) -> float:
    return float(np.ceil(value / tick_size) * tick_size)


def _round_down(value: float, tick_size: float) -> float:
    return float(np.floor(value / tick_size) * tick_size)
