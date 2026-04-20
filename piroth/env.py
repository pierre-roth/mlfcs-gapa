from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TrainConfig
from .data import DayData
from .utils import price_legal_check


@dataclass
class Observation:
    lob: np.ndarray
    flat: np.ndarray


@dataclass
class Fill:
    price: float
    volume: float
    taker: bool = False


class ContinuousMarketEnv:
    def __init__(self, day: DayData, config: TrainConfig) -> None:
        self.day = day
        self.config = config
        self.decision_indices = np.arange(config.lookback - 1 + config.latency, len(day.midprice), dtype=np.int64)
        if len(self.decision_indices) == 0:
            raise RuntimeError(f"No tradable indices for {day.symbol} {day.day}")
        self.eval_episode_index: int | None = None
        self.eval_context_key: str | None = None

    def available_episodes(self) -> list[tuple[int, int]]:
        spans = []
        for start in range(0, len(self.decision_indices), self.config.episode_length):
            end = min(start + self.config.episode_length, len(self.decision_indices))
            if end - start > 4:
                spans.append((start, end))
        return spans

    def selected_episodes(self, limit: int | None) -> list[tuple[int, int]]:
        episodes = self.available_episodes()
        if limit is None or limit >= len(episodes):
            return episodes
        idx = np.linspace(0, len(episodes) - 1, num=limit, dtype=np.int64)
        return [episodes[int(i)] for i in idx]

    def set_eval_context(self, episode_index: int | None) -> None:
        self.eval_episode_index = episode_index

    def _fill_draw(self, event_idx: int, side: str, price: float) -> float:
        if self.eval_context_key is None:
            return float(np.random.random())
        tick = int(round(price / max(self.config.tick_size, 1e-8)))
        key = f"{self.eval_context_key}|{event_idx}|{side}|{tick}"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)

    def reset(self, span: tuple[int, int]) -> Observation:
        self.episode_span = span
        self.episode_decisions = self.decision_indices[span[0] : span[1]]
        if self.config.deterministic_evaluation and self.eval_episode_index is not None:
            self.eval_context_key = "|".join([self.day.symbol, self.day.day, str(self.eval_episode_index), str(self.config.eval_seed_base)])
        else:
            self.eval_context_key = None
        self.cursor = 0
        self.cash = 0.0
        self.value = 0.0
        self.prev_value = 0.0
        self.inventory = 0.0
        self.turnover = 0.0
        self.trades = 0
        self.fill_steps = 0
        self.rewards = 0.0
        self.quote_spreads: list[float] = []
        self.quote_biases: list[float] = []
        self.inventory_history: list[float] = []
        self.logs: list[dict[str, float | int | str | pd.Timestamp]] = []
        return self._build_observation(self.episode_decisions[self.cursor] - self.config.latency)

    def _build_observation(self, idx: int) -> Observation:
        start = idx - self.config.lookback + 1
        lob = self.day.normalized_lob[start : idx + 1]
        flat = np.concatenate([self.day.dynamic[idx], self._agent_state(idx)]).astype(np.float32)
        return Observation(lob=lob, flat=flat)

    def _agent_state(self, idx: int) -> np.ndarray:
        inv_scaled = self.inventory / max(self.config.max_inventory_units * self.config.trade_unit, 1)
        remaining = 1.0 - float(self.cursor) / max(len(self.episode_decisions), 1)
        return np.asarray([inv_scaled] * 12 + [remaining] * 12, dtype=np.float32)

    def action_to_orders(self, action: np.ndarray | dict[str, float], quote_idx: int) -> dict[str, float]:
        if isinstance(action, dict):
            return dict(action)
        action = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        mid = float(self.day.midprice[quote_idx])
        delta = float(action[0]) * self.config.max_bias
        reservation = mid - np.sign(self.inventory) * delta
        spread = max(self.config.tick_size, float(action[1]) * self.config.max_spread)
        ask, bid = price_legal_check(reservation + spread / 2.0, reservation - spread / 2.0, self.config.tick_size)
        ask_volume = -float(self.config.trade_unit)
        bid_volume = float(self.config.trade_unit)
        inv_limit = self.config.max_inventory_units * self.config.trade_unit
        if self.inventory >= inv_limit:
            bid_volume = 0.0
        if self.inventory <= -inv_limit:
            ask_volume = 0.0
        return {
            "ask_price": ask,
            "ask_volume": ask_volume,
            "bid_price": bid,
            "bid_volume": bid_volume,
            "spread": ask - bid,
            "reservation": reservation,
        }

    def _level_volume(self, event_idx: int, side: str, price: float) -> float:
        row = self.day.lob[event_idx]
        for level in range(10):
            base = level * 4
            level_price = row[base] if side == "ask" else row[base + 2]
            level_volume = row[base + 1] if side == "ask" else row[base + 3]
            if abs(level_price - price) < self.config.tick_size / 2:
                return float(level_volume)
        return 0.0

    def _match_side(self, event_idx: int, side: str, price: float, volume: float) -> list[Fill]:
        if volume == 0 or price == 0:
            return []
        trades = self.day.trades_by_index.get(event_idx)
        if trades is None or trades.price.size == 0:
            return []
        desired_side = "B" if side == "ask" else "A"
        side_mask = trades.aggressor_side == desired_side
        if not np.any(side_mask):
            return []
        trade_prices = trades.price[side_mask]
        trade_sizes = trades.size[side_mask]
        if side == "ask":
            cross = float(self.day.bid1[event_idx])
            signed = -abs(volume)
            if price <= cross:
                return [Fill(cross, signed, taker=True)]
            better = trade_prices > price
            exact = np.isclose(trade_prices, price)
            if np.any(better):
                return [Fill(price, signed)]
            if np.any(exact):
                exact_size = float(trade_sizes[exact].sum())
                depth = self._level_volume(event_idx, "ask", price)
                probability = exact_size / max(exact_size + depth, 1e-8)
                if self._fill_draw(event_idx, side, price) < probability:
                    return [Fill(price, signed)]
        else:
            cross = float(self.day.ask1[event_idx])
            signed = abs(volume)
            if price >= cross:
                return [Fill(cross, signed, taker=True)]
            better = trade_prices < price
            exact = np.isclose(trade_prices, price)
            if np.any(better):
                return [Fill(price, signed)]
            if np.any(exact):
                exact_size = float(trade_sizes[exact].sum())
                depth = self._level_volume(event_idx, "bid", price)
                probability = exact_size / max(exact_size + depth, 1e-8)
                if self._fill_draw(event_idx, side, price) < probability:
                    return [Fill(price, signed)]
        return []

    def _inventory_penalty(self) -> float:
        base = self.config.zeta * (self.inventory / max(self.config.trade_unit, 1)) ** 2
        return float(self.config.inventory_penalty_weight * base)

    def _maker_rebate(self, fill: Fill) -> float:
        if not self.config.use_maker_rebate or fill.taker:
            return 0.0
        return float(abs(fill.volume) * self.config.maker_rebate_per_share)

    def _apply_fill(self, fill: Fill) -> None:
        self.inventory += fill.volume
        self.cash -= fill.volume * fill.price
        self.cash += self._maker_rebate(fill)
        self.turnover += abs(fill.volume * fill.price)
        self.trades += 1

    def _reward(self, fills: list[Fill], mid: float) -> float:
        delta_pnl = self.value - self.prev_value
        dampened = self.config.dampened_pnl_weight * (delta_pnl - max(0.0, self.config.eta * delta_pnl))
        trading = float(sum(fill.volume * (mid - fill.price) for fill in fills))
        trading *= self.config.trade_reward_weight
        return float(dampened + trading - self._inventory_penalty())

    def _close_position(self, event_idx: int) -> float:
        if self.inventory == 0:
            return 0.0
        mid = float(self.day.midprice[event_idx])
        flatten_price = float(self.day.bid1[event_idx] if self.inventory > 0 else self.day.ask1[event_idx])
        fill = Fill(flatten_price, -self.inventory, taker=True)
        self._apply_fill(fill)
        self.prev_value = self.value
        self.value = self.cash
        return self._reward([fill], mid)

    def step(self, action: np.ndarray | dict[str, float]) -> tuple[Observation, float, bool, dict[str, float]]:
        event_idx = int(self.episode_decisions[self.cursor])
        quote_idx = max(event_idx - self.config.latency, self.config.lookback - 1)
        orders = self.action_to_orders(action, quote_idx)
        fills = []
        fills.extend(self._match_side(event_idx, "ask", float(orders["ask_price"]), abs(float(orders["ask_volume"]))))
        fills.extend(self._match_side(event_idx, "bid", float(orders["bid_price"]), abs(float(orders["bid_volume"]))))
        if fills:
            self.fill_steps += 1
        mid = float(self.day.midprice[event_idx])
        self.quote_spreads.append(float(orders["spread"]))
        self.quote_biases.append(float(orders["reservation"] - mid))
        for fill in fills:
            self._apply_fill(fill)
        self.prev_value = self.value
        self.value = self.cash + self.inventory * mid
        reward = self._reward(fills, mid)
        self.rewards += reward
        self.inventory_history.append(float(self.inventory))
        self.logs.append(
            {
                "timestamp": self.day.timestamps[event_idx],
                "midprice": mid,
                "inventory": float(self.inventory),
                "ask_quote": float(orders["ask_price"]),
                "bid_quote": float(orders["bid_price"]),
                "reservation": float(orders["reservation"]),
                "quote_bias": float(orders["reservation"] - mid),
                "spread": float(orders["spread"]),
                "latent_alpha": float(self.day.latent.iloc[event_idx]["latent_alpha"]),
                "regime_shift": int(self.day.latent.iloc[event_idx]["regime_shift"]),
                "reward": float(reward),
            }
        )
        self.cursor += 1
        done = self.cursor >= len(self.episode_decisions)
        if done:
            terminal_reward = self._close_position(event_idx)
            reward += terminal_reward
            self.rewards += terminal_reward
            self.logs[-1]["reward"] = float(reward)
        next_idx = self.episode_decisions[min(self.cursor, len(self.episode_decisions) - 1)] - self.config.latency
        obs = self._build_observation(max(int(next_idx), self.config.lookback - 1))
        return obs, float(reward), done, {"inventory": float(self.inventory)}

    def episode_trace(self) -> pd.DataFrame:
        return pd.DataFrame(self.logs)

    def episode_result(self, method: str, episode_index: int) -> dict[str, float | int | str]:
        avg_spread = float(np.mean(self.quote_spreads)) if self.quote_spreads else 0.0
        inv = np.asarray(self.inventory_history or [0.0], dtype=np.float64)
        pnl = float(self.value)
        return {
            "symbol": self.day.symbol,
            "day": self.day.day,
            "method": method,
            "episode_index": episode_index,
            "pnl": pnl,
            "nd_pnl": float(pnl / max(avg_spread, self.config.tick_size)),
            "pnl_map": float(pnl / max(np.mean(np.abs(inv)), 1e-8)),
            "profit_ratio": float(pnl / max(self.turnover, 1e-8)),
            "avg_position": float(np.mean(inv)),
            "avg_abs_position": float(np.mean(np.abs(inv))),
            "avg_spread": avg_spread,
            "turnover": float(self.turnover),
            "reward": float(self.rewards),
            "trades": int(self.trades),
            "latency": int(self.config.latency),
            "fill_rate": float(self.fill_steps / max(len(self.episode_decisions), 1)),
            "avg_bias": float(np.mean(self.quote_biases)) if self.quote_biases else 0.0,
        }
