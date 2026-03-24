from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RLTrainConfig
from .data import DayData
from .metrics import EpisodeResult
from .utils import price_legal_check


@dataclass
class Observation:
    lob: np.ndarray | None
    flat: np.ndarray


class MarketMakingEnv:
    def __init__(
        self,
        day: DayData,
        config: RLTrainConfig,
        state_mode: str = "full",
        wo_lob_state: bool = False,
        wo_dynamic_state: bool = False,
        reward_mode: str = "hybrid",
    ) -> None:
        self.day = day
        self.config = config
        self.state_mode = state_mode
        self.wo_lob_state = wo_lob_state
        self.wo_dynamic_state = wo_dynamic_state
        self.reward_mode = reward_mode
        self.decision_indices = np.arange(config.lookback - 1 + config.latency, len(day.midprice), dtype=np.int64)
        if len(self.decision_indices) == 0:
            raise RuntimeError(f"No tradable indices for {day.symbol} {day.day}")
        self.num_discrete_actions = 8

    def _episode_ranges(self) -> list[tuple[int, int]]:
        length = self.config.episode_length
        segments = []
        for start in range(0, len(self.decision_indices), length):
            end = min(start + length, len(self.decision_indices))
            if end - start > 4:
                segments.append((start, end))
        return segments

    def available_episodes(self) -> list[tuple[int, int]]:
        return self._episode_ranges()

    def selected_episodes(self, limit: int | None) -> list[tuple[int, int]]:
        episodes = self.available_episodes()
        if limit is None or limit >= len(episodes):
            return episodes
        indices = np.linspace(0, len(episodes) - 1, num=limit, dtype=np.int64)
        return [episodes[int(idx)] for idx in indices]

    def reset(self, episode_span: tuple[int, int]) -> Observation:
        self.episode_span = episode_span
        self.episode_decisions = self.decision_indices[episode_span[0] : episode_span[1]]
        self.step_cursor = 0
        self.inventory = 0
        self.cash = 0.0
        self.value = 0.0
        self.value_prev = 0.0
        self.turnover = 0.0
        self.trades = 0
        self.rewards = 0.0
        self.quote_spreads: list[float] = []
        self.inventory_history: list[float] = []
        self.step_logs: list[dict[str, float]] = []
        return self._build_observation(self.episode_decisions[self.step_cursor] - self.config.latency)

    def _build_flat_features(self, data_idx: int) -> np.ndarray:
        time_ratio = self.step_cursor / max(len(self.episode_decisions), 1)
        agent = np.array([self.inventory / max(self.config.max_inventory * self.config.trade_unit, 1), time_ratio], dtype=np.float32)
        if self.state_mode == "inventory_only":
            return agent
        if self.state_mode == "handcrafted":
            return np.concatenate([self.day.handcrafted[data_idx], agent]).astype(np.float32)
        dynamic = np.zeros(0, dtype=np.float32) if self.wo_dynamic_state else self.day.dynamic[data_idx]
        return np.concatenate([dynamic, agent]).astype(np.float32)

    def _build_observation(self, data_idx: int) -> Observation:
        lob = None
        if not self.wo_lob_state and self.state_mode == "full":
            start = data_idx - self.config.lookback + 1
            lob = self.day.normalized_lob[start : data_idx + 1]
        return Observation(lob=lob, flat=self._build_flat_features(data_idx))

    def _level_volume(self, event_idx: int, side: str, price: float) -> float:
        row = self.day.lob[event_idx]
        for level in range(10):
            base = level * 4
            level_price = row[base] if side == "ask" else row[base + 2]
            level_vol = row[base + 1] if side == "ask" else row[base + 3]
            if abs(level_price - price) < self.config.tick_size / 2:
                return float(level_vol)
        return 0.0

    def _continuous_orders(self, action: np.ndarray, quote_idx: int) -> dict[str, float]:
        mid = float(self.day.midprice[quote_idx])
        bias = float(np.clip(action[0], 0.0, 1.0)) * self.config.max_bias
        spread = max(self.config.tick_size, float(np.clip(action[1], 0.0, 1.0)) * self.config.max_spread)
        sign = 0.0 if self.inventory == 0 else np.sign(self.inventory)
        reservation = mid - sign * bias
        ask_price, bid_price = price_legal_check(reservation + spread / 2, reservation - spread / 2, self.config.tick_size)
        return {
            "ask_price": ask_price,
            "ask_volume": -self.config.trade_unit,
            "bid_price": bid_price,
            "bid_volume": self.config.trade_unit,
            "spread": ask_price - bid_price,
        }

    def _discrete_orders(self, action: int, quote_idx: int) -> dict[str, float]:
        ask = float(self.day.ask1[quote_idx])
        bid = float(self.day.bid1[quote_idx])
        tick = self.config.tick_size
        if action == 0:
            ask_price, bid_price = ask, bid
        elif action == 1:
            ask_price, bid_price = ask, bid - tick
        elif action == 2:
            ask_price, bid_price = ask + tick, bid
        elif action == 3:
            ask_price, bid_price = ask + tick, bid - tick
        elif action == 4:
            ask_price, bid_price = ask, bid - 2 * tick
        elif action == 5:
            ask_price, bid_price = ask + 2 * tick, bid
        elif action == 6:
            ask_price, bid_price = ask + 2 * tick, bid - 2 * tick
        elif action == 7:
            if self.inventory > 0:
                return {"ask_price": float(self.day.bid1[quote_idx]), "ask_volume": -abs(self.inventory), "bid_price": 0.0, "bid_volume": 0.0, "spread": 0.0}
            if self.inventory < 0:
                return {"ask_price": 0.0, "ask_volume": 0.0, "bid_price": float(self.day.ask1[quote_idx]), "bid_volume": abs(self.inventory), "spread": 0.0}
            return {"ask_price": 0.0, "ask_volume": 0.0, "bid_price": 0.0, "bid_volume": 0.0, "spread": 0.0}
        else:
            raise ValueError(f"Unknown discrete action {action}")
        ask_price, bid_price = price_legal_check(ask_price, bid_price, tick)
        return {
            "ask_price": ask_price,
            "ask_volume": -self.config.trade_unit,
            "bid_price": bid_price,
            "bid_volume": self.config.trade_unit,
            "spread": ask_price - bid_price,
        }

    def action_to_orders(self, action: np.ndarray | int | dict[str, float], quote_idx: int) -> dict[str, float]:
        if isinstance(action, dict):
            orders = dict(action)
        elif self.state_mode == "discrete":
            orders = self._discrete_orders(int(action), quote_idx)
        elif np.isscalar(action):
            orders = self._discrete_orders(int(action), quote_idx)
        else:
            orders = self._continuous_orders(np.asarray(action, dtype=np.float32), quote_idx)
        inv_limit = self.config.max_inventory * self.config.trade_unit
        if self.inventory >= inv_limit:
            orders["bid_volume"] = 0.0
        if self.inventory <= -inv_limit:
            orders["ask_volume"] = 0.0
        return orders

    def _match_one_side(self, event_idx: int, side: str, price: float, volume: float) -> list[tuple[float, float]]:
        if volume == 0 or price == 0:
            return []
        trades = self.day.trades_by_index.get(event_idx)
        if trades is None or trades.empty:
            return []
        if "aggressor_side" in trades.columns:
            desired_side = "B" if side == "ask" else "A"
            trades = trades[trades["aggressor_side"].eq(desired_side)]
            if trades.empty:
                return []
        fills: list[tuple[float, float]] = []
        traded_prices = trades["price"].to_numpy(dtype=np.float32)
        traded_sizes = trades["size"].to_numpy(dtype=np.float32)
        if side == "ask":
            book_cross_price = float(self.day.bid1[event_idx])
            signed_volume = -abs(volume)
            if price <= book_cross_price:
                fills.append((book_cross_price, signed_volume))
                return fills
            better = traded_prices > price
            exact = np.isclose(traded_prices, price)
            if np.any(better):
                fills.append((price, signed_volume))
            elif np.any(exact):
                exact_volume = float(traded_sizes[exact].sum())
                depth = self._level_volume(event_idx, "ask", price)
                probability = exact_volume / max(exact_volume + depth, 1e-8)
                if np.random.random() < probability:
                    fills.append((price, signed_volume))
        else:
            book_cross_price = float(self.day.ask1[event_idx])
            signed_volume = abs(volume)
            if price >= book_cross_price:
                fills.append((book_cross_price, signed_volume))
                return fills
            better = traded_prices < price
            exact = np.isclose(traded_prices, price)
            if np.any(better):
                fills.append((price, signed_volume))
            elif np.any(exact):
                exact_volume = float(traded_sizes[exact].sum())
                depth = self._level_volume(event_idx, "bid", price)
                probability = exact_volume / max(exact_volume + depth, 1e-8)
                if np.random.random() < probability:
                    fills.append((price, signed_volume))
        return fills

    def _reward(self, fills: list[tuple[float, float]], midprice: float) -> float:
        pnl_delta = self.value - self.value_prev
        dampened = pnl_delta - max(0.0, self.config.eta * pnl_delta)
        trading = sum(volume * (midprice - price) for price, volume in fills)
        inventory_penalty = self.config.zeta * (self.inventory / max(self.config.trade_unit, 1)) ** 2
        self.value_prev = self.value
        if self.reward_mode == "pnl":
            return float(pnl_delta)
        reward = 0.0
        reward += dampened
        reward += trading
        reward -= inventory_penalty
        return float(reward)

    def step(self, action: np.ndarray | int | dict[str, float]) -> tuple[Observation, float, bool, dict[str, float]]:
        event_idx = int(self.episode_decisions[self.step_cursor])
        quote_idx = max(int(event_idx - self.config.latency), self.config.lookback - 1)
        orders = self.action_to_orders(action, quote_idx)
        fills = []
        fills.extend(self._match_one_side(event_idx, "ask", float(orders["ask_price"]), float(abs(orders["ask_volume"]))))
        fills.extend(self._match_one_side(event_idx, "bid", float(orders["bid_price"]), float(abs(orders["bid_volume"]))))
        self.quote_spreads.append(float(orders.get("spread", 0.0)))
        midprice = float(self.day.midprice[event_idx])

        for price, volume in fills:
            self.inventory += volume
            self.cash -= volume * price
            self.turnover += abs(volume * price)
            self.trades += 1

        self.value = self.cash + self.inventory * midprice
        reward = self._reward(fills, midprice)
        self.rewards += reward
        self.inventory_history.append(float(self.inventory))
        self.step_logs.append(
            {
                "timestamp": self.day.timestamps[event_idx],
                "midprice": midprice,
                "inventory": float(self.inventory),
                "ask_quote": float(orders["ask_price"]),
                "bid_quote": float(orders["bid_price"]),
                "reward": reward,
                "cash": float(self.cash),
                "value": float(self.value),
                "turnover": float(self.turnover),
            }
        )

        self.step_cursor += 1
        done = self.step_cursor >= len(self.episode_decisions)
        if done and self.inventory != 0:
            flatten_price = float(self.day.bid1[event_idx] if self.inventory > 0 else self.day.ask1[event_idx])
            flatten_volume = -self.inventory
            self.inventory += flatten_volume
            self.cash -= flatten_volume * flatten_price
            self.turnover += abs(flatten_volume * flatten_price)
            self.trades += 1
            self.value = self.cash
            flatten_reward = self._reward([(flatten_price, flatten_volume)], midprice)
            reward += flatten_reward
            self.rewards += flatten_reward
        next_obs = self._build_observation(max(quote_idx, self.config.lookback - 1)) if done else self._build_observation(max(int(self.episode_decisions[self.step_cursor] - self.config.latency), self.config.lookback - 1))
        return next_obs, reward, done, {"fills": len(fills), "inventory": float(self.inventory)}

    def episode_result(self, method: str, episode_index: int, latency: int | None = None) -> EpisodeResult:
        avg_spread = float(np.mean([spread for spread in self.quote_spreads if spread > 0])) if self.quote_spreads else 0.0
        avg_position = float(np.mean(self.inventory_history)) if self.inventory_history else 0.0
        avg_abs_position = float(np.mean(np.abs(self.inventory_history))) if self.inventory_history else 0.0
        pnl = float(self.value)
        nd_pnl = pnl / avg_spread if avg_spread > 0 else 0.0
        pnl_map = pnl / avg_abs_position if avg_abs_position > 0 else 0.0
        profit_ratio = pnl / self.turnover if self.turnover > 0 else 0.0
        return EpisodeResult(
            symbol=self.day.symbol,
            day=self.day.day,
            method=method,
            episode_index=episode_index,
            pnl=pnl,
            nd_pnl=nd_pnl,
            pnl_map=pnl_map,
            profit_ratio=profit_ratio,
            avg_position=avg_position,
            avg_abs_position=avg_abs_position,
            avg_spread=avg_spread,
            turnover=float(self.turnover),
            reward=float(self.rewards),
            trades=int(self.trades),
            latency=latency if latency is not None else self.config.latency,
        )

    def episode_trace(self) -> pd.DataFrame:
        if not self.step_logs:
            return pd.DataFrame()
        return pd.DataFrame(self.step_logs)
