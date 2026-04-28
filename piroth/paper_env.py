from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .config import DiagnosticsConfig
from .paper_features import LOB_COLUMNS, PaperState, agent_state, combine_orderbook, dynamic_market_state_matrix, ensure_paper_msg, lob_tensor_from_values
from .simulator import SyntheticDay


@dataclass(frozen=True)
class PaperAction:
    ask_price: float
    ask_volume: int
    bid_price: float
    bid_volume: int


@dataclass
class PaperStepResult:
    state: PaperState | None
    reward: float
    terminal: bool
    info: dict[str, float | int | str]


@dataclass
class EpisodeMetrics:
    policy: str
    day: str
    episode_index: int
    pnl: float
    nd_pnl: float
    pnl_map: float
    profit_ratio: float
    avg_position: float
    avg_abs_position: float
    avg_spread: float
    turnover: float
    trades: int
    fill_rate: float
    reward: float


class PaperPolicy(Protocol):
    name: str

    def act(self, state: PaperState | None, env: PaperTradingEnv) -> PaperAction:
        ...


@dataclass
class PaperTradingEnv:
    day: SyntheticDay
    config: DiagnosticsConfig
    episode_start: int
    episode_stop: int
    episode_index: int = 0
    policy_name: str = "policy"
    rng_seed: int = 0

    orderbook: pd.DataFrame = field(init=False)
    lob_values: np.ndarray = field(init=False)
    price_ts: pd.DataFrame = field(init=False)
    msg_ts: pd.DataFrame = field(init=False)
    market_states: np.ndarray = field(init=False)
    grouped_trades: object = field(init=False)
    trade_indices: list[int] = field(init=False)
    step_pos: int = field(init=False, default=0)
    cash: float = field(init=False, default=0.0)
    value: float = field(init=False, default=0.0)
    previous_value: float = field(init=False, default=0.0)
    inventory: int = field(init=False, default=0)
    previous_inventory: int = field(init=False, default=0)
    turnover: float = field(init=False, default=0.0)
    trades: int = field(init=False, default=0)
    fill_steps: int = field(init=False, default=0)
    cumulative_reward: float = field(init=False, default=0.0)
    inventory_path: list[float] = field(init=False, default_factory=list)
    spread_path: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.orderbook = combine_orderbook(self.day.ask, self.day.bid)
        cached_lob_values = getattr(self.day, "_paper_lob_values", None)
        if cached_lob_values is None or len(cached_lob_values) != len(self.day.price):
            cached_lob_values = self.orderbook[LOB_COLUMNS].to_numpy(dtype=np.float32)
            setattr(self.day, "_paper_lob_values", cached_lob_values)
        self.lob_values = cached_lob_values
        self.price_ts = self.day.price.set_index(pd.to_datetime(self.day.price["timestamp"]))
        self.msg_ts = ensure_paper_msg(self.day.msg).set_index(pd.to_datetime(self.day.msg["timestamp"]))
        cached_market_states = getattr(self.day, "_paper_market_states", None)
        if cached_market_states is None or len(cached_market_states) != len(self.day.price):
            cached_market_states = dynamic_market_state_matrix(self.day.price, self.day.msg)
            setattr(self.day, "_paper_market_states", cached_market_states)
        self.market_states = cached_market_states
        self.grouped_trades = self.day.trades.groupby("timestamp") if not self.day.trades.empty else None
        trade_times = set(self.day.trades["timestamp"]) if not self.day.trades.empty else set()
        self.trade_indices = [
            idx
            for idx in range(max(self.episode_start, self.config.lookback + self.config.latency), self.episode_stop)
            if self.day.price.iloc[idx]["timestamp"] in trade_times
        ]
        if not self.trade_indices:
            self.trade_indices = list(range(max(self.episode_start, self.config.lookback + self.config.latency), self.episode_stop))

    def reset(self) -> PaperState:
        self.step_pos = 0
        self.cash = self.value = self.previous_value = 0.0
        self.inventory = self.previous_inventory = 0
        self.turnover = 0.0
        self.trades = 0
        self.fill_steps = 0
        self.cumulative_reward = 0.0
        self.inventory_path = []
        self.spread_path = []
        return self.state()

    @property
    def event_idx(self) -> int:
        return self.trade_indices[min(self.step_pos, len(self.trade_indices) - 1)]

    def state(self) -> PaperState:
        idx = max(self.event_idx - self.config.latency, self.config.lookback)
        return PaperState(
            lob_state=lob_tensor_from_values(self.lob_values, idx, self.config.lookback),
            market_state=self.market_states[idx],
            agent_state=agent_state(
                inventory=self.inventory,
                event_idx=idx - self.episode_start,
                episode_length=self.episode_stop - self.episode_start,
                lot_size=self.config.trade_unit,
                max_inventory_units=self.config.max_inventory_units,
            ),
        )

    def step(self, action: PaperAction, *, compute_next_state: bool = True) -> PaperStepResult:
        idx = self.event_idx
        self.previous_inventory = self.inventory
        self.previous_value = self.value
        action = self._apply_inventory_guard(action)
        fills = self.match(action, idx)
        trade_price, trade_volume, matched_pnl = self._apply_fills(fills, idx)
        reward = self.reward(trade_price, trade_volume, action, matched_pnl)
        self.cumulative_reward += reward
        self.inventory_path.append(float(self.inventory))
        self.spread_path.append(self._quoted_spread(action))
        self.step_pos += 1
        terminal = self.step_pos >= len(self.trade_indices)
        if terminal:
            self.previous_inventory = self.inventory
            self.previous_value = self.value
            close_price, close_volume = self.close_position(idx)
            close_reward = self.reward(close_price, close_volume, action)
            reward += close_reward
            self.cumulative_reward += close_reward
        next_state = None if terminal or not compute_next_state else self.state()
        return PaperStepResult(
            state=next_state,
            reward=float(reward),
            terminal=terminal,
            info={"trade_price": trade_price, "trade_volume": trade_volume, "event_idx": idx},
        )

    def match(self, action: PaperAction, event_idx: int) -> list[tuple[float, int]]:
        fills: list[tuple[float, int]] = []
        previous_idx = max(event_idx - self.config.latency, 0)
        previous = self.day.price.iloc[previous_idx]
        block = _trades_at(self.grouped_trades, self.day.price.iloc[event_idx]["timestamp"])
        if action.ask_price and action.ask_volume < 0:
            fill = self._match_sell(action.ask_price, action.ask_volume, previous, block, event_idx)
            if fill is not None:
                fills.append(fill)
        if action.bid_price and action.bid_volume > 0:
            fill = self._match_buy(action.bid_price, action.bid_volume, previous, block, event_idx)
            if fill is not None:
                if self.config.matching_mode == "author_single":
                    return [fill]
                fills.append(fill)
        if self.config.matching_mode not in {"author_single", "multi_fill"}:
            raise ValueError(f"Unknown matching_mode: {self.config.matching_mode}")
        return fills

    def reward(self, trade_price: float, trade_volume: int, action: PaperAction, matched_pnl: float | None = None) -> float:
        pnl = self.value - self.previous_value
        if matched_pnl is None:
            matched_pnl = (self.midprice(self.event_idx) - trade_price) * trade_volume if trade_volume else 0.0
        spread = self._quoted_spread(action)
        spread_penalty = 0.0
        if self.inventory == 0 and self._is_two_sided(action) and spread > self.config.reward_spread_penalty_threshold:
            spread_penalty = self.config.reward_spread_penalty_scale * spread
        if self.config.reward_mode == "author_pnl":
            return float(pnl - spread_penalty)
        if self.config.reward_mode != "hybrid":
            raise ValueError(f"Unknown reward_mode: {self.config.reward_mode}")
        dampened_pnl = pnl - max(0.0, self.config.reward_eta * pnl)
        inventory_penalty = self.config.reward_zeta * (self.inventory / self.config.trade_unit) ** 2
        reward = 0.0
        base_pnl = dampened_pnl if self.config.reward_use_dampened_pnl else pnl
        reward += self.config.reward_pnl_weight * base_pnl
        if self.config.reward_use_trading_pnl:
            reward += self.config.reward_trading_pnl_weight * matched_pnl
        if self.config.reward_use_inventory_penalty:
            reward -= self.config.reward_inventory_penalty_weight * inventory_penalty
        reward -= self.config.reward_spread_penalty_weight * spread_penalty
        return float(reward)

    def close_position(self, event_idx: int) -> tuple[float, int]:
        if self.inventory == 0:
            return 0.0, 0
        previous_idx = max(event_idx - self.config.latency, 0)
        price = self.day.price.iloc[previous_idx]
        if self.inventory < 0:
            trade_price = float(price["ask1_price"])
            trade_volume = -self.inventory
        else:
            trade_price = float(price["bid1_price"])
            trade_volume = -self.inventory
        self._update_agent(trade_price, trade_volume, event_idx)
        return trade_price, trade_volume

    def metrics(self) -> EpisodeMetrics:
        avg_spread = float(np.mean(self.spread_path)) if self.spread_path else self.config.symbol_spec.tick_size
        inv = np.asarray(self.inventory_path if self.inventory_path else [0.0], dtype=np.float64)
        pnl = self.value
        return EpisodeMetrics(
            policy=self.policy_name,
            day=self.day.day,
            episode_index=self.episode_index,
            pnl=float(pnl),
            nd_pnl=float(pnl / max(avg_spread, self.config.symbol_spec.tick_size)),
            pnl_map=float(pnl / max(np.mean(np.abs(inv)), 1.0)),
            profit_ratio=float(pnl / max(self.turnover, 1e-8)),
            avg_position=float(np.mean(inv)),
            avg_abs_position=float(np.mean(np.abs(inv))),
            avg_spread=avg_spread,
            turnover=float(self.turnover),
            trades=int(self.trades),
            fill_rate=float(self.fill_steps / max(len(self.trade_indices), 1)),
            reward=float(self.cumulative_reward),
        )

    def midprice(self, event_idx: int) -> float:
        return float(self.day.price.iloc[event_idx]["midprice"])

    def _match_sell(self, ask_price: float, ask_volume: int, previous: pd.Series, block: pd.DataFrame, event_idx: int) -> tuple[float, int] | None:
        if ask_price <= float(previous["bid1_price"]):
            return float(previous["bid1_price"]), ask_volume
        buys = block[block["aggressor_side"] == "B"] if not block.empty else block
        if buys.empty:
            return None
        best_trade = float(buys["price"].max())
        if best_trade > ask_price:
            return ask_price, ask_volume
        if np.isclose(best_trade, ask_price):
            traded = int(buys.loc[np.isclose(buys["price"], ask_price), "size"].sum())
            depth = _depth_at(self.day.ask.iloc[event_idx], "ask", ask_price)
            if _stable_uniform((self.day.day, self.episode_index, event_idx, "ask", ask_price, self.rng_seed)) < traded / max(traded + depth, 1):
                return ask_price, ask_volume
        return None

    def _match_buy(self, bid_price: float, bid_volume: int, previous: pd.Series, block: pd.DataFrame, event_idx: int) -> tuple[float, int] | None:
        if bid_price >= float(previous["ask1_price"]):
            return float(previous["ask1_price"]), bid_volume
        sells = block[block["aggressor_side"] == "A"] if not block.empty else block
        if sells.empty:
            return None
        best_trade = float(sells["price"].min())
        if best_trade < bid_price:
            return bid_price, bid_volume
        if np.isclose(best_trade, bid_price):
            traded = int(sells.loc[np.isclose(sells["price"], bid_price), "size"].sum())
            depth = _depth_at(self.day.bid.iloc[event_idx], "bid", bid_price)
            if _stable_uniform((self.day.day, self.episode_index, event_idx, "bid", bid_price, self.rng_seed)) < traded / max(traded + depth, 1):
                return bid_price, bid_volume
        return None

    def _apply_fills(self, fills: list[tuple[float, int]], event_idx: int) -> tuple[float, int, float]:
        if not fills:
            self.value = self.cash + self.inventory * self.midprice(event_idx)
            return 0.0, 0, 0.0
        gross_volume = sum(abs(volume) for _, volume in fills)
        net_volume = sum(volume for _, volume in fills)
        avg_price = sum(price * abs(volume) for price, volume in fills) / max(gross_volume, 1)
        mid = self.midprice(event_idx)
        matched_pnl = 0.0
        for trade_price, trade_volume in fills:
            rebate = abs(trade_volume) * self.config.maker_rebate_per_share
            matched_pnl += (mid - trade_price) * trade_volume + rebate
            self._update_agent(trade_price, trade_volume, event_idx, count_fill_step=False, maker_rebate=rebate)
        self.fill_steps += 1
        return float(avg_price), int(net_volume), float(matched_pnl)

    def _update_agent(
        self,
        trade_price: float,
        trade_volume: int,
        event_idx: int,
        *,
        count_fill_step: bool = True,
        maker_rebate: float = 0.0,
    ) -> None:
        if trade_volume:
            self.inventory += trade_volume
            self.cash -= trade_volume * trade_price
            self.cash += maker_rebate
            self.turnover += abs(trade_volume * trade_price)
            self.trades += 1
            if count_fill_step:
                self.fill_steps += 1
        self.value = self.cash + self.inventory * self.midprice(event_idx)

    def _apply_inventory_guard(self, action: PaperAction) -> PaperAction:
        max_inventory = self.config.max_inventory_units * self.config.trade_unit
        ask_volume = action.ask_volume
        bid_volume = action.bid_volume
        ask_price = action.ask_price
        bid_price = action.bid_price
        if self.inventory <= -max_inventory:
            ask_price = 0.0
            ask_volume = 0
        if self.inventory >= max_inventory:
            bid_price = 0.0
            bid_volume = 0
        return PaperAction(ask_price=ask_price, ask_volume=ask_volume, bid_price=bid_price, bid_volume=bid_volume)

    def _quoted_spread(self, action: PaperAction) -> float:
        if not self._is_two_sided(action):
            return self.config.symbol_spec.tick_size
        return float(max(action.ask_price - action.bid_price, self.config.symbol_spec.tick_size))

    @staticmethod
    def _is_two_sided(action: PaperAction) -> bool:
        return bool(action.ask_price > 0.0 and action.bid_price > 0.0 and np.isfinite(action.ask_price) and np.isfinite(action.bid_price))


def run_episode(env: PaperTradingEnv, policy: PaperPolicy) -> EpisodeMetrics:
    env.policy_name = policy.name
    uses_state = bool(getattr(policy, "uses_state", True))
    state = env.reset() if uses_state else None
    terminal = False
    while not terminal:
        result = env.step(policy.act(state, env), compute_next_state=uses_state)
        terminal = result.terminal
        if uses_state and result.state is not None:
            state = result.state
    return env.metrics()


def _trades_at(grouped_trades, timestamp: pd.Timestamp) -> pd.DataFrame:
    if grouped_trades is None or timestamp not in grouped_trades.groups:
        return pd.DataFrame(columns=["price", "size", "aggressor_side"])
    return grouped_trades.get_group(timestamp)


def _depth_at(row: pd.Series, prefix: str, price: float) -> int:
    for level in range(1, 11):
        if np.isclose(float(row[f"{prefix}{level}_price"]), price):
            return int(row[f"{prefix}{level}_volume"])
    return 0


def _stable_uniform(parts: tuple[object, ...]) -> float:
    payload = json.dumps(parts, separators=(",", ":"), default=str).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) / float(1 << 64)
