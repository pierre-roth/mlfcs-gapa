from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyrallis

from .config import GenerateConfig
from .utils import ensure_dir, save_json, set_seed


MSG_COLUMNS = [
    "market_buy_volume",
    "market_buy_n",
    "market_sell_volume",
    "market_sell_n",
    "limit_buy_volume",
    "limit_buy_n",
    "limit_sell_volume",
    "limit_sell_n",
    "withdraw_buy_volume",
    "withdraw_buy_n",
    "withdraw_sell_volume",
    "withdraw_sell_n",
]


@dataclass
class SymbolProfile:
    base_price: float
    fair_value_persistence: float
    signal_noise: float
    noise_taker_rate: float
    informed_taker_rate: float
    maker_add_rate: float
    maker_cancel_rate: float
    liquidity_refill_rate: float
    maker_join_touch_prob: float
    depth_scale: float


@dataclass
class RestingOrder:
    order_id: int
    owner: str
    side: str
    price: float
    size: float
    created_event: int


@dataclass
class TradeRecord:
    price: float
    size: float
    aggressor_side: str
    taker_agent: str
    maker_agent: str
    maker_order_id: int
    queue_ahead: float


def _day_label(idx: int) -> str:
    return (pd.Timestamp("2019-11-01") + pd.tseries.offsets.BDay(idx)).strftime("%Y%m%d")


def _stable_seed(seed: int, symbol: str, day: str) -> int:
    digest = hashlib.blake2b(f"{seed}|{symbol}|{day}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (2**32)


def _session_timestamps(anchor_day: pd.Timestamp, session_windows: list[str], event_count: int, rng: np.random.Generator) -> pd.DatetimeIndex:
    windows = []
    for raw in session_windows:
        start, end = raw.split("-", maxsplit=1)
        windows.append((anchor_day + pd.to_timedelta(start), anchor_day + pd.to_timedelta(end)))
    durations = np.asarray([(end - start).total_seconds() for start, end in windows], dtype=np.float64)
    counts = np.floor(event_count * durations / durations.sum()).astype(int)
    counts[-1] += event_count - int(counts.sum())
    parts: list[pd.DatetimeIndex] = []
    for (start, end), count, seconds in zip(windows, counts, durations, strict=True):
        if count <= 0:
            continue
        waits = np.cumsum(rng.gamma(shape=2.0, scale=1.0, size=count))
        waits = waits / waits[-1] * max(seconds - 1e-6, 1e-6)
        ns = np.round(waits * 1_000_000_000).astype(np.int64)
        parts.append(pd.DatetimeIndex(start.value + ns))
    return parts[0].append(parts[1:]).sort_values() if parts else pd.DatetimeIndex([])


def _exp_weights(levels: int, decay: float = 0.55) -> np.ndarray:
    weights = np.exp(-decay * np.arange(levels))
    return weights / weights.sum()


class AgentBasedLOB:
    def __init__(self, config: GenerateConfig, profile: SymbolProfile, rng: np.random.Generator) -> None:
        self.config = config
        self.profile = profile
        self.rng = rng
        self.tick = config.tick_size
        self.trade_unit = float(config.trade_unit)
        self.event_seq = 0
        self.next_order_id = 1

        self.reference_price = round(profile.base_price / self.tick) * self.tick
        self.fair_value = float(profile.base_price)
        self.signal = 0.0
        self.metaorder_bias = 0.0
        self.regime = 0
        self.regime_clock = 0
        self.signed_flow_state = 0.0

        self.bids: dict[float, deque[RestingOrder]] = {}
        self.asks: dict[float, deque[RestingOrder]] = {}
        self._initialize_book()

    def _initialize_book(self) -> None:
        base_bid = np.floor(self.reference_price / self.tick) * self.tick
        for level in range(10):
            bid_price = round(base_bid - level * self.tick, 6)
            ask_price = round(base_bid + (level + 1) * self.tick, 6)
            scale = self.profile.depth_scale * (1.3 - 0.05 * level)
            maker_orders = 3 if level < 3 else 2
            for _ in range(maker_orders):
                self._add_limit("bid", bid_price, self._draw_size(scale), owner="competing_mm", silent=True)
                self._add_limit("ask", ask_price, self._draw_size(scale), owner="competing_mm", silent=True)
            self._add_limit("bid", bid_price, self._draw_size(scale * 1.3), owner="liquidity_provider", silent=True)
            self._add_limit("ask", ask_price, self._draw_size(scale * 1.3), owner="liquidity_provider", silent=True)
        self._ensure_depth()

    def _draw_size(self, scale: float = 1.0) -> float:
        lots = int(self.rng.choice([1, 2, 3, 4, 5], p=[0.42, 0.24, 0.16, 0.10, 0.08]))
        return float(max(self.trade_unit, round(lots * scale) * self.trade_unit))

    def _stress_level(self) -> float:
        edge_ticks = abs(self.fair_value - self.midprice) / max(self.tick, 1e-8)
        flow_stress = abs(self.signed_flow_state) / 12.0
        meta_stress = abs(self.metaorder_bias)
        return float(min(3.0, 0.35 * edge_ticks + 0.4 * flow_stress + 0.8 * meta_stress))

    def _best_price(self, side: str) -> float:
        book = self.bids if side == "bid" else self.asks
        if not book:
            return self.reference_price - self.tick if side == "bid" else self.reference_price + self.tick
        return max(book) if side == "bid" else min(book)

    @property
    def best_bid(self) -> float:
        return self._best_price("bid")

    @property
    def best_ask(self) -> float:
        return self._best_price("ask")

    @property
    def midprice(self) -> float:
        return round((self.best_bid + self.best_ask) / 2.0, 6)

    @property
    def spread_ticks(self) -> int:
        return max(1, int(round((self.best_ask - self.best_bid) / self.tick)))

    def _level_depth(self, side: str, price: float) -> float:
        book = self.bids if side == "bid" else self.asks
        return float(sum(order.size for order in book.get(price, [])))

    def _depth_ladder(self, side: str, depth: int = 10) -> list[tuple[float, float]]:
        book = self.bids if side == "bid" else self.asks
        prices = sorted(book.keys(), reverse=(side == "bid"))
        ladder = [(price, float(sum(order.size for order in book[price]))) for price in prices[:depth]]
        if not ladder:
            anchor = self.reference_price - self.tick if side == "bid" else self.reference_price + self.tick
            for level in range(depth):
                price = round(anchor - level * self.tick, 6) if side == "bid" else round(anchor + level * self.tick, 6)
                ladder.append((price, 0.0))
        while len(ladder) < depth:
            last_price = ladder[-1][0]
            next_price = round(last_price - self.tick, 6) if side == "bid" else round(last_price + self.tick, 6)
            ladder.append((next_price, 0.0))
        return ladder

    def _top_imbalance(self) -> float:
        bid = self._level_depth("bid", self.best_bid)
        ask = self._level_depth("ask", self.best_ask)
        return float((bid - ask) / max(bid + ask, 1e-8))

    def _queue_pressure(self) -> float:
        ask_depth = sum(depth for _, depth in self._depth_ladder("ask", depth=3))
        bid_depth = sum(depth for _, depth in self._depth_ladder("bid", depth=3))
        return float((ask_depth - bid_depth) / max(ask_depth + bid_depth, 1e-8))

    def _add_limit(self, side: str, price: float, size: float, owner: str, silent: bool = False) -> None:
        price = round(price, 6)
        if side == "bid":
            price = min(price, round(self.best_ask - self.tick, 6))
            book = self.bids
        else:
            price = max(price, round(self.best_bid + self.tick, 6))
            book = self.asks
        order = RestingOrder(
            order_id=self.next_order_id,
            owner=owner,
            side=side,
            price=price,
            size=float(size),
            created_event=self.event_seq,
        )
        self.next_order_id += 1
        queue = book.setdefault(price, deque())
        # Cap queue depth per level so per-level iteration stays O(1).
        # Without this, liquidity_provider orders accumulate indefinitely
        # because they are added on nearly every event but rarely consumed.
        if len(queue) < 8:
            queue.append(order)
        if not silent:
            self.event_seq += 1

    def _remove_order(self, side: str, price: float, queue_index: int, remove_size: float) -> RestingOrder | None:
        book = self.bids if side == "bid" else self.asks
        queue = book.get(price)
        if not queue:
            return None
        queue_index = max(0, min(queue_index, len(queue) - 1))
        order = queue[queue_index]
        removed = min(order.size, remove_size)
        order.size -= removed
        removed_order = RestingOrder(order.order_id, order.owner, order.side, order.price, removed, order.created_event)
        if order.size <= 1e-8:
            del queue[queue_index]
        if not queue:
            book.pop(price, None)
        return removed_order

    def _ensure_depth(self) -> None:
        # Extend from the deepest existing level, not from the touch.
        # Using best_bid - tick caused an infinite loop: that level already
        # exists after the first iteration so len(bids) never increases.
        while len(self.bids) < 10:
            deepest = min(self.bids) if self.bids else self.best_ask - self.tick
            price = round(deepest - self.tick, 6)
            self._add_limit("bid", price, self._draw_size(self.profile.depth_scale * 1.25), owner="liquidity_provider", silent=True)
        while len(self.asks) < 10:
            deepest = max(self.asks) if self.asks else self.best_bid + self.tick
            price = round(deepest + self.tick, 6)
            self._add_limit("ask", price, self._draw_size(self.profile.depth_scale * 1.25), owner="liquidity_provider", silent=True)
        # Prune levels far from the touch so max/min over the dict stays fast.
        cutoff = 30 * self.tick
        best_b = self.best_bid
        best_a = self.best_ask
        for p in [p for p in self.bids if p < best_b - cutoff]:
            del self.bids[p]
        for p in [p for p in self.asks if p > best_a + cutoff]:
            del self.asks[p]

    def _step_latent(self) -> tuple[int, float]:
        self.regime_clock += 1
        regime_shift = 0
        if self.regime_clock > 300 and self.rng.random() < 0.004:
            self.regime = int(self.rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3]))
            self.regime_clock = 0
            regime_shift = 1
        if self.rng.random() < self.config.metaorder_start_prob:
            self.metaorder_bias += float(self.rng.choice([-1.0, 1.0]) * self.config.metaorder_scale * self.rng.uniform(0.6, 1.2))
        target = self.config.alpha_signal_scale * 0.4 * self.regime
        self.signal = (
            self.profile.fair_value_persistence * self.signal
            + (1.0 - self.profile.fair_value_persistence) * target
            + self.rng.normal(0.0, self.profile.signal_noise * self.config.alpha_signal_scale)
        )
        self.metaorder_bias = self.config.metaorder_persistence * self.metaorder_bias + self.rng.normal(0.0, 0.015)
        fair_move = (
            self.config.fair_value_signal_scale * (self.signal + self.metaorder_bias)
            + self.rng.normal(0.0, self.config.fair_value_noise_scale)
        )
        if self.rng.random() < self.config.shock_event_prob:
            fair_move += float(self.rng.choice([-1.0, 1.0]) * self.config.shock_size_ticks * self.tick * self.rng.uniform(0.6, 1.4))
        self.fair_value = max(self.tick, self.fair_value + fair_move)
        return regime_shift, fair_move

    def _event_weights(self) -> tuple[list[str], np.ndarray]:
        mid = self.midprice
        signal_edge = np.clip((self.fair_value - mid) / max(self.tick, 1e-8), -4.0, 4.0)
        imbalance = self._top_imbalance()
        flow_term = np.tanh(self.signed_flow_state / 20.0)
        stress = self._stress_level()
        informed_edge = signal_edge + 0.8 * self.metaorder_bias
        names = [
            "noise_market_buy",
            "noise_market_sell",
            "informed_market_buy",
            "informed_market_sell",
            "maker_add_bid",
            "maker_add_ask",
            "maker_cancel_bid",
            "maker_cancel_ask",
            "refill_bid",
            "refill_ask",
        ]
        weights = np.asarray(
            [
                self.profile.noise_taker_rate * np.exp(0.10 * max(-imbalance, 0.0) - 0.10 * flow_term + 0.15 * stress),
                self.profile.noise_taker_rate * np.exp(0.10 * max(imbalance, 0.0) + 0.10 * flow_term + 0.15 * stress),
                self.profile.informed_taker_rate * np.exp(0.8 * informed_edge + 0.18 * stress),
                self.profile.informed_taker_rate * np.exp(-0.8 * informed_edge + 0.18 * stress),
                self.profile.maker_add_rate * np.exp(-0.14 * informed_edge + 0.16 * max(-imbalance, 0.0) - self.config.stress_liquidity_withdraw_scale * stress),
                self.profile.maker_add_rate * np.exp(0.14 * informed_edge + 0.16 * max(imbalance, 0.0) - self.config.stress_liquidity_withdraw_scale * stress),
                self.profile.maker_cancel_rate * np.exp(0.22 * informed_edge + 0.10 * flow_term + self.config.stress_liquidity_withdraw_scale * stress),
                self.profile.maker_cancel_rate * np.exp(-0.22 * informed_edge - 0.10 * flow_term + self.config.stress_liquidity_withdraw_scale * stress),
                self.profile.liquidity_refill_rate * np.exp(0.1 * stress),
                self.profile.liquidity_refill_rate * np.exp(0.1 * stress),
            ],
            dtype=np.float64,
        )
        weights = weights / weights.sum()
        return names, weights

    def _choose_event(self) -> str:
        names, weights = self._event_weights()
        return str(self.rng.choice(np.asarray(names), p=weights))

    def _cancel_from_side(self, side: str, owner_bias: str = "competing_mm") -> tuple[float, float, str]:
        ladder = self._depth_ladder(side)
        weights = _exp_weights(len(ladder))
        level_idx = int(self.rng.choice(np.arange(len(ladder)), p=weights))
        price = ladder[level_idx][0]
        book = self.bids if side == "bid" else self.asks
        queue = book.get(price)
        if not queue:
            return 0.0, 0.0, ""
        owner_scores = np.asarray([1.8 if order.owner == owner_bias else 1.0 for order in queue], dtype=np.float64)
        owner_scores = owner_scores / owner_scores.sum()
        queue_idx = int(self.rng.choice(np.arange(len(queue)), p=owner_scores))
        remove_size = min(queue[queue_idx].size, self._draw_size(0.7))
        removed = self._remove_order(side, price, queue_idx, remove_size)
        if removed is None:
            return 0.0, 0.0, ""
        return removed.price, removed.size, removed.owner

    def _place_competing_mm(self, side: str) -> tuple[float, float]:
        dynamic_touch_prob = float(np.clip(self.profile.maker_join_touch_prob - 0.14 * self._stress_level(), 0.02, 0.98))
        join_touch = self.rng.random() < dynamic_touch_prob
        level = 0 if join_touch else int(self.rng.choice([1, 2], p=[0.7, 0.3]))
        if side == "bid":
            target = min(np.floor(self.fair_value / self.tick) * self.tick, self.best_ask - self.tick)
            reference = self.best_bid if join_touch else min(target, self.best_bid - level * self.tick)
            price = round(reference, 6)
        else:
            target = max(np.ceil(self.fair_value / self.tick) * self.tick, self.best_bid + self.tick)
            reference = self.best_ask if join_touch else max(target, self.best_ask + level * self.tick)
            price = round(reference, 6)
        size = self._draw_size(self.profile.depth_scale)
        self._add_limit(side, price, size, owner="competing_mm", silent=True)
        return price, size

    def _place_liquidity_provider(self, side: str, deep: bool) -> tuple[float, float]:
        level = int(self.rng.choice(np.arange(10), p=_exp_weights(10, decay=0.35 if deep else 0.7)))
        if side == "bid":
            price = round(self.best_bid - level * self.tick, 6)
        else:
            price = round(self.best_ask + level * self.tick, 6)
        size = self._draw_size(self.profile.depth_scale * (1.4 if deep else 1.1))
        self._add_limit(side, price, size, owner="liquidity_provider", silent=True)
        return price, size

    def _market_order(self, side: str, taker_agent: str) -> tuple[list[TradeRecord], float]:
        aggressor_side = "B" if side == "buy" else "A"
        book = self.asks if side == "buy" else self.bids
        signed = 1.0 if side == "buy" else -1.0
        taker_scale = 1.4 if taker_agent == "informed_taker" else 1.0
        remaining = self._draw_size(taker_scale * (1.0 + 0.25 * self._stress_level()))
        initial = remaining
        trades: list[TradeRecord] = []
        while remaining > 1e-8 and book:
            best_price = min(book) if side == "buy" else max(book)
            queue = book[best_price]
            while remaining > 1e-8 and queue:
                queue_ahead = float(sum(order.size for order in list(queue)[:-1]))
                order = queue[0]
                matched = min(remaining, order.size)
                trades.append(
                    TradeRecord(
                        price=best_price,
                        size=matched,
                        aggressor_side=aggressor_side,
                        taker_agent=taker_agent,
                        maker_agent=order.owner,
                        maker_order_id=order.order_id,
                        queue_ahead=queue_ahead,
                    )
                )
                order.size -= matched
                remaining -= matched
                if order.size <= 1e-8:
                    queue.popleft()
            if not queue:
                book.pop(best_price, None)
        self.signed_flow_state = 0.985 * self.signed_flow_state + signed * (initial / max(self.trade_unit, 1.0))
        informed_scale = 1.3 if taker_agent == "informed_taker" else 0.55
        self.fair_value += signed * self.config.market_order_impact_scale * (
            self.config.market_order_tick_impact * self.tick + informed_scale * self.config.market_order_alpha_impact * max(abs(self.signal), 0.2)
        )
        if side == "buy":
            self._add_limit("ask", self.best_ask, max(self.trade_unit, 0.40 * initial), owner="liquidity_provider", silent=True)
        if side == "sell":
            self._add_limit("bid", self.best_bid, max(self.trade_unit, 0.40 * initial), owner="liquidity_provider", silent=True)
        return trades, initial - remaining

    def snapshot_rows(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        ask_row: dict[str, float] = {"timestamp": None}
        bid_row: dict[str, float] = {"timestamp": None}
        ask_levels = self._depth_ladder("ask")
        bid_levels = self._depth_ladder("bid")
        for level, (ask_level, bid_level) in enumerate(zip(ask_levels, bid_levels, strict=True), start=1):
            ask_price, ask_volume = ask_level
            bid_price, bid_volume = bid_level
            ask_row[f"ask{level}_price"] = float(ask_price)
            ask_row[f"ask{level}_volume"] = float(ask_volume)
            bid_row[f"bid{level}_price"] = float(bid_price)
            bid_row[f"bid{level}_volume"] = float(bid_volume)
        price_row = {
            "timestamp": None,
            "ask1_price": float(self.best_ask),
            "bid1_price": float(self.best_bid),
            "midprice": float(self.midprice),
        }
        return ask_row, bid_row, price_row

    def step(self) -> tuple[dict[str, float | int | str], dict[str, float], list[TradeRecord]]:
        regime_shift, fair_move = self._step_latent()
        event_name = self._choose_event()
        msg = {key: 0.0 for key in MSG_COLUMNS}
        trades: list[TradeRecord] = []
        event_side = ""
        event_actor = ""
        cancellation_owner = ""
        if event_name == "noise_market_buy":
            trades, matched = self._market_order("buy", taker_agent="noise_taker")
            msg["market_buy_volume"] = matched
            msg["market_buy_n"] = 1.0 if matched > 0 else 0.0
            event_side = "buy"
            event_actor = "noise_taker"
        elif event_name == "noise_market_sell":
            trades, matched = self._market_order("sell", taker_agent="noise_taker")
            msg["market_sell_volume"] = matched
            msg["market_sell_n"] = 1.0 if matched > 0 else 0.0
            event_side = "sell"
            event_actor = "noise_taker"
        elif event_name == "informed_market_buy":
            trades, matched = self._market_order("buy", taker_agent="informed_taker")
            msg["market_buy_volume"] = matched
            msg["market_buy_n"] = 1.0 if matched > 0 else 0.0
            event_side = "buy"
            event_actor = "informed_taker"
        elif event_name == "informed_market_sell":
            trades, matched = self._market_order("sell", taker_agent="informed_taker")
            msg["market_sell_volume"] = matched
            msg["market_sell_n"] = 1.0 if matched > 0 else 0.0
            event_side = "sell"
            event_actor = "informed_taker"
        elif event_name == "maker_add_bid":
            _, size = self._place_competing_mm("bid")
            msg["limit_buy_volume"] = size
            msg["limit_buy_n"] = 1.0
            event_side = "buy"
            event_actor = "competing_mm"
        elif event_name == "maker_add_ask":
            _, size = self._place_competing_mm("ask")
            msg["limit_sell_volume"] = size
            msg["limit_sell_n"] = 1.0
            event_side = "sell"
            event_actor = "competing_mm"
        elif event_name == "maker_cancel_bid":
            _, removed, cancellation_owner = self._cancel_from_side("bid", owner_bias="competing_mm")
            msg["withdraw_buy_volume"] = removed
            msg["withdraw_buy_n"] = 1.0 if removed > 0 else 0.0
            event_side = "buy"
            event_actor = "cancel"
        elif event_name == "maker_cancel_ask":
            _, removed, cancellation_owner = self._cancel_from_side("ask", owner_bias="competing_mm")
            msg["withdraw_sell_volume"] = removed
            msg["withdraw_sell_n"] = 1.0 if removed > 0 else 0.0
            event_side = "sell"
            event_actor = "cancel"
        elif event_name == "refill_bid":
            _, size = self._place_liquidity_provider("bid", deep=True)
            msg["limit_buy_volume"] = size
            msg["limit_buy_n"] = 1.0
            event_side = "buy"
            event_actor = "liquidity_provider"
        else:
            _, size = self._place_liquidity_provider("ask", deep=True)
            msg["limit_sell_volume"] = size
            msg["limit_sell_n"] = 1.0
            event_side = "sell"
            event_actor = "liquidity_provider"

        self._ensure_depth()
        best_bid_depth = self._level_depth("bid", self.best_bid)
        best_ask_depth = self._level_depth("ask", self.best_ask)
        traded_volume = float(sum(trade.size for trade in trades))
        maker_agent = trades[0].maker_agent if trades else cancellation_owner
        latent = {
            "fair_value": float(self.fair_value),
            "efficient_price": float(self.fair_value),
            "latent_alpha": float(self.signal),
            "metaorder_bias": float(self.metaorder_bias),
            "regime": int(self.regime),
            "signed_flow_state": float(self.signed_flow_state),
            "spread_ticks": int(self.spread_ticks),
            "top_imbalance": float(self._top_imbalance()),
            "queue_pressure": float(self._queue_pressure()),
            "event_type": event_name.split("_", maxsplit=1)[0],
            "event_name": event_name,
            "event_side": event_side,
            "event_actor": event_actor,
            "maker_agent": maker_agent,
            "regime_shift": int(regime_shift),
            "efficient_move": float(fair_move),
            "trade_count": int(len(trades)),
            "traded_volume": traded_volume,
            "best_bid_depth": best_bid_depth,
            "best_ask_depth": best_ask_depth,
        }
        self.event_seq += 1
        return latent, msg, trades


def _symbol_profile(symbol: str, config: GenerateConfig) -> SymbolProfile:
    return SymbolProfile(
        base_price=config.base_prices[symbol],
        fair_value_persistence={"000001": 0.985, "000858": 0.988, "002415": 0.99}.get(symbol, 0.988),
        signal_noise={"000001": 0.035, "000858": 0.03, "002415": 0.028}.get(symbol, 0.03),
        noise_taker_rate={"000001": 1.3, "000858": 1.0, "002415": 0.85}.get(symbol, 1.0) * config.noise_taker_rate_scale,
        informed_taker_rate={"000001": 0.45, "000858": 0.35, "002415": 0.28}.get(symbol, 0.35) * config.informed_taker_rate_scale,
        maker_add_rate={"000001": 1.15, "000858": 1.0, "002415": 0.9}.get(symbol, 1.0) * config.maker_add_rate_scale,
        maker_cancel_rate={"000001": 0.9, "000858": 0.82, "002415": 0.75}.get(symbol, 0.82) * config.maker_cancel_rate_scale,
        liquidity_refill_rate={"000001": 1.1, "000858": 0.95, "002415": 0.85}.get(symbol, 0.95) * config.liquidity_refill_rate_scale,
        maker_join_touch_prob=float(
            np.clip(
                {"000001": 0.78, "000858": 0.72, "002415": 0.68}.get(symbol, 0.72) + config.maker_join_touch_prob_shift,
                0.05,
                0.98,
            )
        ),
        depth_scale={"000001": 3.0, "000858": 2.2, "002415": 1.8}.get(symbol, 2.0),
    )


def generate_day_frames(symbol: str, day_index: int, config: GenerateConfig) -> dict[str, pd.DataFrame]:
    day = _day_label(day_index)
    seed = _stable_seed(config.seed, symbol, day)
    rng = np.random.default_rng(seed)
    anchor_day = pd.Timestamp(day)
    timestamps = _session_timestamps(anchor_day, config.session_windows, config.events_per_day[symbol], rng)
    profile = _symbol_profile(symbol, config)
    book = AgentBasedLOB(config, profile, rng)

    ask_rows = []
    bid_rows = []
    price_rows = []
    msg_rows = []
    trade_rows = []
    latent_rows = []

    for ts in timestamps:
        ask_row, bid_row, price_row = book.snapshot_rows()
        ask_row["timestamp"] = ts
        bid_row["timestamp"] = ts
        price_row["timestamp"] = ts
        latent, msg, trades = book.step()
        msg_rows.append({"timestamp": ts, **msg})
        for trade in trades:
            trade_rows.append(
                {
                    "timestamp": ts,
                    "price": trade.price,
                    "size": trade.size,
                    "aggressor_side": trade.aggressor_side,
                    "taker_agent": trade.taker_agent,
                    "maker_agent": trade.maker_agent,
                    "maker_order_id": trade.maker_order_id,
                    "queue_ahead": trade.queue_ahead,
                }
            )
        latent_rows.append({"timestamp": ts, **latent})
        ask_rows.append(ask_row)
        bid_rows.append(bid_row)
        price_rows.append(price_row)

    return {
        "ask": pd.DataFrame(ask_rows),
        "bid": pd.DataFrame(bid_rows),
        "price": pd.DataFrame(price_rows),
        "msg": pd.DataFrame(msg_rows),
        "trades": pd.DataFrame(trade_rows),
        "latent": pd.DataFrame(latent_rows),
    }


def generate_dataset(config: GenerateConfig) -> dict[str, object]:
    config.apply_mode_defaults()
    set_seed(config.seed)
    root = ensure_dir(config.data_dir)
    manifest = {"symbols": config.symbols, "days": []}
    for symbol in config.symbols:
        for day_index in range(config.num_days):
            day = _day_label(day_index)
            day_root = root / symbol / day
            if day_root.exists() and not config.overwrite:
                manifest["days"].append({"symbol": symbol, "day": day, "status": "kept"})
                continue
            ensure_dir(day_root)
            frames = generate_day_frames(symbol, day_index, config)
            for name, frame in frames.items():
                frame.to_csv(day_root / f"{name}.csv", index=False)
            manifest["days"].append({"symbol": symbol, "day": day, "status": "generated"})
    save_json(root / "manifest.json", manifest)
    return manifest


@pyrallis.wrap()
def main(config: GenerateConfig) -> None:
    generate_dataset(config)


if __name__ == "__main__":
    main()
