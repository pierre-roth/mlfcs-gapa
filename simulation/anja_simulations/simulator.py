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
        self.regime = 0
        self.regime_clock = 0
        self.signed_flow_state = 0.0
        self.vol_state = 1.0  # volatility clustering multiplier
        # Exponentially-decaying count of recent informed trades (Hawkes kernel).
        # Decays every _step_latent, bumps +1 on each informed market order.
        self.informed_clock = 0.0

        self.bids: dict[float, deque[RestingOrder]] = {}
        self.asks: dict[float, deque[RestingOrder]] = {}
        self._initialize_book()

    def _initialize_book(self) -> None:
        base_bid = np.floor(self.reference_price / self.tick) * self.tick
        for level in range(10):
            bid_price = round(base_bid - level * self.tick, 6)
            ask_price = round(base_bid + (level + 1) * self.tick, 6)
            # Inverted depth curve: thin touch, thick deeper levels
            scale = self.profile.depth_scale * (0.9 + 0.08 * level)
            # Touch gets 2 MMs; mid (2-4) gets 2; deep (5+) gets 3
            maker_orders = 2 if level < 5 else 3
            for _ in range(maker_orders):
                self._add_limit("bid", bid_price, self._draw_size(scale), owner="competing_mm", silent=True)
                self._add_limit("ask", ask_price, self._draw_size(scale), owner="competing_mm", silent=True)
            self._add_limit("bid", bid_price, self._draw_size(scale * 1.3), owner="liquidity_provider", silent=True)
            self._add_limit("ask", ask_price, self._draw_size(scale * 1.3), owner="liquidity_provider", silent=True)
        self._ensure_depth()

    def _draw_size(self, scale: float = 1.0) -> float:
        lots = int(self.rng.choice([1, 2, 3, 4, 5], p=[0.42, 0.24, 0.16, 0.10, 0.08]))
        return float(max(self.trade_unit, round(lots * scale) * self.trade_unit))

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
        book.setdefault(price, deque()).append(order)
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


    def _ensure_side_depth(self, side: str, depth: int = 10) -> None:
        book = self.bids if side == "bid" else self.asks
        best_price = self.best_bid if side == "bid" else self.best_ask
        step = -self.tick if side == "bid" else self.tick
        target_prices = {round(best_price + level * step, 6) for level in range(depth)}
        missing_prices = sorted(target_prices.difference(book.keys()), reverse=(side == "bid"))
        # Phase 3c: when signal is strong, thicken the "own" side (direction of
        # signal) and thin the "away" side during depth refills. This creates a
        # leading LOB imbalance that Attn-LOB can read off the raw book shape.
        depth_multiplier = 1.0
        if (
            self.config.lob_leak_strength > 0.0
            and abs(self.signal) > self.config.signal_threshold_for_lob_leak
        ):
            own_side_is_bid = self.signal > 0.0
            placing_bid = side == "bid"
            mag = min(abs(self.signal), 2.0)
            if placing_bid == own_side_is_bid:
                depth_multiplier = 1.0 + 0.4 * self.config.lob_leak_strength * mag
            else:
                depth_multiplier = 1.0 - 0.4 * self.config.lob_leak_strength * mag
            depth_multiplier = float(np.clip(depth_multiplier, 0.4, 1.8))
        for price in missing_prices:
            self._add_limit(
                side,
                price,
                self._draw_size(self.profile.depth_scale * depth_multiplier),
                owner="liquidity_provider",
                silent=True,
            )

    def _ensure_depth(self) -> None:
        # Maintain 8 levels from current best price outward
        self._ensure_side_depth("bid", depth=8)
        self._ensure_side_depth("ask", depth=8)
        # --- Spread narrowing: market makers compete to close wide spreads ---
        # In real markets, a wide spread is a profit opportunity: MMs race to
        # improve the quote and capture the spread. Probability of narrowing
        # increases with spread width — 1-tick is the equilibrium for liquid stocks.
        spread = self.spread_ticks
        if spread > 1:
            # Narrowing probability: fairly aggressive at 2 ticks, very strong at 3+
            narrow_prob = min(0.92, 0.55 * (spread - 1))
            if self.rng.random() < narrow_prob:
                mid = self.midprice
                inner_levels = min(spread - 1, 3)
                for i in range(1, inner_levels + 1):
                    bid_price = round(self.best_bid + i * self.tick, 6)
                    if bid_price < mid:
                        self._add_limit("bid", bid_price, self._draw_size(self.profile.depth_scale * 0.7),
                                        owner="competing_mm", silent=True)
                    ask_price = round(self.best_ask - i * self.tick, 6)
                    if ask_price > mid:
                        self._add_limit("ask", ask_price, self._draw_size(self.profile.depth_scale * 0.7),
                                        owner="competing_mm", silent=True)


    def _step_latent(self) -> tuple[int, float]:
        self.regime_clock += 1
        regime_shift = 0
        if self.regime_clock > 300 and self.rng.random() < 0.004:
            self.regime = int(self.rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3]))
            self.regime_clock = 0
            regime_shift = 1
        target = self.config.alpha_signal_scale * 0.4 * self.regime
        self.signal = (
            self.profile.fair_value_persistence * self.signal
            + (1.0 - self.profile.fair_value_persistence) * target
            + self.rng.normal(0.0, self.profile.signal_noise * self.config.alpha_signal_scale)
        )
        # --- volatility clustering (gentle GARCH-like) ---
        vol_target = 1.0 + 0.3 * abs(self.regime) + 0.1 * min(abs(self.signal), 2.0)
        vol_shock = 0.003 * abs(self.rng.normal())
        self.vol_state = np.clip(
            0.99 * self.vol_state + 0.01 * vol_target + vol_shock,
            0.6, 1.8,
        )
        # Decay informed-trade Hawkes kernel. The +1 bumps happen in _market_order.
        self.informed_clock *= self.config.informed_hawkes_decay
        # --- fair value update with nonlinear mean-reversion toward displayed mid ---
        # Soft for small deviations (allows natural price movement) but hard
        # for large divergences (prevents the feedback loop where MMs retreat
        # and the book collapses). This is realistic: real MMs arbitrage
        # large persistent deviations, but tolerate small temporary edges.
        mid = self.midprice
        edge_ticks = abs(self.fair_value - mid) / max(self.tick, 1e-8)
        reversion_strength = 0.003 + 0.04 * min(edge_ticks / 3.0, 1.0) ** 2
        reversion = reversion_strength * (mid - self.fair_value)
        # Price-proportional noise: ensures all stocks have similar % daily range
        # regardless of absolute price level (a ¥135 stock should move ~1% just
        # like a ¥12.5 stock, but with the same tick_size=0.01)
        price_scale = self.reference_price / 12.5
        fair_move = (
            reversion
            + price_scale * 0.0022 * self.signal
            + self.rng.normal(0.0, price_scale * self.vol_state * 0.35 * self.config.price_noise_scale)
        )
        self.fair_value = max(self.tick, self.fair_value + fair_move)
        return regime_shift, fair_move

    def _event_weights(self) -> tuple[list[str], np.ndarray]:
        mid = self.midprice
        signal_edge = np.clip((self.fair_value - mid) / max(self.tick, 1e-8), -2.0, 2.0)
        imbalance = self._top_imbalance()
        flow_term = np.tanh(self.signed_flow_state / 20.0)
        vol_scale = self.vol_state
        # --- Hawkes self-excitation on informed rate (Phase 3b) ---
        hawkes_boost = 1.0 + self.config.informed_hawkes_alpha * self.informed_clock
        # --- LOB alpha leak: asymmetric maker add/cancel (Phase 3a) ---
        # When |signal| is large, informed-ish makers skew the book: thicker queue
        # on the "own side" (direction of signal), aggressive cancels on the
        # opposite side. This creates LOB imbalance BEFORE the informed trades
        # arrive, which Attn-LOB can observe over its 50-event window.
        leak = 0.0
        if abs(self.signal) > self.config.signal_threshold_for_lob_leak:
            # signal_edge is already signed and clipped to [-2, 2]
            leak = self.config.lob_leak_strength * signal_edge
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
                # Noise takers: gentle vol sensitivity
                self.profile.noise_taker_rate * np.exp(0.08 * max(-imbalance, 0.0) - 0.08 * flow_term) * (0.92 + 0.08 * vol_scale),
                self.profile.noise_taker_rate * np.exp(0.08 * max(imbalance, 0.0) + 0.08 * flow_term) * (0.92 + 0.08 * vol_scale),
                # Informed takers: moderately stronger signal sensitivity (was 0.65, now 0.55 with tight clip)
                self.profile.informed_taker_rate * np.exp(0.55 * signal_edge) * (0.88 + 0.12 * vol_scale) * hawkes_boost,
                self.profile.informed_taker_rate * np.exp(-0.55 * signal_edge) * (0.88 + 0.12 * vol_scale) * hawkes_boost,
                # Competing MMs add: very slight retreat during strong signal + leak
                # leak > 0 (signal positive) → boost bid adds, suppress ask adds
                self.profile.maker_add_rate * np.exp(-0.12 * signal_edge + 0.14 * max(-imbalance, 0.0) + 0.6 * leak) * np.exp(-0.04 * abs(signal_edge)),
                self.profile.maker_add_rate * np.exp(0.12 * signal_edge + 0.14 * max(imbalance, 0.0) - 0.6 * leak) * np.exp(-0.04 * abs(signal_edge)),
                # Competing MMs cancel: moderately stronger signal response + leak
                # leak > 0 (signal positive) → fewer bid cancels (keep queue), more ask cancels
                self.profile.maker_cancel_rate * np.exp(0.22 * signal_edge + 0.10 * flow_term - 0.6 * leak) * (0.92 + 0.08 * vol_scale),
                self.profile.maker_cancel_rate * np.exp(-0.22 * signal_edge - 0.10 * flow_term + 0.6 * leak) * (0.92 + 0.08 * vol_scale),
                # Refill: very slight vol sensitivity
                self.profile.liquidity_refill_rate * (1.03 - 0.03 * vol_scale),
                self.profile.liquidity_refill_rate * (1.03 - 0.03 * vol_scale),
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
        # Bias cancellations toward THICK levels (MMs pull from queues
        # where they're far back) rather than always the touch
        depths = np.asarray([depth for _, depth in ladder], dtype=np.float64)
        base_weights = _exp_weights(len(ladder))
        depth_weights = depths / max(depths.sum(), 1e-8)
        combined = 0.4 * base_weights + 0.6 * depth_weights
        combined = combined / combined.sum()
        level_idx = int(self.rng.choice(np.arange(len(ladder)), p=combined))
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
        # Competing MMs slightly reduce touch-joining during strong signals
        signal_edge = abs(self.fair_value - self.midprice) / max(self.tick, 1e-8)
        touch_penalty = min(0.12, 0.03 * signal_edge + 0.02 * max(self.vol_state - 1.0, 0.0))
        effective_join_prob = max(0.30, self.profile.maker_join_touch_prob - touch_penalty)
        join_touch = self.rng.random() < effective_join_prob
        # --- Depth ceiling: no MM joins a queue that's already very thick ---
        # Real MMs avoid adding to deep queues (no benefit to being 50th)
        touch_depth = self._level_depth(side, self.best_bid if side == "bid" else self.best_ask)
        depth_cap = 8.0 * self.trade_unit * self.profile.depth_scale
        if touch_depth > depth_cap and join_touch:
            join_touch = False  # force placement deeper
        # --- Spread-tightening competition ---
        # When spread > 1 tick, MMs compete to improve the best price.
        current_spread = self.spread_ticks
        if current_spread > 1 and join_touch:
            improve_prob = min(0.7, 0.2 + 0.15 * (current_spread - 1))
            if self.rng.random() < improve_prob:
                if side == "bid":
                    improved = round(self.best_bid + self.tick, 6)
                    if improved < self.best_ask:
                        price = improved
                        size = self._draw_size(self.profile.depth_scale)
                        self._add_limit(side, price, size, owner="competing_mm", silent=True)
                        return price, size
                else:
                    improved = round(self.best_ask - self.tick, 6)
                    if improved > self.best_bid:
                        price = improved
                        size = self._draw_size(self.profile.depth_scale)
                        self._add_limit(side, price, size, owner="competing_mm", silent=True)
                        return price, size
        # --- Standard placement logic ---
        level = 0 if join_touch else int(self.rng.choice([1, 2, 3], p=[0.55, 0.30, 0.15]))
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
        # Check if touch is already thick enough
        touch_price = self.best_bid if side == "bid" else self.best_ask
        touch_depth = self._level_depth(side, touch_price)
        depth_cap = 6.0 * self.trade_unit * self.profile.depth_scale
        if touch_depth > depth_cap:
            # Force placement deeper when touch is thick
            level = int(self.rng.choice(np.arange(2, 10), p=_exp_weights(8, decay=0.35)))
        else:
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
        # Capture the touch price BEFORE the walk so touch replenishment
        # restores the original level (not the new-best-after-consumption).
        # This eliminates the spurious "1 tick of adverse selection per noise trade"
        # caused by the touch being entirely consumed and best_bid/ask jumping.
        pre_touch = self.best_ask if side == "buy" else self.best_bid
        # Informed takers occasionally send larger orders (right-skewed)
        if taker_agent == "informed_taker":
            size_mult = 1.1 + 0.3 * self.rng.exponential(1.0)
        else:
            size_mult = 1.0
        remaining = self._draw_size(size_mult)
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
        # Bump Hawkes self-excitation kernel on informed trades.
        if taker_agent == "informed_taker":
            self.informed_clock += 1.0
        # --- size-proportional fair value impact ---
        size_lots = initial / max(self.trade_unit, 1.0)
        informed_scale = 1.5 if taker_agent == "informed_taker" else 0.5
        price_scale = self.reference_price / 12.5
        impact = self.config.market_order_impact_scale * price_scale * (
            self.config.market_order_tick_impact * self.tick * (0.7 + 0.3 * size_lots)
            + informed_scale * self.config.market_order_alpha_impact * max(abs(self.signal), 0.2)
        )
        self.fair_value += signed * impact
        # --- Touch replenishment at the ORIGINAL price ---
        # In real markets, when touch gets consumed, a wave of new orders arrive
        # at the same price level (LPs see the opportunity and refill). Without
        # this, the best bid/ask jumps by 1 tick on every trade that eats the
        # touch, creating spurious adverse selection even for noise flow.
        # We only replenish noise-driven trades — informed flow is *supposed* to
        # cause real price discovery (no replenishment).
        if self.config.touch_replenish_fraction > 0 and taker_agent == "noise_taker":
            replenish_size = max(self.trade_unit, self.config.touch_replenish_fraction * initial)
            if side == "buy":
                # Ask touch got consumed — restore at pre_touch if the best_ask
                # has moved up (i.e. we fully ate the old touch)
                if self.best_ask > pre_touch + self.tick / 2:
                    self._add_limit("ask", pre_touch, replenish_size, owner="liquidity_provider", silent=True)
            else:
                if self.best_bid < pre_touch - self.tick / 2:
                    self._add_limit("bid", pre_touch, replenish_size, owner="liquidity_provider", silent=True)
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
            "regime": int(self.regime),
            "signed_flow_state": float(self.signed_flow_state),
            "vol_state": float(self.vol_state),
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
        depth_scale={"000001": 1.35, "000858": 1.0, "002415": 0.82}.get(symbol, 1.0),
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