from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import pandas as pd


@dataclass
class RestingOrder:
    order_id: int
    agent_id: str
    agent_type: str
    side: str
    tick: int
    quantity: int
    timestamp: pd.Timestamp


@dataclass
class TradeFill:
    timestamp: pd.Timestamp
    price: float
    size: int
    aggressor_side: str
    taker_agent: str
    maker_agent_id: str
    maker_agent: str
    maker_order_id: int
    queue_ahead: int


class FIFOOrderBook:
    def __init__(self, tick_size: float, levels: int) -> None:
        self.tick_size = tick_size
        self.levels = levels
        self.bids: dict[int, Deque[RestingOrder]] = {}
        self.asks: dict[int, Deque[RestingOrder]] = {}
        self.next_order_id = 1

    def _side_map(self, side: str) -> dict[int, Deque[RestingOrder]]:
        return self.bids if side == "bid" else self.asks

    def _best_tick(self, side: str) -> int | None:
        book = self._side_map(side)
        if not book:
            return None
        return max(book) if side == "bid" else min(book)

    def best_bid_tick(self) -> int | None:
        return self._best_tick("bid")

    def best_ask_tick(self) -> int | None:
        return self._best_tick("ask")

    def best_bid_price(self) -> float | None:
        tick = self.best_bid_tick()
        return None if tick is None else tick * self.tick_size

    def best_ask_price(self) -> float | None:
        tick = self.best_ask_tick()
        return None if tick is None else tick * self.tick_size

    def midprice(self) -> float:
        bid = self.best_bid_tick()
        ask = self.best_ask_tick()
        if bid is None and ask is None:
            return 0.0
        if bid is None:
            return ask * self.tick_size
        if ask is None:
            return bid * self.tick_size
        return 0.5 * (bid + ask) * self.tick_size

    def aggregated_depth(self, side: str, tick: int) -> int:
        queue = self._side_map(side).get(tick)
        if not queue:
            return 0
        return int(sum(order.quantity for order in queue))

    def add_limit_order(
        self,
        side: str,
        tick: int,
        quantity: int,
        agent_id: str,
        agent_type: str,
        timestamp: pd.Timestamp,
    ) -> RestingOrder:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        order = RestingOrder(
            order_id=self.next_order_id,
            agent_id=agent_id,
            agent_type=agent_type,
            side=side,
            tick=tick,
            quantity=int(quantity),
            timestamp=timestamp,
        )
        self.next_order_id += 1
        book = self._side_map(side)
        book.setdefault(tick, deque()).append(order)
        return order

    def cancel_order(self, side: str, tick: int, order_id: int) -> RestingOrder | None:
        book = self._side_map(side)
        queue = book.get(tick)
        if not queue:
            return None
        for idx, order in enumerate(queue):
            if order.order_id == order_id:
                removed = queue[idx]
                del queue[idx]
                if not queue:
                    book.pop(tick, None)
                return removed
        return None

    def cancel_agent_orders(self, agent_id: str) -> int:
        removed = 0
        for book in (self.bids, self.asks):
            empty_ticks = []
            for tick, queue in book.items():
                kept = deque(order for order in queue if order.agent_id != agent_id)
                removed += len(queue) - len(kept)
                book[tick] = kept
                if not kept:
                    empty_ticks.append(tick)
            for tick in empty_ticks:
                book.pop(tick, None)
        return removed

    def cancel_random(
        self,
        side: str,
        rng: np.random.Generator,
        near_touch_bias: float = 0.7,
    ) -> RestingOrder | None:
        book = self._side_map(side)
        if not book:
            return None
        ranked_ticks = sorted(book, reverse=(side == "bid"))
        level_weights = np.asarray([near_touch_bias ** idx for idx in range(len(ranked_ticks))], dtype=np.float64)
        level_weights /= level_weights.sum()
        chosen_tick = ranked_ticks[int(rng.choice(len(ranked_ticks), p=level_weights))]
        queue = book[chosen_tick]
        if not queue:
            return None
        queue_sizes = np.asarray([max(order.quantity, 1) for order in queue], dtype=np.float64)
        queue_sizes /= queue_sizes.sum()
        chosen_idx = int(rng.choice(len(queue), p=queue_sizes))
        removed = queue[chosen_idx]
        del queue[chosen_idx]
        if not queue:
            book.pop(chosen_tick, None)
        return removed

    def market_order(
        self,
        side: str,
        quantity: int,
        timestamp: pd.Timestamp,
        taker_agent: str,
    ) -> list[TradeFill]:
        if quantity <= 0:
            return []
        book = self.asks if side == "buy" else self.bids
        fills: list[TradeFill] = []
        remaining = int(quantity)
        while remaining > 0 and book:
            best_tick = min(book) if side == "buy" else max(book)
            queue = book[best_tick]
            while remaining > 0 and queue:
                top = queue[0]
                queue_ahead = 0
                executed = min(remaining, top.quantity)
                top.quantity -= executed
                remaining -= executed
                fills.append(
                    TradeFill(
                        timestamp=timestamp,
                        price=best_tick * self.tick_size,
                        size=executed,
                        aggressor_side="B" if side == "buy" else "A",
                        taker_agent=taker_agent,
                        maker_agent_id=top.agent_id,
                        maker_agent=top.agent_type,
                        maker_order_id=top.order_id,
                        queue_ahead=queue_ahead,
                    )
                )
                if top.quantity == 0:
                    queue.popleft()
            if not queue:
                book.pop(best_tick, None)
        return fills

    def top_levels(self) -> dict[str, list[tuple[float, int]]]:
        bid_ticks = sorted(self.bids, reverse=True)[: self.levels]
        ask_ticks = sorted(self.asks)[: self.levels]
        if bid_ticks:
            last_bid = bid_ticks[-1]
        else:
            last_bid = 0
        if ask_ticks:
            last_ask = ask_ticks[-1]
        else:
            last_ask = last_bid + 1
        bids = [(tick * self.tick_size, self.aggregated_depth("bid", tick)) for tick in bid_ticks]
        asks = [(tick * self.tick_size, self.aggregated_depth("ask", tick)) for tick in ask_ticks]
        while len(bids) < self.levels:
            last_bid -= 1
            bids.append((last_bid * self.tick_size, 0))
        while len(asks) < self.levels:
            last_ask += 1
            asks.append((last_ask * self.tick_size, 0))
        return {"bid": bids, "ask": asks}

    def relative_depth(self, center_tick: int, radius_ticks: int) -> dict[int, int]:
        depth: dict[int, int] = {}
        for rel in range(-radius_ticks, radius_ticks + 1):
            tick = center_tick + rel
            if rel < 0:
                depth[rel] = self.aggregated_depth("bid", tick)
            elif rel > 0:
                depth[rel] = self.aggregated_depth("ask", tick)
            else:
                depth[rel] = 0
        return depth
