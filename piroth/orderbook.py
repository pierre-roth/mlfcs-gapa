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
        self.bid_depths: dict[int, int] = {}
        self.ask_depths: dict[int, int] = {}
        self.orders_by_id: dict[int, tuple[str, int, RestingOrder]] = {}
        self.agent_order_ids: dict[str, set[int]] = {}
        self._best_bid_tick: int | None = None
        self._best_ask_tick: int | None = None
        self.next_order_id = 1

    def _side_map(self, side: str) -> dict[int, Deque[RestingOrder]]:
        return self.bids if side == "bid" else self.asks

    def _depth_map(self, side: str) -> dict[int, int]:
        return self.bid_depths if side == "bid" else self.ask_depths

    def _best_tick(self, side: str) -> int | None:
        return self._best_bid_tick if side == "bid" else self._best_ask_tick

    def _refresh_best_tick(self, side: str) -> None:
        book = self._side_map(side)
        best = None if not book else (max(book) if side == "bid" else min(book))
        if side == "bid":
            self._best_bid_tick = best
        else:
            self._best_ask_tick = best

    def _note_added_level(self, side: str, tick: int) -> None:
        if side == "bid":
            if self._best_bid_tick is None or tick > self._best_bid_tick:
                self._best_bid_tick = tick
        elif self._best_ask_tick is None or tick < self._best_ask_tick:
            self._best_ask_tick = tick

    def _drop_empty_level(self, side: str, tick: int) -> None:
        self._side_map(side).pop(tick, None)
        self._depth_map(side).pop(tick, None)
        if tick == self._best_tick(side):
            self._refresh_best_tick(side)

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
        return int(self._depth_map(side).get(tick, 0))

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
        depths = self._depth_map(side)
        depths[tick] = depths.get(tick, 0) + order.quantity
        self._note_added_level(side, tick)
        self.orders_by_id[order.order_id] = (side, tick, order)
        self.agent_order_ids.setdefault(agent_id, set()).add(order.order_id)
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
                depths = self._depth_map(side)
                depths[tick] = max(0, depths.get(tick, 0) - removed.quantity)
                self._drop_order_index(removed)
                if not queue:
                    self._drop_empty_level(side, tick)
                return removed
        return None

    def cancel_agent_orders(self, agent_id: str) -> int:
        removed = 0
        for order_id in list(self.agent_order_ids.get(agent_id, ())):
            located = self.orders_by_id.get(order_id)
            if located is None:
                continue
            side, tick, order = located
            queue = self._side_map(side).get(tick)
            if not queue:
                self._drop_order_index(order)
                continue
            for idx, candidate in enumerate(queue):
                if candidate.order_id != order_id:
                    continue
                removed_order = queue[idx]
                del queue[idx]
                depths = self._depth_map(side)
                depths[tick] = max(0, depths.get(tick, 0) - removed_order.quantity)
                if not queue:
                    self._drop_empty_level(side, tick)
                self._drop_order_index(removed_order)
                removed += 1
                break
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
        depths = self._depth_map(side)
        depths[chosen_tick] = max(0, depths.get(chosen_tick, 0) - removed.quantity)
        self._drop_order_index(removed)
        if not queue:
            self._drop_empty_level(side, chosen_tick)
        return removed

    def cancel_random_fraction(
        self,
        side: str,
        rng: np.random.Generator,
        mean_fraction: float = 0.35,
        near_touch_bias: float = 0.7,
        lot_size: int = 1,
    ) -> RestingOrder | None:
        book = self._side_map(side)
        if not book:
            return None
        ranked_ticks = sorted(book, reverse=(side == "bid"))
        level_weights = np.asarray([near_touch_bias**idx for idx in range(len(ranked_ticks))], dtype=np.float64)
        level_weights /= level_weights.sum()
        chosen_tick = ranked_ticks[int(rng.choice(len(ranked_ticks), p=level_weights))]
        queue = book[chosen_tick]
        if not queue:
            return None
        queue_sizes = np.asarray([max(order.quantity, 1) for order in queue], dtype=np.float64)
        queue_sizes /= queue_sizes.sum()
        chosen_idx = int(rng.choice(len(queue), p=queue_sizes))
        target = queue[chosen_idx]
        fraction = float(np.clip(rng.normal(mean_fraction, 0.15), 0.05, 1.0))
        round_lot = max(int(lot_size), 1)
        remove_qty = max(round_lot, int(round(target.quantity * fraction / round_lot)) * round_lot)
        remove_qty = min(remove_qty, target.quantity)
        removed = RestingOrder(
            order_id=target.order_id,
            agent_id=target.agent_id,
            agent_type=target.agent_type,
            side=target.side,
            tick=target.tick,
            quantity=remove_qty,
            timestamp=target.timestamp,
        )
        target.quantity -= remove_qty
        depths = self._depth_map(side)
        depths[chosen_tick] = max(0, depths.get(chosen_tick, 0) - remove_qty)
        if target.quantity <= 0:
            del queue[chosen_idx]
            self._drop_order_index(target)
        if not queue:
            self._drop_empty_level(side, chosen_tick)
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
        book_side = "ask" if side == "buy" else "bid"
        book = self._side_map(book_side)
        depths = self._depth_map(book_side)
        fills: list[TradeFill] = []
        remaining = int(quantity)
        while remaining > 0 and book:
            best_tick = self._best_tick(book_side)
            if best_tick is None:
                break
            queue = book[best_tick]
            while remaining > 0 and queue:
                top = queue[0]
                queue_ahead = 0
                executed = min(remaining, top.quantity)
                top.quantity -= executed
                depths[best_tick] = max(0, depths.get(best_tick, 0) - executed)
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
                    filled = queue.popleft()
                    self._drop_order_index(filled)
            if not queue:
                self._drop_empty_level(book_side, best_tick)
        return fills

    def _drop_order_index(self, order: RestingOrder) -> None:
        self.orders_by_id.pop(order.order_id, None)
        order_ids = self.agent_order_ids.get(order.agent_id)
        if order_ids is None:
            return
        order_ids.discard(order.order_id)
        if not order_ids:
            self.agent_order_ids.pop(order.agent_id, None)

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
