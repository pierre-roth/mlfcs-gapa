from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Order:
    side: str
    price: int
    size: int


class OrderBook:
    def __init__(self) -> None:
        self.orders: Dict[int, Order] = {}
        self.ask_levels: Dict[int, int] = {}
        self.bid_levels: Dict[int, int] = {}
        self.ask_prices: List[int] = []
        self.bid_prices: List[int] = []

    def clear(self) -> None:
        self.orders.clear()
        self.ask_levels.clear()
        self.bid_levels.clear()
        self.ask_prices.clear()
        self.bid_prices.clear()

    def _level_state(self, side: str) -> Tuple[Dict[int, int], List[int]]:
        if side == "A":
            return self.ask_levels, self.ask_prices
        return self.bid_levels, self.bid_prices

    def _add_level_size(self, side: str, price: int, size: int) -> None:
        levels, prices = self._level_state(side)
        previous = levels.get(price, 0)
        levels[price] = previous + size
        if previous == 0:
            insort(prices, price)

    def _remove_level_size(self, side: str, price: int, size: int) -> int:
        levels, prices = self._level_state(side)
        previous = levels.get(price, 0)
        removed = min(previous, size)
        remaining = previous - removed
        if remaining > 0:
            levels[price] = remaining
        elif previous > 0:
            del levels[price]
            idx = bisect_left(prices, price)
            if idx < len(prices) and prices[idx] == price:
                prices.pop(idx)
        return removed

    def add(self, order_id: int, side: str, price: int, size: int) -> None:
        if size <= 0:
            return
        self.orders[order_id] = Order(side=side, price=price, size=size)
        self._add_level_size(side, price, size)

    def cancel(self, order_id: int, size: int) -> Optional[Tuple[str, int]]:
        order = self.orders.get(order_id)
        if order is None or size <= 0:
            return None
        removed = min(order.size, size)
        self._remove_level_size(order.side, order.price, removed)
        order.size -= removed
        if order.size == 0:
            self.orders.pop(order_id, None)
        return order.side, removed

    def fill(self, order_id: int, size: int) -> Optional[Tuple[str, int]]:
        return self.cancel(order_id, size)

    def modify(self, order_id: int, price: int, size: int) -> Optional[Tuple[str, int, int]]:
        order = self.orders.get(order_id)
        if order is None:
            return None

        old_side = order.side
        old_price = order.price
        old_size = order.size

        self._remove_level_size(old_side, old_price, old_size)

        if size <= 0:
            self.orders.pop(order_id, None)
            return old_side, old_size, 0

        order.price = price
        order.size = size
        self._add_level_size(order.side, order.price, order.size)
        return old_side, old_size, size

    def top_n(self, levels: int = 10) -> Optional[Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]]:
        if len(self.ask_prices) < levels or len(self.bid_prices) < levels:
            return None
        ask = tuple((price, self.ask_levels[price]) for price in self.ask_prices[:levels])
        bid = tuple((price, self.bid_levels[price]) for price in reversed(self.bid_prices[-levels:]))
        return ask, bid
