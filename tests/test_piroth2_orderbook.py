from __future__ import annotations

import pandas as pd

from piroth.orderbook import FIFOOrderBook


def test_market_order_consumes_best_queue_fifo() -> None:
    book = FIFOOrderBook(tick_size=0.01, levels=10)
    ts = pd.Timestamp("2019-11-01 10:00:00")
    first = book.add_limit_order("ask", 1001, 100, "mm_1", "competing_mm", ts)
    second = book.add_limit_order("ask", 1001, 200, "lp_1", "liquidity_provider", ts)

    fills = book.market_order("buy", 150, ts, "noise_taker")

    assert len(fills) == 2
    assert fills[0].maker_order_id == first.order_id
    assert fills[0].size == 100
    assert fills[1].maker_order_id == second.order_id
    assert fills[1].size == 50
    assert book.aggregated_depth("ask", 1001) == 150
