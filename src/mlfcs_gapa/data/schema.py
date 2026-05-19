"""Canonical event-level LOB schema used by the replication.

The paper's model consumes a rolling window of 10 LOB levels with this feature
order per level:

``ask_price, ask_volume, bid_price, bid_volume``.

The simulator also needs event-message aggregates for OSI features and trade
extrema for paper-style historical fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mlfcs_gapa.paper.constants import PAPER

if TYPE_CHECKING:
    import polars as pl


TIMESTAMP = "timestamp"
STOCK = "stock"
DAY = "day"


def lob_columns(levels: int = PAPER.lob_levels) -> list[str]:
    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend(
            [
                f"ask{level}_price",
                f"ask{level}_volume",
                f"bid{level}_price",
                f"bid{level}_volume",
            ]
        )
    return columns


def price_columns() -> list[str]:
    return [TIMESTAMP, "midprice", "ask1_price", "bid1_price"]


def message_columns() -> list[str]:
    return [
        TIMESTAMP,
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


def trade_summary_columns() -> list[str]:
    return [
        TIMESTAMP,
        "trade_price_min",
        "trade_price_min_volume",
        "trade_price_max",
        "trade_price_max_volume",
        "trade_volume_total",
    ]


def orderbook_columns(levels: int = PAPER.lob_levels) -> list[str]:
    return [TIMESTAMP, *lob_columns(levels)]


@dataclass(frozen=True)
class LobDataset:
    """In-memory event-level data for one stock/day segment."""

    stock: str
    day: str
    orderbook: "pl.DataFrame"
    messages: "pl.DataFrame"
    trades: "pl.DataFrame"

    def __post_init__(self) -> None:
        import polars as pl

        if not isinstance(self.orderbook, pl.DataFrame):
            raise TypeError("orderbook must be a polars DataFrame")
        if not isinstance(self.messages, pl.DataFrame):
            raise TypeError("messages must be a polars DataFrame")
        if not isinstance(self.trades, pl.DataFrame):
            raise TypeError("trades must be a polars DataFrame")

        expected_orderbook = orderbook_columns()
        missing_orderbook = set(expected_orderbook) - set(self.orderbook.columns)
        if missing_orderbook:
            raise ValueError(f"missing orderbook columns: {sorted(missing_orderbook)}")

        missing_messages = set(message_columns()) - set(self.messages.columns)
        if missing_messages:
            raise ValueError(f"missing message columns: {sorted(missing_messages)}")

        missing_trades = set(trade_summary_columns()) - set(self.trades.columns)
        if missing_trades:
            raise ValueError(f"missing trade columns: {sorted(missing_trades)}")

        lengths = {self.orderbook.height, self.messages.height, self.trades.height}
        if len(lengths) != 1:
            raise ValueError("orderbook, messages, and trades must have identical lengths")


def assert_lob_window_shape(window: object) -> None:
    shape = getattr(window, "shape", None)
    if shape != PAPER.lob_window_shape:
        raise ValueError(f"LOB window shape must be {PAPER.lob_window_shape}, got {shape}")
