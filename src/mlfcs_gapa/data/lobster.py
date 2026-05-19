"""LOBSTER CSV adapter for the canonical paper-replication schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from mlfcs_gapa.data.schema import LobDataset
from mlfcs_gapa.paper.constants import PAPER


MESSAGE_COLUMNS = ["seconds", "event_type", "order_id", "size", "price", "direction"]


def load_lobster_csv(
    *,
    message_path: Path,
    orderbook_path: Path,
    stock: str,
    day: str,
    levels: int = PAPER.lob_levels,
    price_scale: float = 10_000.0,
) -> LobDataset:
    """Load LOBSTER message/orderbook CSV files.

    LOBSTER rows are event-aligned: row `k` in the message file describes the
    update that produced row `k` in the orderbook file.
    """

    if levels < PAPER.lob_levels:
        raise ValueError(f"at least {PAPER.lob_levels} levels are required")

    messages_raw = pl.read_csv(message_path, has_header=False, new_columns=MESSAGE_COLUMNS)
    orderbook_raw = pl.read_csv(
        orderbook_path,
        has_header=False,
        new_columns=_lobster_orderbook_columns(levels),
    )
    if messages_raw.height != orderbook_raw.height:
        raise ValueError("LOBSTER message and orderbook files must have identical row counts")

    timestamps = _timestamps_from_seconds(day, messages_raw["seconds"].to_numpy())
    orderbook = _canonical_orderbook(orderbook_raw, timestamps, levels, price_scale)
    messages = _canonical_messages(messages_raw, timestamps)
    trades = _canonical_trades(messages_raw, timestamps, price_scale)
    return LobDataset(stock=stock, day=day, orderbook=orderbook, messages=messages, trades=trades)


def _lobster_orderbook_columns(levels: int) -> list[str]:
    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
    return columns


def _timestamps_from_seconds(day: str, seconds: np.ndarray) -> np.ndarray:
    seconds = np.asarray(seconds, dtype=np.float64)
    nanoseconds = np.rint(seconds * 1_000_000_000).astype("timedelta64[ns]")
    return np.datetime64(day, "ns") + nanoseconds


def _canonical_orderbook(
    orderbook_raw: pl.DataFrame,
    timestamps: np.ndarray,
    levels: int,
    price_scale: float,
) -> pl.DataFrame:
    columns: dict[str, object] = {"timestamp": timestamps}
    for level in range(1, PAPER.lob_levels + 1):
        columns[f"ask{level}_price"] = (
            orderbook_raw[f"ask_price_{level}"].to_numpy().astype(np.float64) / price_scale
        )
        columns[f"ask{level}_volume"] = orderbook_raw[f"ask_size_{level}"].to_numpy()
        columns[f"bid{level}_price"] = (
            orderbook_raw[f"bid_price_{level}"].to_numpy().astype(np.float64) / price_scale
        )
        columns[f"bid{level}_volume"] = orderbook_raw[f"bid_size_{level}"].to_numpy()

    del levels
    return pl.DataFrame(columns)


def _canonical_messages(messages_raw: pl.DataFrame, timestamps: np.ndarray) -> pl.DataFrame:
    event_type = messages_raw["event_type"].to_numpy()
    direction = messages_raw["direction"].to_numpy()
    size = messages_raw["size"].to_numpy()

    is_limit = event_type == 1
    is_withdraw = np.isin(event_type, [2, 3])
    is_trade = np.isin(event_type, [4, 5])

    limit_buy = np.where(is_limit & (direction == 1), size, 0)
    limit_sell = np.where(is_limit & (direction == -1), size, 0)
    withdraw_buy = np.where(is_withdraw & (direction == 1), size, 0)
    withdraw_sell = np.where(is_withdraw & (direction == -1), size, 0)

    # In LOBSTER executions, direction is the side of the resting limit order.
    # A sell limit execution corresponds to an aggressive market buy, and vice versa.
    market_buy = np.where(is_trade & (direction == -1), size, 0)
    market_sell = np.where(is_trade & (direction == 1), size, 0)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "market_buy_volume": market_buy,
            "market_buy_n": (market_buy > 0).astype(np.int64),
            "market_sell_volume": market_sell,
            "market_sell_n": (market_sell > 0).astype(np.int64),
            "limit_buy_volume": limit_buy,
            "limit_buy_n": (limit_buy > 0).astype(np.int64),
            "limit_sell_volume": limit_sell,
            "limit_sell_n": (limit_sell > 0).astype(np.int64),
            "withdraw_buy_volume": withdraw_buy,
            "withdraw_buy_n": (withdraw_buy > 0).astype(np.int64),
            "withdraw_sell_volume": withdraw_sell,
            "withdraw_sell_n": (withdraw_sell > 0).astype(np.int64),
        }
    )


def _canonical_trades(
    messages_raw: pl.DataFrame,
    timestamps: np.ndarray,
    price_scale: float,
) -> pl.DataFrame:
    event_type = messages_raw["event_type"].to_numpy()
    is_trade = np.isin(event_type, [4, 5])
    price = messages_raw["price"].to_numpy().astype(np.float64) / price_scale
    size = messages_raw["size"].to_numpy()
    trade_price = np.where(is_trade, price, 0.0)
    trade_volume = np.where(is_trade, size, 0)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "trade_price_min": trade_price,
            "trade_price_min_volume": trade_volume,
            "trade_price_max": trade_price,
            "trade_price_max_volume": trade_volume,
            "trade_volume_total": trade_volume,
        }
    )
