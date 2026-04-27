from __future__ import annotations

import pandas as pd

import piroth.config as config_module
from piroth.config import DiagnosticsConfig, SymbolSpec
from piroth.orderbook import FIFOOrderBook
from piroth.simulator import SyntheticMarketGenerator


def _tiny_config(monkeypatch) -> DiagnosticsConfig:
    old = config_module.DEFAULT_SYMBOLS["000001"]
    monkeypatch.setitem(
        config_module.DEFAULT_SYMBOLS,
        "000001",
        SymbolSpec(**{**old.__dict__, "events_per_day": 600}),
    )
    return DiagnosticsConfig(mode="smoke", symbol="000001", seed=1, num_days=1, train_days=0, test_days=1)


def test_generated_day_has_expected_fileshape(monkeypatch) -> None:
    config = _tiny_config(monkeypatch)
    generator = SyntheticMarketGenerator(config)
    day = generator.generate_day(generator.business_days()[0])

    assert set(["timestamp", "midprice", "ask1_price", "bid1_price"]).issubset(day.price.columns)
    assert set(["timestamp", "ask1_price", "ask1_volume"]).issubset(day.ask.columns)
    assert set(["timestamp", "bid1_price", "bid1_volume"]).issubset(day.bid.columns)
    assert set(["timestamp", "market_buy_volume", "limit_sell_n", "withdraw_buy_volume"]).issubset(day.msg.columns)
    assert set(["timestamp", "event_type", "agent_type"]).issubset(day.event_log.columns)
    assert set(["timestamp", "fair_value", "event_kind"]).issubset(day.latent.columns)
    assert len(day.price) == len(day.ask) == len(day.bid) == len(day.msg) == len(day.latent)
    assert len(day.event_log) >= len(day.price)
    assert day.depth_cube.shape[0] == len(day.price)


def test_generated_day_keeps_positive_spread(monkeypatch) -> None:
    config = _tiny_config(monkeypatch)
    generator = SyntheticMarketGenerator(config)
    day = generator.generate_day(generator.business_days()[0])

    assert (day.price["spread_ticks"] > 0).all()


def test_orderbook_depth_cache_tracks_add_cancel_and_trade() -> None:
    book = FIFOOrderBook(tick_size=0.01, levels=3)
    ts = pd.Timestamp("2019-11-01 10:00:00")
    bid = book.add_limit_order("bid", 999, 300, "a", "test", ts)
    book.add_limit_order("bid", 999, 200, "b", "test", ts)
    book.add_limit_order("ask", 1001, 400, "c", "test", ts)

    assert book.best_bid_tick() == 999
    assert book.best_ask_tick() == 1001
    assert book.aggregated_depth("bid", 999) == 500
    assert book.aggregated_depth("ask", 1001) == 400

    book.add_limit_order("bid", 1000, 100, "d", "test", ts)
    assert book.best_bid_tick() == 1000
    book.market_order("sell", 100, ts, "taker")
    assert book.best_bid_tick() == 999

    book.cancel_order("bid", 999, bid.order_id)
    assert book.aggregated_depth("bid", 999) == 200

    fills = book.market_order("buy", 150, ts, "taker")

    assert sum(fill.size for fill in fills) == 150
    assert book.aggregated_depth("ask", 1001) == 250

    book.add_limit_order("ask", 1002, 100, "agent-x", "test", ts)
    book.add_limit_order("bid", 998, 100, "agent-x", "test", ts)

    assert book.cancel_agent_orders("agent-x") == 2
    assert book.aggregated_depth("ask", 1002) == 0
    assert book.aggregated_depth("bid", 998) == 0
