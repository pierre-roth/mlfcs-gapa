import numpy as np
import pytest

from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.env.actions import continuous_action_to_quote
from mlfcs_gapa.env.replay import Account, Fill, HistoricalReplay, compute_episode_metrics


def test_account_updates_cash_inventory_and_value() -> None:
    account = Account()
    account.apply_fill(Fill(trade_price=9.99, trade_volume=100), mid_price=10.0)

    assert account.inventory == 100
    assert account.cash == pytest.approx(-999.0)
    assert account.value == pytest.approx(1.0)
    assert account.buy_notional == pytest.approx(999.0)

    account.apply_fill(Fill(trade_price=10.01, trade_volume=-100), mid_price=10.0)
    assert account.inventory == 0
    assert account.cash == pytest.approx(2.0)
    assert account.value == pytest.approx(2.0)


def test_crossing_quote_executes_against_counterparty_best_price() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=80, seed=31))
    replay = HistoricalReplay(dataset, rng=np.random.default_rng(1))
    bid1, ask1 = replay.best_bid_ask(10)

    crossing_bid = continuous_action_to_quote(
        np.array([0.0, 0.0]), mid_price=ask1 + 0.1, inventory=0
    )
    fill = replay.match(11, crossing_bid)

    assert fill.trade_volume == 100
    assert fill.trade_price == pytest.approx(ask1)

    crossing_ask = continuous_action_to_quote(
        np.array([0.0, 0.0]), mid_price=bid1 - 0.1, inventory=0
    )
    fill = replay.match(11, crossing_ask)

    assert fill.trade_volume == -100
    assert fill.trade_price == pytest.approx(bid1)


def test_close_position_uses_counterparty_price() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=80, seed=32))
    replay = HistoricalReplay(dataset, rng=np.random.default_rng(1))
    bid1, ask1 = replay.best_bid_ask(20)

    short = Account(inventory=-300)
    fill = replay.close_position(21, short)
    assert fill.trade_volume == 300
    assert fill.trade_price == pytest.approx(ask1)

    long = Account(inventory=300)
    fill = replay.close_position(21, long)
    assert fill.trade_volume == -300
    assert fill.trade_price == pytest.approx(bid1)


def test_episode_metrics_match_paper_definitions() -> None:
    metrics = compute_episode_metrics(
        values=[0.0, 1.0, 2.0],
        inventories=[0, 100, 200],
        quoted_spreads=[0.02, 0.02],
        buy_notional=1_000.0,
    )

    assert metrics.pnl == pytest.approx(2.0)
    assert metrics.nd_pnl == pytest.approx(100.0, rel=1e-5)
    assert metrics.pnl_map == pytest.approx(2.0 / 100.0)
    assert metrics.profit_ratio == pytest.approx(0.002)
