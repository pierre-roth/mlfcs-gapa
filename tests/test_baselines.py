import numpy as np

from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.env.baselines import (
    AvellanedaStoikovStrategy,
    FixedLevelStrategy,
    RandomLevelStrategy,
    estimate_event_volatility,
    evaluate_quote_strategy,
)
from mlfcs_gapa.env.replay import Account, HistoricalReplay


def test_fixed_and_random_baselines_quote_lob_levels() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=120, seed=71))
    replay = HistoricalReplay(dataset)

    fixed = FixedLevelStrategy(level=2).quote(replay, Account(), 50, 0.5)
    row = dataset.orderbook.row(50, named=True)
    assert fixed.ask_price == row["ask2_price"]
    assert fixed.bid_price == row["bid2_price"]

    random = RandomLevelStrategy(max_level=5, seed=1).quote(replay, Account(), 50, 0.5)
    ask_levels = {row[f"ask{i}_price"] for i in range(1, 6)}
    bid_levels = {row[f"bid{i}_price"] for i in range(1, 6)}
    assert random.ask_price in ask_levels
    assert random.bid_price in bid_levels


def test_avellaneda_stoikov_quote_skews_with_inventory() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=120, seed=72))
    replay = HistoricalReplay(dataset)
    sigma = estimate_event_volatility(dataset)
    strategy = AvellanedaStoikovStrategy(sigma=max(sigma, 0.01), gamma=0.1, kappa=1.5)

    flat = strategy.quote(replay, Account(inventory=0), 50, 0.2)
    long = strategy.quote(replay, Account(inventory=500), 50, 0.2)

    assert long.reservation_price < flat.reservation_price
    assert long.ask_price <= flat.ask_price


def test_evaluate_quote_strategy_returns_metrics_and_log() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=220, seed=73))
    metrics, log_rows = evaluate_quote_strategy(
        dataset,
        FixedLevelStrategy(level=1),
        episode_events=120,
        latency_events=1,
        seed=1,
    )

    assert metrics["method"] == "Fixed_1"
    assert "pnl" in metrics
    assert "nd_pnl" in metrics
    assert len(log_rows) > 0
    assert np.isfinite(metrics["pnl"])
