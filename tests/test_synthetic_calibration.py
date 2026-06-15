"""Acceptance tests for the synthetic market calibration.

These bounds encode the microstructure properties that make the paper's
Table II reproducible: lively but not explosive price paths, bid-ask bounce
(negative lag-1 return autocorrelation), mild adverse selection, realistic
passive fill rates, and profitable passive market making for the simple
baselines. If one of these fails after a generator change, the full
replication will not line up with the paper.
"""

import numpy as np
import pytest

from mlfcs_gapa.data.features import calibrate_label_threshold, midprice_direction_labels
from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.env.baselines import (
    AvellanedaStoikovStrategy,
    FixedLevelStrategy,
    estimate_episode_volatility,
    evaluate_quote_strategy,
)

PROFILES = [("000001", 16.45), ("000858", 130.0), ("002415", 35.0)]
N_EVENTS = 8_000
SEED = 11
TICK = 0.01
EPISODE = 2_000


@pytest.fixture(scope="module", params=PROFILES, ids=[stock for stock, _ in PROFILES])
def dataset(request):
    stock, base_price = request.param
    return generate_synthetic_lob_day(
        SyntheticLobConfig(stock=stock, base_price=base_price, n_events=N_EVENTS, seed=SEED)
    )


def _mid(dataset) -> np.ndarray:
    ask1 = dataset.orderbook["ask1_price"].to_numpy()
    bid1 = dataset.orderbook["bid1_price"].to_numpy()
    return (ask1 + bid1) / 2.0


def test_mid_price_is_lively_but_not_explosive(dataset) -> None:
    mid = _mid(dataset)
    changes = np.diff(mid)
    move_fraction = float((changes != 0).mean())
    assert 0.05 <= move_fraction <= 0.40

    episode_stds = [
        np.std(mid[start : start + EPISODE]) / TICK
        for start in range(0, len(mid) - EPISODE, 1_000)
    ]
    assert 1.5 <= float(np.mean(episode_stds)) <= 40.0

    episode_ranges = [
        (mid[start : start + EPISODE].max() - mid[start : start + EPISODE].min())
        / mid[start]
        for start in range(0, len(mid) - EPISODE, 1_000)
    ]
    assert 0.001 <= float(np.mean(episode_ranges)) <= 0.03


def test_returns_show_bid_ask_bounce_not_momentum(dataset) -> None:
    changes = np.diff(_mid(dataset))
    lag1 = float(np.corrcoef(changes[:-1], changes[1:])[0, 1])
    lag5 = float(np.corrcoef(changes[:-5], changes[5:])[0, 1])
    assert -0.5 < lag1 < -0.01
    assert abs(lag5) < 0.10


def test_passive_fills_suffer_only_mild_adverse_selection(dataset) -> None:
    mid = _mid(dataset)
    ask1 = dataset.orderbook["ask1_price"].to_numpy()
    spread_ticks = float(np.mean((ask1 - _mid(dataset)) * 2.0 / TICK))
    print_price = dataset.trades["trade_price_max"].to_numpy()
    print_volume = dataset.trades["trade_price_max_volume"].to_numpy()

    horizon = 20
    index = np.arange(len(mid) - horizon)
    ask_prints = print_volume[: len(index)] > 0
    drift_ticks = float(
        ((mid[index + horizon] - print_price[: len(index)])[ask_prints]).mean() / TICK
    )
    # Negative drift means the passive seller keeps part of the spread on
    # average; strongly positive drift means toxic flow.
    assert -1.5 * spread_ticks < drift_ticks < 0.5


def test_calibrated_labels_are_balanced(dataset) -> None:
    mid = _mid(dataset)
    alpha = calibrate_label_threshold(mid)
    labels = midprice_direction_labels(mid, threshold=alpha)
    valid = labels[labels >= 0]
    shares = np.bincount(valid, minlength=3) / len(valid)
    assert shares.min() >= 0.15
    assert shares.max() <= 0.55


def test_passive_baselines_fill_realistically_and_profit(dataset) -> None:
    fixed_pnls, fill_counts, as_pnls = [], [], []
    sigma = max(estimate_episode_volatility(dataset, EPISODE), 1e-6)
    for episode_index in range(3):
        start = episode_index * EPISODE
        metrics, log_rows = evaluate_quote_strategy(
            dataset,
            FixedLevelStrategy(level=1),
            episode_start=start,
            episode_events=EPISODE,
            seed=SEED + episode_index,
        )
        fixed_pnls.append(metrics["pnl"])
        fill_counts.append(sum(1 for row in log_rows if row["trade_volume"] != 0))
        as_metrics, _ = evaluate_quote_strategy(
            dataset,
            AvellanedaStoikovStrategy(sigma=sigma),
            episode_start=start,
            episode_events=EPISODE,
            seed=SEED + episode_index,
        )
        as_pnls.append(as_metrics["pnl"])

    assert 5 <= float(np.mean(fill_counts)) <= 130
    assert float(np.mean(fixed_pnls)) > 0.0
    assert float(np.mean(as_pnls)) > -5.0
