import numpy as np

from mlfcs_gapa.data.schema import lob_columns, message_columns, trade_summary_columns
from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.paper.constants import PAPER


def test_synthetic_dataset_matches_paper_shape() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=250, seed=7))

    assert dataset.orderbook.height == 250
    assert dataset.messages.height == 250
    assert dataset.trades.height == 250
    assert dataset.orderbook.columns == ["timestamp", *lob_columns()]
    assert dataset.messages.columns == message_columns()
    assert dataset.trades.columns == trade_summary_columns()


def test_synthetic_lob_prices_are_cross_free_and_cent_tick() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=250, seed=8))
    ask1 = dataset.orderbook["ask1_price"].to_numpy()
    bid1 = dataset.orderbook["bid1_price"].to_numpy()

    assert np.all(ask1 > bid1)
    assert np.allclose(np.round(ask1 * 100), ask1 * 100)
    assert np.allclose(np.round(bid1 * 100), bid1 * 100)


def test_synthetic_window_can_feed_attn_lob() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=100, seed=9))
    window = dataset.orderbook.select(lob_columns()).head(PAPER.window_length).to_numpy()
    assert window.shape == PAPER.lob_window_shape
