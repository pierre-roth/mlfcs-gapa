import numpy as np

from mlfcs_gapa.data.features import (
    build_lob_windows,
    midprice_direction_labels,
    normalize_lob_window,
)
from mlfcs_gapa.data.schema import lob_columns
from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.paper.constants import PAPER


def test_midprice_direction_labels_use_three_classes() -> None:
    midprices = np.array([100.0] * 20 + list(np.linspace(100.0, 101.0, 30)) + [101.0] * 20)
    labels = midprice_direction_labels(midprices, horizon=10, threshold=1e-5)

    valid = labels[labels >= 0]
    assert set(valid.tolist()) <= {0, 1, 2}
    assert 2 in valid
    assert 1 in valid


def test_normalize_lob_window_preserves_paper_shape() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=80, seed=22))
    values = dataset.orderbook.select(lob_columns()).to_numpy()
    window = normalize_lob_window(values[: PAPER.window_length])

    assert window.shape == PAPER.lob_window_shape
    assert np.isfinite(window).all()
    volume_cols = list(range(1, PAPER.lob_width, 4)) + list(range(3, PAPER.lob_width, 4))
    assert np.all(window[:, volume_cols] >= 0)
    assert np.all(window[:, volume_cols] <= 1.000001)


def test_normalize_lob_window_supports_longer_pretrain_windows() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=130, seed=24))
    values = dataset.orderbook.select(lob_columns()).to_numpy()
    window = normalize_lob_window(values[:100])

    assert window.shape == (100, PAPER.lob_width)
    assert np.isfinite(window).all()


def test_build_lob_windows_shape() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=75, seed=23))
    values = dataset.orderbook.select(lob_columns()).to_numpy()
    windows = build_lob_windows(values)

    assert windows.shape == (26, PAPER.window_length, PAPER.lob_width)
