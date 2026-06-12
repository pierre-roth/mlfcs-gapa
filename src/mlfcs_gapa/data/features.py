"""Paper-level labels and LOB preprocessing."""

from __future__ import annotations

import numpy as np

from mlfcs_gapa.paper.constants import PAPER


def midprice_direction_labels(
    midprices: np.ndarray,
    horizon: int = PAPER.midprice_horizon_events,
    threshold: float = PAPER.midprice_label_threshold,
) -> np.ndarray:
    """Create up/stationary/down labels from the paper's Equations 5-7.

    Output encoding is intentionally training-friendly:

    - `0`: down
    - `1`: stationary
    - `2`: up

    Entries without enough past or future context are set to `-1`.
    """

    prices = np.asarray(midprices, dtype=np.float64)
    labels = np.full(prices.shape, -1, dtype=np.int64)

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if prices.ndim != 1:
        raise ValueError("midprices must be one-dimensional")
    if len(prices) < (2 * horizon + 1):
        return labels

    for t in range(horizon, len(prices) - horizon):
        m_minus = prices[t - horizon + 1 : t + 1].mean()
        m_plus = prices[t + 1 : t + horizon + 1].mean()
        relative_move = (m_plus - m_minus) / m_minus
        if relative_move > threshold:
            labels[t] = 2
        elif relative_move < -threshold:
            labels[t] = 0
        else:
            labels[t] = 1

    return labels


def normalize_lob_window(lob_window: np.ndarray) -> np.ndarray:
    """Normalize one `(T, 40)` LOB window.

    Paper-faithful interpretation:

    - transform prices to a stationary relative-to-mid representation.
    - z-normalize transformed price columns inside the window.
    - max-normalize volume columns inside the window.

    The input column order must be the canonical paper order:
    `ask_price, ask_volume, bid_price, bid_volume` repeated for 10 levels.
    """

    window = np.asarray(lob_window, dtype=np.float32)
    if window.ndim != 2 or window.shape[1] != PAPER.lob_width:
        raise ValueError(f"expected LOB window shape (T, {PAPER.lob_width}), got {window.shape}")
    return _normalize_lob_windows_batch(window[None])[0]


def _normalize_lob_windows_batch(windows: np.ndarray) -> np.ndarray:
    """Normalize `(N, T, 40)` windows at once; one window is the `N=1` case."""

    output = windows.astype(np.float32, copy=True)
    width = output.shape[2]
    price_cols = np.array([*range(0, width, 4), *range(2, width, 4)])
    volume_cols = price_cols + 1

    mid = (output[:, :, 0] + output[:, :, 2]) / 2.0
    prices = output[:, :, price_cols] / (mid[:, :, None] + 1e-7) - 1.0
    output[:, :, price_cols] = (prices - prices.mean(axis=1, keepdims=True)) / (
        prices.std(axis=1, keepdims=True) + 1e-7
    )
    volumes = output[:, :, volume_cols]
    output[:, :, volume_cols] = volumes / (volumes.max(axis=1, keepdims=True) + 1e-7)
    return output


def build_lob_windows(
    lob_values: np.ndarray, window_length: int = PAPER.window_length
) -> np.ndarray:
    """Build rolling normalized LOB windows from canonical LOB values."""

    values = np.asarray(lob_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != PAPER.lob_width:
        raise ValueError(f"expected 2D LOB values with width {PAPER.lob_width}, got {values.shape}")
    if len(values) < window_length:
        return np.empty((0, window_length, PAPER.lob_width), dtype=np.float32)

    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_length, axis=0
    ).transpose(0, 2, 1)
    return _normalize_lob_windows_batch(windows)
