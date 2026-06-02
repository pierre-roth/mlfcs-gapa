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

    output = window.copy()
    ask1 = output[:, 0]
    bid1 = output[:, 2]
    mid = (ask1 + bid1) / 2.0

    price_cols = list(range(0, PAPER.lob_width, 4)) + list(range(2, PAPER.lob_width, 4))
    volume_cols = list(range(1, PAPER.lob_width, 4)) + list(range(3, PAPER.lob_width, 4))

    for col in price_cols:
        output[:, col] = output[:, col] / (mid + 1e-7) - 1.0
        std = output[:, col].std()
        output[:, col] = (output[:, col] - output[:, col].mean()) / (std + 1e-7)

    for col in volume_cols:
        max_value = output[:, col].max()
        output[:, col] = output[:, col] / (max_value + 1e-7)

    return output


def build_lob_windows(
    lob_values: np.ndarray, window_length: int = PAPER.window_length
) -> np.ndarray:
    """Build rolling LOB windows from canonical LOB values."""

    values = np.asarray(lob_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != PAPER.lob_width:
        raise ValueError(f"expected 2D LOB values with width {PAPER.lob_width}, got {values.shape}")
    if len(values) < window_length:
        return np.empty((0, window_length, PAPER.lob_width), dtype=np.float32)

    windows = np.empty(
        (len(values) - window_length + 1, window_length, PAPER.lob_width), dtype=np.float32
    )
    for end in range(window_length, len(values) + 1):
        windows[end - window_length] = normalize_lob_window(values[end - window_length : end])
    return windows
