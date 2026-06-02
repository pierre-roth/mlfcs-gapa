"""Build Attn-LOB pretraining arrays from canonical LOB datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs_gapa.data.features import build_lob_windows, midprice_direction_labels
from mlfcs_gapa.data.schema import LobDataset, lob_columns
from mlfcs_gapa.paper.constants import PAPER


@dataclass(frozen=True)
class PretrainArrays:
    x: np.ndarray
    y: np.ndarray


def build_pretrain_arrays(
    dataset: LobDataset,
    *,
    window_length: int = PAPER.window_length,
) -> PretrainArrays:
    """Build `(N, window_length, 40)` windows and aligned direction labels.

    A window ending at event `t` receives the mid-price direction label for
    event `t`, matching the paper's "past T timestamps predict future direction"
    setup.
    """

    lob_values = dataset.orderbook.select(lob_columns()).to_numpy()
    windows = build_lob_windows(lob_values, window_length=window_length)

    ask1 = dataset.orderbook["ask1_price"].to_numpy()
    bid1 = dataset.orderbook["bid1_price"].to_numpy()
    midprices = (ask1 + bid1) / 2.0
    labels = midprice_direction_labels(midprices)
    aligned_labels = labels[window_length - 1 :]

    valid = aligned_labels >= 0
    return PretrainArrays(
        x=windows[valid].astype(np.float32),
        y=aligned_labels[valid].astype(np.int64),
    )
