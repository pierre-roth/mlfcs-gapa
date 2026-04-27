from __future__ import annotations

import numpy as np
import pandas as pd

from piroth.paper_features import LOB_COLUMNS, lob_tensor_at, lob_tensor_from_values


def test_lob_columns_match_author_ask_then_bid_order() -> None:
    assert LOB_COLUMNS[:4] == ["ask1_price", "ask1_volume", "ask2_price", "ask2_volume"]
    assert LOB_COLUMNS[18:22] == ["ask10_price", "ask10_volume", "bid1_price", "bid1_volume"]
    assert LOB_COLUMNS[-2:] == ["bid10_price", "bid10_volume"]


def test_lob_tensor_from_values_matches_dataframe_path() -> None:
    rows = []
    for idx in range(64):
        row: dict[str, float] = {}
        mid = 10.0 + idx * 0.001
        for level in range(1, 11):
            row[f"ask{level}_price"] = mid + level * 0.01
            row[f"ask{level}_volume"] = 1000 + idx + level
            row[f"bid{level}_price"] = mid - level * 0.01
            row[f"bid{level}_volume"] = 1200 + 2 * idx + level
        rows.append(row)
    orderbook = pd.DataFrame(rows, columns=LOB_COLUMNS)

    from_frame = lob_tensor_at(orderbook, event_idx=32, lookback=50)
    from_values = lob_tensor_from_values(orderbook[LOB_COLUMNS].to_numpy(dtype=np.float32), event_idx=32, lookback=50)

    np.testing.assert_allclose(from_values, from_frame, rtol=1e-6, atol=1e-7)
