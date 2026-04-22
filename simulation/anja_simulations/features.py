from __future__ import annotations

import numpy as np
import pandas as pd


def _search_left(times_ns: np.ndarray, idx: int, window_s: int) -> int:
    left_ns = times_ns[idx] - int(window_s * 1_000_000_000)
    return int(np.searchsorted(times_ns, left_ns, side="left"))


def realized_volatility(series: np.ndarray) -> float:
    if len(series) < 2:
        return 0.0
    logp = np.log(np.clip(series, 1e-8, None))
    squared = np.square(np.diff(logp))
    return float(squared.mean() * 1e4)


def relative_strength_index(series: np.ndarray) -> float:
    if len(series) < 2:
        return 0.5
    pct = np.diff(series) / np.clip(series[:-1], 1e-8, None)
    gains = pct[pct > 0].sum()
    losses = -pct[pct < 0].sum()
    denom = gains + losses
    if denom <= 1e-8:
        return 0.5
    return float(gains / denom)


def build_market_features(timestamps: pd.DatetimeIndex, midprice: np.ndarray, msg: pd.DataFrame, rv_windows_s: list[int], rsi_windows_s: list[int], osi_windows_s: list[int]) -> np.ndarray:
    times_ns = timestamps.view("int64")
    n = len(midprice)
    feature_count = len(rv_windows_s) + len(rsi_windows_s) + 6 * len(osi_windows_s)
    out = np.zeros((n, feature_count), dtype=np.float32)
    msg_cols = [
        "market_buy_volume",
        "market_buy_n",
        "market_sell_volume",
        "market_sell_n",
        "limit_buy_volume",
        "limit_buy_n",
        "limit_sell_volume",
        "limit_sell_n",
        "withdraw_buy_volume",
        "withdraw_buy_n",
        "withdraw_sell_volume",
        "withdraw_sell_n",
    ]
    msg_values = msg[msg_cols].to_numpy(dtype=np.float64)
    msg_cumsum = np.vstack([np.zeros((1, msg_values.shape[1])), np.cumsum(msg_values, axis=0)])
    for idx in range(n):
        cursor = 0
        for window in rv_windows_s:
            left = _search_left(times_ns, idx, window)
            out[idx, cursor] = realized_volatility(midprice[left : idx + 1])
            cursor += 1
        for window in rsi_windows_s:
            left = _search_left(times_ns, idx, window)
            out[idx, cursor] = relative_strength_index(midprice[left : idx + 1])
            cursor += 1
        for window in osi_windows_s:
            left = _search_left(times_ns, idx, window)
            sums = msg_cumsum[idx + 1] - msg_cumsum[left]
            market_vol = (sums[0] - sums[2]) / max(sums[0] + sums[2], 1e-8)
            market_n = (sums[1] - sums[3]) / max(sums[1] + sums[3], 1e-8)
            limit_vol = (sums[4] - sums[6]) / max(sums[4] + sums[6], 1e-8)
            limit_n = (sums[5] - sums[7]) / max(sums[5] + sums[7], 1e-8)
            withdraw_vol = (sums[8] - sums[10]) / max(sums[8] + sums[10], 1e-8)
            withdraw_n = (sums[9] - sums[11]) / max(sums[9] + sums[11], 1e-8)
            out[idx, cursor : cursor + 6] = [market_vol, market_n, limit_vol, limit_n, withdraw_vol, withdraw_n]
            cursor += 6
    return out

