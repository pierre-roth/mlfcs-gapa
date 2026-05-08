from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LOB_COLUMNS: list[str] = [
    *[column for level in range(1, 11) for column in (f"ask{level}_price", f"ask{level}_volume")],
    *[column for level in range(1, 11) for column in (f"bid{level}_price", f"bid{level}_volume")],
]

MSG_COLUMNS: list[str] = [
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


@dataclass(frozen=True)
class PaperState:
    lob_state: np.ndarray
    market_state: np.ndarray
    agent_state: np.ndarray


def combine_orderbook(ask: pd.DataFrame, bid: pd.DataFrame) -> pd.DataFrame:
    bid_no_ts = bid.drop(columns=["timestamp"], errors="ignore")
    frame = pd.concat([ask.reset_index(drop=True), bid_no_ts.reset_index(drop=True)], axis=1)
    return frame[["timestamp", *LOB_COLUMNS]]


def normalize_lob_window(window: pd.DataFrame, *, price_z_norm: bool = False) -> np.ndarray:
    return normalize_lob_values(window[LOB_COLUMNS].to_numpy(dtype=np.float32), price_z_norm=price_z_norm)


def normalize_lob_values(values: np.ndarray, *, price_z_norm: bool = False) -> np.ndarray:
    data = values.astype(np.float32, copy=True)
    mid = (data[:, 0].astype(np.float64) + data[:, 20].astype(np.float64)) / 2.0
    mid = np.clip(mid, 1e-8, None)
    price_columns: list[int] = []
    for level in range(1, 11):
        ask_base = (level - 1) * 2
        bid_base = 20 + (level - 1) * 2
        data[:, ask_base] = data[:, ask_base] / mid - 1.0
        data[:, bid_base] = data[:, bid_base] / mid - 1.0
        price_columns.extend([ask_base, bid_base])
        ask_v = data[:, ask_base + 1]
        bid_v = data[:, bid_base + 1]
        data[:, ask_base + 1] = ask_v / max(float(np.max(ask_v)), 1.0)
        data[:, bid_base + 1] = bid_v / max(float(np.max(bid_v)), 1.0)
    if price_z_norm:
        for column in price_columns:
            series = data[:, column]
            data[:, column] = (series - float(np.mean(series))) / (float(np.std(series, ddof=1)) + 1e-7)
    return data


def lob_tensor_at(orderbook: pd.DataFrame, event_idx: int, lookback: int, *, price_z_norm: bool = False) -> np.ndarray:
    return lob_tensor_from_values(orderbook[LOB_COLUMNS].to_numpy(dtype=np.float32), event_idx, lookback, price_z_norm=price_z_norm)


def lob_tensor_from_values(values: np.ndarray, event_idx: int, lookback: int, *, price_z_norm: bool = False) -> np.ndarray:
    start = max(event_idx - lookback, 0)
    window = values[start:event_idx]
    if len(window) < lookback:
        if len(window):
            pad = np.repeat(window[[0]], lookback - len(window), axis=0)
            window = np.concatenate([pad, window], axis=0)
        else:
            window = np.repeat(values[[0]], lookback, axis=0)
    return normalize_lob_values(window, price_z_norm=price_z_norm).reshape(lookback, 40, 1)


def midprice_direction_labels(midprice: pd.Series, horizon: int, threshold: float) -> pd.Series:
    past = midprice.rolling(window=horizon, min_periods=horizon).mean()
    future = past.shift(-horizon)
    pct_change = (future - past) / past.clip(lower=1e-8)
    labels = pd.Series(np.nan, index=midprice.index, dtype="float64")
    labels[pct_change >= threshold] = 0
    labels[(pct_change < threshold) & (pct_change > -threshold)] = 1
    labels[pct_change <= -threshold] = 2
    return labels


def realized_volatility(midprice: pd.Series, resample: str = "s") -> float:
    data = midprice.resample(resample).last().ffill()
    log_ret = np.log(data.clip(lower=1e-8)).diff()
    return float(np.square(log_ret.dropna()).sum())


def relative_strength_index(midprice: pd.Series) -> float:
    data = midprice.resample("s").last().ffill().pct_change()
    gain = float(data[data > 0].sum())
    loss = float(-data[data < 0].sum())
    return gain / (gain + loss) if gain or loss else 0.5


def order_strength_index(msg: pd.DataFrame) -> list[float]:
    frame = ensure_paper_msg(msg)
    mbv = frame["market_buy_volume"].sum()
    msv = frame["market_sell_volume"].sum()
    mbn = frame["market_buy_n"].sum()
    msn = frame["market_sell_n"].sum()
    lbv = frame["limit_buy_volume"].sum()
    lsv = frame["limit_sell_volume"].sum()
    lbn = frame["limit_buy_n"].sum()
    lsn = frame["limit_sell_n"].sum()
    wbv = frame["withdraw_buy_volume"].sum()
    wsv = frame["withdraw_sell_volume"].sum()
    wbn = frame["withdraw_buy_n"].sum()
    wsn = frame["withdraw_sell_n"].sum()
    return [
        _imbalance(mbv, msv),
        _imbalance(mbn, msn),
        _imbalance(lbv, lsv),
        _imbalance(lbn, lsn),
        _imbalance(wbv, wsv),
        _imbalance(wbn, wsn),
    ]


def dynamic_market_state(price: pd.DataFrame, msg: pd.DataFrame, event_idx: int) -> np.ndarray:
    timestamp = pd.Timestamp(price.iloc[event_idx]["timestamp"])
    price_ts = price.set_index(pd.to_datetime(price["timestamp"]))
    msg_ts = ensure_paper_msg(msg).set_index(pd.to_datetime(msg["timestamp"]))
    return dynamic_market_state_from_indexed(price_ts, msg_ts, timestamp)


def dynamic_market_state_matrix(price: pd.DataFrame, msg: pd.DataFrame) -> np.ndarray:
    price_ts = price.set_index(pd.to_datetime(price["timestamp"])).sort_index()
    msg_ts = ensure_paper_msg(msg).set_index(pd.to_datetime(msg["timestamp"])).sort_index()
    mid = price_ts["midprice"].astype("float64").clip(lower=1e-8)
    log_ret = np.log(mid).diff().fillna(0.0)
    pct_ret = mid.pct_change().fillna(0.0)
    values: list[np.ndarray] = []
    for seconds in (300, 600, 1800):
        rv = log_ret.pow(2).rolling(f"{seconds}s", min_periods=1).sum() * 1e4
        values.append(rv.fillna(0.0).to_numpy(dtype=np.float32))
    for seconds in (300, 600, 1800):
        gain = pct_ret.clip(lower=0.0).rolling(f"{seconds}s", min_periods=1).sum()
        loss = (-pct_ret.clip(upper=0.0)).rolling(f"{seconds}s", min_periods=1).sum()
        rsi = gain / (gain + loss + 1e-7)
        values.append(rsi.fillna(0.5).to_numpy(dtype=np.float32))
    for seconds in (10, 60, 300):
        rolled = msg_ts[MSG_COLUMNS].rolling(f"{seconds}s", min_periods=1).sum().fillna(0.0)
        for column in _osi_columns(rolled):
            values.append(column.to_numpy(dtype=np.float32))
    return np.column_stack(values).astype(np.float32)


def dynamic_market_state_from_indexed(price_ts: pd.DataFrame, msg_ts: pd.DataFrame, timestamp: pd.Timestamp) -> np.ndarray:
    values: list[float] = []
    for seconds in (300, 600, 1800):
        segment = price_ts.loc[(price_ts.index <= timestamp) & (price_ts.index >= timestamp - pd.Timedelta(seconds=seconds)), "midprice"]
        values.append(realized_volatility(segment) * 1e4 if len(segment) else 0.0)
    for seconds in (300, 600, 1800):
        segment = price_ts.loc[(price_ts.index <= timestamp) & (price_ts.index >= timestamp - pd.Timedelta(seconds=seconds)), "midprice"]
        values.append(relative_strength_index(segment) if len(segment) else 0.5)
    for seconds in (10, 60, 300):
        segment = msg_ts.loc[(msg_ts.index <= timestamp) & (msg_ts.index >= timestamp - pd.Timedelta(seconds=seconds))]
        values.extend(order_strength_index(segment))
    return np.asarray(values, dtype=np.float32)


def agent_state(inventory: int, event_idx: int, episode_length: int, lot_size: int, max_inventory_units: int) -> np.ndarray:
    inv = inventory / max(max_inventory_units * lot_size, 1)
    progress = event_idx / max(episode_length, 1)
    return np.asarray([inv] * 12 + [progress] * 12, dtype=np.float32)


def ensure_paper_msg(msg: pd.DataFrame) -> pd.DataFrame:
    frame = msg.copy()
    if "timestamp" not in frame.columns:
        frame = frame.reset_index().rename(columns={"index": "timestamp"})
    for column in MSG_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0
    return frame[["timestamp", *MSG_COLUMNS]]


def _imbalance(left: float, right: float) -> float:
    return float((left - right) / (left + right + 1e-7))


def _osi_columns(frame: pd.DataFrame) -> list[pd.Series]:
    return [
        _imbalance_series(frame["market_buy_volume"], frame["market_sell_volume"]),
        _imbalance_series(frame["market_buy_n"], frame["market_sell_n"]),
        _imbalance_series(frame["limit_buy_volume"], frame["limit_sell_volume"]),
        _imbalance_series(frame["limit_buy_n"], frame["limit_sell_n"]),
        _imbalance_series(frame["withdraw_buy_volume"], frame["withdraw_sell_volume"]),
        _imbalance_series(frame["withdraw_buy_n"], frame["withdraw_sell_n"]),
    ]


def _imbalance_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left - right) / (left + right + 1e-7)
