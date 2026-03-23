from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .utils import cumulative_window_sums, rolling_left_bounds

LOB_COLUMNS = [
    item
    for level in range(1, 11)
    for item in (
        f"ask{level}_price",
        f"ask{level}_volume",
        f"bid{level}_price",
        f"bid{level}_volume",
    )
]

MSG_COLUMNS = [
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


@dataclass
class DayData:
    symbol: str
    day: str
    timestamps: pd.DatetimeIndex
    lob: np.ndarray
    midprice: np.ndarray
    ask1: np.ndarray
    bid1: np.ndarray
    spread: np.ndarray
    dynamic: np.ndarray
    handcrafted: np.ndarray
    trades_by_index: dict[int, pd.DataFrame]
    trade_indices: np.ndarray
    normalized_lob: np.ndarray | None = None
    norm_mean: np.ndarray | None = None
    norm_std: np.ndarray | None = None

    def valid_label_indices(self, lookback: int, horizon: int) -> np.ndarray:
        return np.arange(lookback - 1, len(self.midprice) - horizon)


def discover_days(data_dir: str | Path, symbol: str) -> list[str]:
    root = Path(data_dir) / symbol
    if not root.exists():
        raise FileNotFoundError(f"Missing symbol directory: {root}")
    days = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not days:
        raise RuntimeError(f"No trading days found in {root}")
    return days


def split_days(days: list[str], train_days: int, val_days: int, test_days: int) -> tuple[list[str], list[str], list[str]]:
    required = train_days + val_days + test_days
    if len(days) < required:
        raise RuntimeError(f"Need at least {required} days, found {len(days)}: {days}")
    selected = days[:required]
    return (
        selected[:train_days],
        selected[train_days : train_days + val_days],
        selected[train_days + val_days : required],
    )


def _row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle) - 1


def _skiprows_for_limit(total_rows: int, max_rows: int | None):
    if max_rows is None or total_rows <= max_rows:
        return None
    stride = max(1, int(np.ceil(total_rows / max_rows)))
    return lambda idx: idx > 0 and ((idx - 1) % stride != 0)


def _read_frame(path: Path, total_rows: int, max_rows: int | None, head_only: bool = False) -> pd.DataFrame:
    if head_only and max_rows is not None:
        return pd.read_csv(path, nrows=max_rows)
    skiprows = _skiprows_for_limit(total_rows, max_rows)
    return pd.read_csv(path, skiprows=skiprows)


def _build_trade_index_map(timestamps: pd.DatetimeIndex, trades: pd.DataFrame) -> dict[int, pd.DataFrame]:
    if trades.empty:
        return {}
    ts_to_index = {ts: idx for idx, ts in enumerate(timestamps)}
    trades = trades[trades["timestamp"].isin(ts_to_index)]
    grouped: dict[int, pd.DataFrame] = {}
    for ts, group in trades.groupby("timestamp"):
        grouped[ts_to_index[ts]] = group.reset_index(drop=True)
    return grouped


def _midprice_labels(midprice: np.ndarray, horizon: int, alpha: float) -> np.ndarray:
    labels = np.full(len(midprice), -100, dtype=np.int64)
    for idx in range(horizon, len(midprice) - horizon):
        past = midprice[idx - horizon : idx].mean()
        future = midprice[idx : idx + horizon].mean()
        ratio = (future - past) / max(abs(past), 1e-8)
        if ratio > alpha:
            labels[idx] = 2
        elif ratio < -alpha:
            labels[idx] = 0
        else:
            labels[idx] = 1
    return labels


def _rolling_market_features(timestamps: pd.DatetimeIndex, midprice: np.ndarray, msg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ts_ns = timestamps.asi8
    log_mid = np.log(np.maximum(midprice, 1e-8))
    returns = np.diff(log_mid, prepend=log_mid[0])
    sq_returns = returns**2
    gains = np.maximum(np.diff(midprice, prepend=midprice[0]), 0.0)
    losses = np.maximum(np.diff(midprice, prepend=midprice[0]) * -1.0, 0.0)
    rv_windows = [300, 600, 1800]
    rsi_windows = [300, 600, 1800]
    osi_windows = [10, 60, 300]

    market_features: list[np.ndarray] = []
    for window in rv_windows:
        left = rolling_left_bounds(ts_ns, window)
        cumsum = np.concatenate([[0.0], np.cumsum(sq_returns, dtype=np.float64)])
        right = np.arange(len(midprice)) + 1
        values = cumsum[right] - cumsum[np.clip(left + 1, 0, len(midprice))]
        market_features.append(np.sqrt(np.maximum(values, 0.0)) * 1e4)
    for window in rsi_windows:
        left = rolling_left_bounds(ts_ns, window)
        gain_sum = np.concatenate([[0.0], np.cumsum(gains, dtype=np.float64)])
        loss_sum = np.concatenate([[0.0], np.cumsum(losses, dtype=np.float64)])
        right = np.arange(len(midprice)) + 1
        gains_w = gain_sum[right] - gain_sum[np.clip(left + 1, 0, len(midprice))]
        losses_w = loss_sum[right] - loss_sum[np.clip(left + 1, 0, len(midprice))]
        market_features.append(gains_w / np.maximum(gains_w + losses_w, 1e-8))
    market = np.stack(market_features, axis=1).astype(np.float32)

    msg_values = msg[MSG_COLUMNS].to_numpy(dtype=np.float64)
    osi_features: list[np.ndarray] = []
    pair_indices = [
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (8, 10),
        (9, 11),
    ]
    for window in osi_windows:
        left = rolling_left_bounds(ts_ns, window)
        sums = cumulative_window_sums(msg_values, left)
        for buy_idx, sell_idx in pair_indices:
            buy = sums[:, buy_idx]
            sell = sums[:, sell_idx]
            osi_features.append(((buy - sell) / np.maximum(buy + sell, 1e-8)).astype(np.float32))
    dynamic = np.concatenate([market, np.stack(osi_features, axis=1)], axis=1).astype(np.float32)
    return dynamic, market.astype(np.float32)


def _handcrafted_features(lob: np.ndarray, spread: np.ndarray, midprice: np.ndarray, market_only: np.ndarray) -> np.ndarray:
    ask1_volume = lob[:, 1]
    bid1_volume = lob[:, 3]
    imbalance = (bid1_volume - ask1_volume) / np.maximum(bid1_volume + ask1_volume, 1e-8)
    mid_return = np.diff(midprice, prepend=midprice[0]) / np.maximum(midprice, 1e-8)
    return np.stack(
        [
            spread,
            imbalance,
            mid_return,
            market_only[:, 0],
            market_only[:, 3],
            market_only[:, 4],
            market_only[:, 5],
        ],
        axis=1,
    ).astype(np.float32)


def load_day_data(symbol: str, day: str, config: ExperimentConfig) -> DayData:
    root = Path(config.data_dir) / symbol / day
    if not root.exists():
        raise FileNotFoundError(f"Missing processed day directory: {root}")
    head_only = config.mode == "smoke"
    total_rows = (config.max_rows_per_day or 0) if head_only else _row_count(root / "ask.csv")
    ask = _read_frame(root / "ask.csv", total_rows, config.max_rows_per_day, head_only=head_only)
    bid = _read_frame(root / "bid.csv", total_rows, config.max_rows_per_day, head_only=head_only)
    price = _read_frame(root / "price.csv", total_rows, config.max_rows_per_day, head_only=head_only)
    msg = _read_frame(root / "msg.csv", total_rows, config.max_rows_per_day, head_only=head_only)

    timestamps = pd.to_datetime(ask["timestamp"])
    bid_timestamps = pd.to_datetime(bid["timestamp"])
    msg_timestamps = pd.to_datetime(msg["timestamp"])
    price_timestamps = pd.to_datetime(price["timestamp"])

    if len(bid) != len(ask) or len(msg) != len(ask) or len(price) != len(ask):
        raise ValueError(
            f"Row count mismatch for {symbol} {day}: "
            f"ask={len(ask)} bid={len(bid)} msg={len(msg)} price={len(price)}"
        )

    if not (bid_timestamps.equals(timestamps) and msg_timestamps.equals(timestamps) and price_timestamps.equals(timestamps)):
        raise ValueError(f"Timestamp alignment mismatch across processed files for {symbol} {day}")

    bid = bid.drop(columns=["timestamp"])
    msg = msg.drop(columns=["timestamp"])
    price = price.drop(columns=["timestamp"])

    trades = pd.read_csv(root / "trades.csv")
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])

    lob_parts = []
    for level in range(1, 11):
        lob_parts.extend(
            [
                ask[f"ask{level}_price"].to_numpy(dtype=np.float32)[:, None],
                ask[f"ask{level}_volume"].to_numpy(dtype=np.float32)[:, None],
                bid[f"bid{level}_price"].to_numpy(dtype=np.float32)[:, None],
                bid[f"bid{level}_volume"].to_numpy(dtype=np.float32)[:, None],
            ]
        )
    lob = np.concatenate(lob_parts, axis=1)
    midprice = price["midprice"].to_numpy(dtype=np.float32)
    ask1 = price["ask1_price"].to_numpy(dtype=np.float32)
    bid1 = price["bid1_price"].to_numpy(dtype=np.float32)
    spread = ask1 - bid1
    dynamic, market_only = _rolling_market_features(pd.DatetimeIndex(timestamps), midprice, msg)
    handcrafted = _handcrafted_features(lob, spread, midprice, market_only)
    trades_by_index = _build_trade_index_map(pd.DatetimeIndex(timestamps), trades)
    trade_indices = np.array(sorted(trades_by_index.keys()), dtype=np.int64)

    return DayData(
        symbol=symbol,
        day=day,
        timestamps=pd.DatetimeIndex(timestamps),
        lob=lob.astype(np.float32),
        midprice=midprice,
        ask1=ask1,
        bid1=bid1,
        spread=spread.astype(np.float32),
        dynamic=dynamic,
        handcrafted=handcrafted,
        trades_by_index=trades_by_index,
        trade_indices=trade_indices,
    )


def fit_lob_normalizer(days: Iterable[DayData]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for day in days:
        rows.append(_normalize_lob(day.lob))
    data = np.concatenate(rows, axis=0)
    mean = data.mean(axis=0).astype(np.float32)
    std = (data.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def _normalize_lob(raw_lob: np.ndarray) -> np.ndarray:
    x = raw_lob.astype(np.float32).copy()
    mid = (x[:, 0] + x[:, 2]) / 2.0
    mid = np.maximum(mid, 1e-8)
    for level in range(10):
        base = level * 4
        x[:, base] = x[:, base] / mid - 1.0
        x[:, base + 2] = x[:, base + 2] / mid - 1.0
        x[:, base + 1] = np.log1p(np.maximum(x[:, base + 1], 0.0))
        x[:, base + 3] = np.log1p(np.maximum(x[:, base + 3], 0.0))
    return x


def apply_lob_normalizer(day: DayData, mean: np.ndarray, std: np.ndarray) -> DayData:
    normed = (_normalize_lob(day.lob) - mean) / std
    day.normalized_lob = normed.astype(np.float32)
    day.norm_mean = mean
    day.norm_std = std
    return day


class PretrainDataset(Dataset):
    def __init__(
        self,
        days: list[DayData],
        lookback: int,
        horizon: int,
        alpha: float,
        max_samples_per_day: int | None = None,
    ) -> None:
        self.days = days
        self.lookback = lookback
        self.samples: list[tuple[int, int]] = []
        self.labels: list[np.ndarray] = []
        for day_idx, day in enumerate(days):
            labels = _midprice_labels(day.midprice, horizon, alpha)
            valid = day.valid_label_indices(lookback, horizon)
            valid = valid[labels[valid] >= 0]
            if max_samples_per_day is not None and len(valid) > max_samples_per_day:
                stride = max(1, len(valid) // max_samples_per_day)
                valid = valid[::stride][:max_samples_per_day]
            for idx in valid:
                self.samples.append((day_idx, int(idx)))
            self.labels.append(labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        day_idx, idx = self.samples[item]
        day = self.days[day_idx]
        assert day.normalized_lob is not None
        start = idx - self.lookback + 1
        window = day.normalized_lob[start : idx + 1]
        label = self.labels[day_idx][idx]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
