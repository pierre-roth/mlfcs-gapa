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
class TradeBatch:
    price: np.ndarray
    size: np.ndarray
    aggressor_side: np.ndarray | None


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
    agent_template: np.ndarray
    trades_by_index: dict[int, TradeBatch]
    latent: pd.DataFrame
    normalized_lob: np.ndarray | None = None

    def valid_label_indices(self, lookback: int, horizon: int) -> np.ndarray:
        return np.arange(lookback - 1, len(self.midprice) - horizon)


@dataclass
class LobNormalizer:
    price_mean: np.ndarray
    price_std: np.ndarray
    volume_max: np.ndarray


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


def _stable_window_mask(timestamps: pd.DatetimeIndex, config: ExperimentConfig) -> np.ndarray:
    if not config.use_stable_hours or not config.stable_windows:
        return np.ones(len(timestamps), dtype=bool)
    seconds = timestamps.hour * 3600 + timestamps.minute * 60 + timestamps.second
    mask = np.zeros(len(timestamps), dtype=bool)
    for window in config.stable_windows:
        start, end = window.split("-", maxsplit=1)
        start_s = int(pd.to_timedelta(start).total_seconds())
        end_s = int(pd.to_timedelta(end).total_seconds())
        mask |= (seconds >= start_s) & (seconds < end_s)
    return mask


def _build_trade_index_map(timestamps: pd.DatetimeIndex, trades: pd.DataFrame) -> dict[int, TradeBatch]:
    if trades.empty:
        return {}
    ts_to_index = {ts: idx for idx, ts in enumerate(timestamps)}
    trades = trades[trades["timestamp"].isin(ts_to_index)]
    out: dict[int, TradeBatch] = {}
    for ts, group in trades.groupby("timestamp"):
        out[ts_to_index[ts]] = TradeBatch(
            price=group["price"].to_numpy(dtype=np.float32, copy=True),
            size=group["size"].to_numpy(dtype=np.float32, copy=True),
            aggressor_side=group["aggressor_side"].to_numpy(copy=True),
        )
    return out


def _rolling_market_features(timestamps: pd.DatetimeIndex, midprice: np.ndarray, msg: pd.DataFrame, config: ExperimentConfig) -> np.ndarray:
    ts_ns = timestamps.asi8
    log_mid = np.log(np.maximum(midprice, 1e-8))
    returns = np.diff(log_mid, prepend=log_mid[0])
    sq_returns = returns**2
    gains = np.maximum(np.diff(midprice, prepend=midprice[0]), 0.0)
    losses = np.maximum(-np.diff(midprice, prepend=midprice[0]), 0.0)

    features: list[np.ndarray] = []
    for window in config.rv_windows_s:
        left = rolling_left_bounds(ts_ns, window)
        csum = np.concatenate([[0.0], np.cumsum(sq_returns, dtype=np.float64)])
        right = np.arange(len(midprice)) + 1
        values = csum[right] - csum[left]
        features.append(np.sqrt(np.maximum(values, 0.0)).astype(np.float32))
    for window in config.rsi_windows_s:
        left = rolling_left_bounds(ts_ns, window)
        gain_sum = np.concatenate([[0.0], np.cumsum(gains, dtype=np.float64)])
        loss_sum = np.concatenate([[0.0], np.cumsum(losses, dtype=np.float64)])
        right = np.arange(len(midprice)) + 1
        gains_w = gain_sum[right] - gain_sum[left]
        losses_w = loss_sum[right] - loss_sum[left]
        features.append((gains_w / np.maximum(gains_w + losses_w, 1e-8)).astype(np.float32))

    msg_values = msg[MSG_COLUMNS].to_numpy(dtype=np.float64)
    pair_indices = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11)]
    for window in config.osi_windows_s:
        left = rolling_left_bounds(ts_ns, window)
        sums = cumulative_window_sums(msg_values, left)
        for buy_idx, sell_idx in pair_indices:
            buy = sums[:, buy_idx]
            sell = sums[:, sell_idx]
            features.append(((buy - sell) / np.maximum(buy + sell, 1e-8)).astype(np.float32))
    return np.stack(features, axis=1).astype(np.float32)


def _agent_state_template(length: int) -> np.ndarray:
    zeros = np.zeros((length, 12), dtype=np.float32)
    time_ratio = (np.arange(length, dtype=np.float32) / max(length, 1)).reshape(-1, 1)
    return np.concatenate([zeros, np.repeat(time_ratio, 12, axis=1)], axis=1).astype(np.float32)


def _stationary_lob(raw_lob: np.ndarray) -> np.ndarray:
    x = raw_lob.astype(np.float32).copy()
    mid = (x[:, 0] + x[:, 2]) / 2.0
    mid = np.maximum(mid, 1e-8)
    for level in range(10):
        base = level * 4
        x[:, base] = x[:, base] - mid
        x[:, base + 2] = x[:, base + 2] - mid
    return x


def fit_lob_normalizer(days: Iterable[DayData]) -> LobNormalizer:
    stacked = np.concatenate([_stationary_lob(day.lob) for day in days], axis=0)
    price_cols = np.asarray([idx for idx in range(stacked.shape[1]) if idx % 4 in {0, 2}], dtype=np.int64)
    volume_cols = np.asarray([idx for idx in range(stacked.shape[1]) if idx % 4 in {1, 3}], dtype=np.int64)
    price_mean = stacked[:, price_cols].mean(axis=0).astype(np.float32)
    price_std = (stacked[:, price_cols].std(axis=0) + 1e-6).astype(np.float32)
    volume_max = np.maximum(stacked[:, volume_cols].max(axis=0), 1.0).astype(np.float32)
    return LobNormalizer(price_mean=price_mean, price_std=price_std, volume_max=volume_max)


def apply_lob_normalizer(day: DayData, normalizer: LobNormalizer) -> DayData:
    x = _stationary_lob(day.lob)
    price_cols = [idx for idx in range(x.shape[1]) if idx % 4 in {0, 2}]
    volume_cols = [idx for idx in range(x.shape[1]) if idx % 4 in {1, 3}]
    x[:, price_cols] = (x[:, price_cols] - normalizer.price_mean) / normalizer.price_std
    x[:, volume_cols] = x[:, volume_cols] / normalizer.volume_max
    day.normalized_lob = x.astype(np.float32)
    return day


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


def load_day_data(symbol: str, day: str, config: ExperimentConfig) -> DayData:
    root = Path(config.data_dir) / symbol / day
    if not root.exists():
        raise FileNotFoundError(f"Missing processed day directory: {root}")
    ask = pd.read_csv(root / "ask.csv")
    bid = pd.read_csv(root / "bid.csv")
    price = pd.read_csv(root / "price.csv")
    msg = pd.read_csv(root / "msg.csv")
    trades = pd.read_csv(root / "trades.csv")
    latent = pd.read_csv(root / "latent.csv")
    if config.max_rows_per_day is not None:
        ask = ask.iloc[: config.max_rows_per_day].copy()
        bid = bid.iloc[: config.max_rows_per_day].copy()
        price = price.iloc[: config.max_rows_per_day].copy()
        msg = msg.iloc[: config.max_rows_per_day].copy()
        valid_ts = set(pd.to_datetime(ask["timestamp"]))
        trades = trades[pd.to_datetime(trades["timestamp"]).isin(valid_ts)].copy()
        latent = latent.iloc[: config.max_rows_per_day].copy()
    timestamps = pd.to_datetime(ask["timestamp"])
    stable_mask = _stable_window_mask(pd.DatetimeIndex(timestamps), config)
    ask = ask.loc[stable_mask].reset_index(drop=True)
    bid = bid.loc[stable_mask].reset_index(drop=True)
    price = price.loc[stable_mask].reset_index(drop=True)
    msg = msg.loc[stable_mask].reset_index(drop=True)
    latent = latent.loc[stable_mask].reset_index(drop=True)
    timestamps = pd.to_datetime(ask["timestamp"])
    valid_ts = set(timestamps)
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades = trades[trades["timestamp"].isin(valid_ts)].copy()

    ask_values = []
    for level in range(1, 11):
        ask_values.extend(
            [
                ask[f"ask{level}_price"].to_numpy(dtype=np.float32)[:, None],
                ask[f"ask{level}_volume"].to_numpy(dtype=np.float32)[:, None],
                bid[f"bid{level}_price"].to_numpy(dtype=np.float32)[:, None],
                bid[f"bid{level}_volume"].to_numpy(dtype=np.float32)[:, None],
            ]
        )
    lob = np.concatenate(ask_values, axis=1)
    midprice = price["midprice"].to_numpy(dtype=np.float32)
    ask1 = price["ask1_price"].to_numpy(dtype=np.float32)
    bid1 = price["bid1_price"].to_numpy(dtype=np.float32)
    spread = ask1 - bid1
    dynamic = _rolling_market_features(pd.DatetimeIndex(timestamps), midprice, msg, config)
    trades_by_index = _build_trade_index_map(pd.DatetimeIndex(timestamps), trades)
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
        agent_template=_agent_state_template(len(timestamps)),
        trades_by_index=trades_by_index,
        latent=latent,
    )


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
        self.regime_labels: list[np.ndarray] = []
        self.sample_labels: list[int] = []
        for day_idx, day in enumerate(days):
            labels = _midprice_labels(day.midprice, horizon, alpha)
            regimes = (day.latent["regime"].to_numpy(dtype=np.int64, copy=True) + 1).clip(0, 2)
            valid = day.valid_label_indices(lookback, horizon)
            valid = valid[labels[valid] >= 0]
            if max_samples_per_day is not None and len(valid) > max_samples_per_day:
                stride = max(1, len(valid) // max_samples_per_day)
                valid = valid[::stride][:max_samples_per_day]
            for idx in valid:
                self.samples.append((day_idx, int(idx)))
                self.sample_labels.append(int(labels[idx]))
            self.labels.append(labels)
            self.regime_labels.append(regimes)
        self.sample_labels_np = np.asarray(self.sample_labels, dtype=np.int64)

    def class_counts(self) -> dict[int, int]:
        if self.sample_labels_np.size == 0:
            return {0: 0, 1: 0, 2: 0}
        counts = np.bincount(self.sample_labels_np, minlength=3)
        return {idx: int(counts[idx]) for idx in range(3)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        day_idx, idx = self.samples[item]
        day = self.days[day_idx]
        assert day.normalized_lob is not None
        start = idx - self.lookback + 1
        window = day.normalized_lob[start : idx + 1]
        label = np.asarray(
            [
                self.labels[day_idx][idx],
                self.regime_labels[day_idx][idx],
            ],
            dtype=np.int64,
        )
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
