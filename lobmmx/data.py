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
    trades_by_index: dict[int, "TradeBatch"]
    trade_indices: np.ndarray
    signed_trade_volume: np.ndarray
    msg: np.ndarray  # per-event message flow: columns follow MSG_COLUMNS order
    row_multiplier: int = 1
    normalized_lob: np.ndarray | None = None
    norm_mean: np.ndarray | None = None
    norm_std: np.ndarray | None = None

    def valid_label_indices(self, lookback: int, horizon: int) -> np.ndarray:
        return np.arange(lookback - 1, len(self.midprice) - horizon)


@dataclass
class TradeBatch:
    price: np.ndarray
    size: np.ndarray
    aggressor_side: np.ndarray | None


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


def _row_stride_for_limit(total_rows: int, max_rows: int | None) -> int:
    if max_rows is None or total_rows <= max_rows:
        return 1
    return max(1, int(np.ceil(total_rows / max_rows)))


def _skiprows_for_limit(total_rows: int, max_rows: int | None):
    stride = _row_stride_for_limit(total_rows, max_rows)
    if stride <= 1:
        return None
    return lambda idx: idx > 0 and ((idx - 1) % stride != 0)


def _read_frame(path: Path, total_rows: int, max_rows: int | None, head_only: bool = False) -> pd.DataFrame:
    if head_only and max_rows is not None:
        return pd.read_csv(path, nrows=max_rows)
    skiprows = _skiprows_for_limit(total_rows, max_rows)
    return pd.read_csv(path, skiprows=skiprows)


def _read_contiguous_frame(path: Path, start_row: int, nrows: int) -> pd.DataFrame:
    if start_row <= 0:
        return pd.read_csv(path, nrows=nrows)
    return pd.read_csv(path, skiprows=range(1, start_row + 1), nrows=nrows)


def _build_trade_index_map(timestamps: pd.DatetimeIndex, trades: pd.DataFrame) -> dict[int, TradeBatch]:
    if trades.empty:
        return {}
    ts_to_index = {ts: idx for idx, ts in enumerate(timestamps)}
    trades = trades[trades["timestamp"].isin(ts_to_index)]
    grouped: dict[int, TradeBatch] = {}
    for ts, group in trades.groupby("timestamp"):
        grouped[ts_to_index[ts]] = TradeBatch(
            price=group["price"].to_numpy(dtype=np.float32, copy=True),
            size=group["size"].to_numpy(dtype=np.float32, copy=True),
            aggressor_side=group["aggressor_side"].to_numpy(copy=True) if "aggressor_side" in group.columns else None,
        )
    return grouped


def _time_to_seconds(value: str) -> int:
    delta = pd.to_timedelta(value)
    return int(delta.total_seconds())


def _stable_window_mask(timestamps: pd.DatetimeIndex, config: ExperimentConfig) -> np.ndarray:
    if not config.use_stable_hours or not config.stable_windows:
        return np.ones(len(timestamps), dtype=bool)
    seconds = timestamps.hour * 3600 + timestamps.minute * 60 + timestamps.second
    mask = np.zeros(len(timestamps), dtype=bool)
    for window in config.stable_windows:
        start, end = window.split("-", maxsplit=1)
        start_s = _time_to_seconds(start)
        end_s = _time_to_seconds(end)
        mask |= (seconds >= start_s) & (seconds < end_s)
    return mask


def _stable_window_mask_from_strings(times: pd.Series, config: ExperimentConfig) -> np.ndarray:
    if not config.use_stable_hours or not config.stable_windows:
        return np.ones(len(times), dtype=bool)
    mask = np.zeros(len(times), dtype=bool)
    for window in config.stable_windows:
        start, end = window.split("-", maxsplit=1)
        mask |= (times >= start) & (times < end)
    return mask


def _find_smoke_stable_start(path: Path, config: ExperimentConfig, chunk_size: int = 250_000) -> int | None:
    if not config.use_stable_hours or not config.stable_windows:
        return None
    row_offset = 0
    for chunk in pd.read_csv(path, usecols=["timestamp"], chunksize=chunk_size):
        times = chunk["timestamp"].astype(str).str.slice(11, 19)
        mask = _stable_window_mask_from_strings(times, config)
        if np.any(mask):
            indices = np.flatnonzero(mask)
            return row_offset + int(indices[0])
        row_offset += len(chunk)
    return None


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
    return _rolling_market_features_with_config(timestamps, midprice, msg, ExperimentConfig())


def _rolling_market_features_with_config(
    timestamps: pd.DatetimeIndex,
    midprice: np.ndarray,
    msg: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    ts_ns = timestamps.asi8
    log_mid = np.log(np.maximum(midprice, 1e-8))
    returns = np.diff(log_mid, prepend=log_mid[0])
    sq_returns = returns**2
    gains = np.maximum(np.diff(midprice, prepend=midprice[0]), 0.0)
    losses = np.maximum(np.diff(midprice, prepend=midprice[0]) * -1.0, 0.0)
    rv_windows = config.rv_windows_s
    rsi_windows = config.rsi_windows_s
    osi_windows = config.osi_windows_s

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
    def market_col(idx: int) -> np.ndarray:
        if idx < market_only.shape[1]:
            return market_only[:, idx]
        return np.zeros(len(midprice), dtype=np.float32)
    return np.stack(
        [
            spread,
            imbalance,
            mid_return,
            market_col(0),
            market_col(3),
            market_col(4),
            market_col(5),
        ],
        axis=1,
    ).astype(np.float32)


def _signed_trade_volume_series(length: int, trades_by_index: dict[int, TradeBatch]) -> np.ndarray:
    series = np.zeros(length, dtype=np.float32)
    for idx, batch in trades_by_index.items():
        if batch.aggressor_side is None:
            continue
        signed = np.where(batch.aggressor_side == "B", batch.size, np.where(batch.aggressor_side == "A", -batch.size, 0.0))
        series[idx] = float(np.sum(signed))
    return series


def _spread_labels(spread: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    labels = np.full(len(spread), -100, dtype=np.int64)
    for idx in range(horizon, len(spread) - horizon):
        past = spread[idx - horizon : idx].mean()
        future = spread[idx : idx + horizon].mean()
        diff = future - past
        if diff > threshold:
            labels[idx] = 2
        elif diff < -threshold:
            labels[idx] = 0
        else:
            labels[idx] = 1
    return labels


def _flow_labels(signed_trade_volume: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    labels = np.full(len(signed_trade_volume), -100, dtype=np.int64)
    cumsum = np.concatenate([[0.0], np.cumsum(signed_trade_volume, dtype=np.float64)])
    for idx in range(horizon, len(signed_trade_volume) - horizon):
        future = cumsum[idx + horizon] - cumsum[idx]
        if future > threshold:
            labels[idx] = 2
        elif future < -threshold:
            labels[idx] = 0
        else:
            labels[idx] = 1
    return labels


def load_day_data(symbol: str, day: str, config: ExperimentConfig) -> DayData:
    root = Path(config.data_dir) / symbol / day
    if not root.exists():
        raise FileNotFoundError(f"Missing processed day directory: {root}")
    head_only = config.mode == "smoke" and not config.use_stable_hours
    smoke_start = None
    if config.mode == "smoke" and config.use_stable_hours and config.max_rows_per_day is not None:
        smoke_start = _find_smoke_stable_start(root / "ask.csv", config)
    if smoke_start is not None:
        sample_rows = config.max_rows_per_day
        ask = _read_contiguous_frame(root / "ask.csv", smoke_start, sample_rows)
        bid = _read_contiguous_frame(root / "bid.csv", smoke_start, sample_rows)
        price = _read_contiguous_frame(root / "price.csv", smoke_start, sample_rows)
        msg = _read_contiguous_frame(root / "msg.csv", smoke_start, sample_rows)
        row_multiplier = 1
    else:
        total_rows = (config.max_rows_per_day or 0) if head_only else _row_count(root / "ask.csv")
        row_multiplier = 1 if head_only else _row_stride_for_limit(total_rows, config.max_rows_per_day)
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

    stable_mask = _stable_window_mask(pd.DatetimeIndex(timestamps), config)
    if not stable_mask.all():
        ask = ask.loc[stable_mask].reset_index(drop=True)
        bid = bid.loc[stable_mask].reset_index(drop=True)
        msg = msg.loc[stable_mask].reset_index(drop=True)
        price = price.loc[stable_mask].reset_index(drop=True)
        timestamps = pd.to_datetime(ask["timestamp"])
        if ask.empty:
            raise ValueError(f"No stable-hour rows left for {symbol} {day} after applying {config.stable_windows}")

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
    dynamic, market_only = _rolling_market_features_with_config(pd.DatetimeIndex(timestamps), midprice, msg, config)
    handcrafted = _handcrafted_features(lob, spread, midprice, market_only)
    trades_by_index = _build_trade_index_map(pd.DatetimeIndex(timestamps), trades)
    trade_indices = np.array(sorted(trades_by_index.keys()), dtype=np.int64)
    signed_trade_volume = _signed_trade_volume_series(len(timestamps), trades_by_index)
    msg_array = msg[MSG_COLUMNS].to_numpy(dtype=np.float32)

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
        signed_trade_volume=signed_trade_volume,
        msg=msg_array,
        row_multiplier=row_multiplier,
    )


def estimate_episode_length(
    days: Iterable[DayData],
    target_seconds: int | None,
    lookback: int,
    latency: int,
    fallback: int,
) -> int:
    if target_seconds is None or target_seconds <= 0:
        return fallback
    total_steps = 0
    total_seconds = 0.0
    for day in days:
        tradable = len(day.midprice) - (lookback - 1 + latency)
        if tradable <= 0 or len(day.timestamps) < 2:
            continue
        span = (day.timestamps[-1] - day.timestamps[0]).total_seconds()
        if span <= 0:
            continue
        total_steps += tradable * max(day.row_multiplier, 1)
        total_seconds += span
    if total_steps <= 0 or total_seconds <= 0:
        return fallback
    steps_per_second = total_steps / total_seconds
    return max(8, int(round(target_seconds * steps_per_second)))


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
        config: ExperimentConfig,
        max_samples_per_day: int | None = None,
    ) -> None:
        self.days = days
        self.lookback = lookback
        self.samples: list[tuple[int, int]] = []
        self.mid_labels: list[np.ndarray] = []
        self.spread_labels: list[np.ndarray] = []
        self.flow_labels: list[np.ndarray] = []
        self.sample_labels: list[int] = []
        for day_idx, day in enumerate(days):
            mid_labels = _midprice_labels(day.midprice, horizon, alpha)
            spread_threshold = max(1e-8, config.tick_size * config.pretrain_spread_alpha_ticks)
            spread_labels = _spread_labels(day.spread, horizon, spread_threshold)
            flow_labels = _flow_labels(day.signed_trade_volume, horizon, config.pretrain_flow_alpha)
            valid = day.valid_label_indices(lookback, horizon)
            valid = valid[(mid_labels[valid] >= 0) & (spread_labels[valid] >= 0) & (flow_labels[valid] >= 0)]
            if max_samples_per_day is not None and len(valid) > max_samples_per_day:
                stride = max(1, len(valid) // max_samples_per_day)
                valid = valid[::stride][:max_samples_per_day]
            for idx in valid:
                self.samples.append((day_idx, int(idx)))
                self.sample_labels.append(int(mid_labels[idx]))
            self.mid_labels.append(mid_labels)
            self.spread_labels.append(spread_labels)
            self.flow_labels.append(flow_labels)
        self.sample_labels_np = np.asarray(self.sample_labels, dtype=np.int64)

    def class_counts(self) -> dict[int, int]:
        if self.sample_labels_np.size == 0:
            return {0: 0, 1: 0, 2: 0}
        counts = np.bincount(self.sample_labels_np, minlength=3)
        return {label: int(counts[label]) for label in range(3)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        day_idx, idx = self.samples[item]
        day = self.days[day_idx]
        assert day.normalized_lob is not None
        start = idx - self.lookback + 1
        window = day.normalized_lob[start : idx + 1]
        labels = np.asarray(
            [
                self.mid_labels[day_idx][idx],
                self.spread_labels[day_idx][idx],
                self.flow_labels[day_idx][idx],
            ],
            dtype=np.int64,
        )
        return torch.tensor(window, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)