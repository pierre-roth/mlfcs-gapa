from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .features import build_market_features


@dataclass
class TradeSlice:
    price: np.ndarray
    size: np.ndarray
    aggressor_side: np.ndarray
    taker_agent: np.ndarray | None = None
    maker_agent: np.ndarray | None = None
    maker_order_id: np.ndarray | None = None
    queue_ahead: np.ndarray | None = None


@dataclass
class DayData:
    symbol: str
    day: str
    timestamps: pd.DatetimeIndex
    ask: pd.DataFrame
    bid: pd.DataFrame
    price: pd.DataFrame
    msg: pd.DataFrame
    trades: pd.DataFrame
    latent: pd.DataFrame
    lob: np.ndarray
    normalized_lob: np.ndarray
    dynamic: np.ndarray
    agent_template: np.ndarray
    labels: np.ndarray
    stable_mask: np.ndarray
    midprice: np.ndarray
    ask1: np.ndarray
    bid1: np.ndarray
    spread: np.ndarray
    trades_by_index: dict[int, TradeSlice]

    def valid_label_indices(self, lookback: int, horizon: int) -> np.ndarray:
        start = max(lookback - 1, horizon)
        stop = len(self.labels) - horizon
        valid = np.arange(start, stop, dtype=np.int64)
        valid = valid[~np.isnan(self.labels[valid])]
        if self.stable_mask.size:
            valid = valid[self.stable_mask[valid]]
        return valid


def _paper_labels(midprice: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    price_series = pd.Series(midprice)
    price_past = price_series.rolling(window=horizon).mean().to_numpy()
    price_future = price_past.copy()
    price_future[:-horizon] = price_past[horizon:]
    price_future[-horizon:] = np.nan
    pct = (price_future - price_past) / np.clip(price_past, 1e-8, None)
    labels = np.full(len(midprice), np.nan, dtype=np.float32)
    labels[pct >= threshold] = 0
    labels[(pct < threshold) & (pct > -threshold)] = 1
    labels[pct <= -threshold] = 2
    return labels


def _compute_volume_normalizers(
    symbol: str,
    days: list[str],
    config: ExperimentConfig,
) -> tuple[dict[int, float], dict[int, float]]:
    """Compute corpus-wide per-level max volumes across the given days.

    Using training-day maxes (not per-day) ensures the same absolute volume
    normalizes to the same value on every day, so the backbone learns actual
    depth patterns instead of per-day scale artifacts.
    """
    ask_maxes = {level: 1.0 for level in range(1, 11)}
    bid_maxes = {level: 1.0 for level in range(1, 11)}
    for day in days:
        root = Path(config.data_dir) / symbol / day
        ask_cols = [f"ask{level}_volume" for level in range(1, 11)]
        bid_cols = [f"bid{level}_volume" for level in range(1, 11)]
        ask = pd.read_csv(root / "ask.csv", usecols=ask_cols)
        bid = pd.read_csv(root / "bid.csv", usecols=bid_cols)
        for level in range(1, 11):
            ask_maxes[level] = max(ask_maxes[level], float(ask[f"ask{level}_volume"].max()))
            bid_maxes[level] = max(bid_maxes[level], float(bid[f"bid{level}_volume"].max()))
    return ask_maxes, bid_maxes


def _paper_normalize_lob(
    ask: pd.DataFrame,
    bid: pd.DataFrame,
    midprice: np.ndarray,
    volume_normalizers: tuple[dict[int, float], dict[int, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_rows = []
    norm_rows = []
    if volume_normalizers is not None:
        ask_vol_max, bid_vol_max = volume_normalizers
    else:
        ask_vol_max = {level: max(float(ask[f"ask{level}_volume"].max()), 1.0) for level in range(1, 11)}
        bid_vol_max = {level: max(float(bid[f"bid{level}_volume"].max()), 1.0) for level in range(1, 11)}
    for idx, mid in enumerate(midprice):
        raw = []
        norm = []
        for level in range(1, 11):
            ask_p = float(ask.iloc[idx][f"ask{level}_price"])
            ask_v = float(ask.iloc[idx][f"ask{level}_volume"])
            bid_p = float(bid.iloc[idx][f"bid{level}_price"])
            bid_v = float(bid.iloc[idx][f"bid{level}_volume"])
            raw.extend([ask_p, ask_v, bid_p, bid_v])
            norm.extend([
                ask_p / max(mid, 1e-8) - 1.0,
                ask_v / ask_vol_max[level],
                bid_p / max(mid, 1e-8) - 1.0,
                bid_v / bid_vol_max[level],
            ])
        raw_rows.append(raw)
        norm_rows.append(norm)
    return np.asarray(raw_rows, dtype=np.float32), np.asarray(norm_rows, dtype=np.float32)


def _stable_mask(timestamps: pd.DatetimeIndex, windows: list[str]) -> np.ndarray:
    if not windows:
        return np.ones(len(timestamps), dtype=bool)
    mask = np.zeros(len(timestamps), dtype=bool)
    clock = timestamps.strftime("%H:%M:%S")
    for raw in windows:
        start, end = raw.split("-", maxsplit=1)
        mask |= (clock >= start) & (clock <= end)
    return mask


def _agent_template(length: int) -> np.ndarray:
    return np.zeros((length, 24), dtype=np.float32)


def _trades_by_event(trades: pd.DataFrame, timestamps: pd.DatetimeIndex) -> dict[int, TradeSlice]:
    if trades.empty:
        return {}
    grouped = trades.groupby("timestamp")
    mapping: dict[int, TradeSlice] = {}
    for idx, ts in enumerate(timestamps):
        if ts not in grouped.groups:
            continue
        block = grouped.get_group(ts)
        mapping[idx] = TradeSlice(
            price=block["price"].to_numpy(dtype=np.float32),
            size=block["size"].to_numpy(dtype=np.float32),
            aggressor_side=block["aggressor_side"].astype(str).to_numpy(),
            taker_agent=block["taker_agent"].astype(str).to_numpy() if "taker_agent" in block else None,
            maker_agent=block["maker_agent"].astype(str).to_numpy() if "maker_agent" in block else None,
            maker_order_id=block["maker_order_id"].to_numpy(dtype=np.int64) if "maker_order_id" in block else None,
            queue_ahead=block["queue_ahead"].to_numpy(dtype=np.float32) if "queue_ahead" in block else None,
        )
    return mapping


def load_day(
    symbol: str,
    day: str,
    config: ExperimentConfig,
    volume_normalizers: tuple[dict[int, float], dict[int, float]] | None = None,
) -> DayData:
    root = Path(config.data_dir) / symbol / day
    ask = pd.read_csv(root / "ask.csv", parse_dates=["timestamp"])
    bid = pd.read_csv(root / "bid.csv", parse_dates=["timestamp"])
    price = pd.read_csv(root / "price.csv", parse_dates=["timestamp"])
    msg = pd.read_csv(root / "msg.csv", parse_dates=["timestamp"])
    trades = pd.read_csv(root / "trades.csv", parse_dates=["timestamp"]) if (root / "trades.csv").exists() else pd.DataFrame(columns=["timestamp", "price", "size", "aggressor_side"])
    latent = pd.read_csv(root / "latent.csv", parse_dates=["timestamp"])
    timestamps = pd.DatetimeIndex(price["timestamp"])
    midprice = price["midprice"].to_numpy(dtype=np.float32)
    ask1 = price["ask1_price"].to_numpy(dtype=np.float32)
    bid1 = price["bid1_price"].to_numpy(dtype=np.float32)
    spread = ask1 - bid1
    lob, normalized_lob = _paper_normalize_lob(ask, bid, midprice, volume_normalizers)
    dynamic = build_market_features(timestamps, midprice.astype(np.float64), msg, config.rv_windows_s, config.rsi_windows_s, config.osi_windows_s)
    labels = _paper_labels(midprice.astype(np.float64), config.pretrain_horizon, config.pretrain_alpha)
    stable_mask = _stable_mask(timestamps, config.stable_windows if config.use_stable_hours else [])
    return DayData(
        symbol=symbol,
        day=day,
        timestamps=timestamps,
        ask=ask,
        bid=bid,
        price=price,
        msg=msg,
        trades=trades,
        latent=latent,
        lob=lob,
        normalized_lob=normalized_lob,
        dynamic=dynamic,
        agent_template=_agent_template(len(price)),
        labels=labels,
        stable_mask=stable_mask,
        midprice=midprice,
        ask1=ask1,
        bid1=bid1,
        spread=spread,
        trades_by_index=_trades_by_event(trades, timestamps),
    )


def load_splits(config: ExperimentConfig, symbol: str) -> dict[str, list[DayData]]:
    config.apply_mode_defaults()
    all_day_strs = [
        (pd.Timestamp("2019-11-01") + pd.tseries.offsets.BDay(i)).strftime("%Y%m%d")
        for i in range(config.num_days)
    ]
    train_end = config.train_days
    val_end = train_end + config.val_days
    # Compute volume normalizers from training days only, then apply to all splits.
    # This ensures the same absolute volume normalizes to the same value on every
    # day, fixing the per-day-scale overfitting that tanks test_f1.
    volume_normalizers = _compute_volume_normalizers(symbol, all_day_strs[:train_end], config)
    all_days = [load_day(symbol, d, config, volume_normalizers) for d in all_day_strs]
    # Z-norm price columns using training-day statistics (Tsantekidis Eq. 3-4 + paper Sec. III-B2).
    # Prices are already stationarized (p/mid - 1); now standardize across the corpus.
    # Volume columns stay as max-normed values in [0, 1] — no z-norm.
    price_cols = [level * 4 + offset for level in range(10) for offset in (0, 2)]
    train_price_rows = [day.normalized_lob[:, price_cols] for day in all_days[:train_end]]
    all_train_prices = np.concatenate(train_price_rows, axis=0)
    price_mean = all_train_prices.mean(axis=0).astype(np.float32)
    price_std = (all_train_prices.std(axis=0) + 1e-6).astype(np.float32)
    for day in all_days:
        day.normalized_lob[:, price_cols] = (
            (day.normalized_lob[:, price_cols] - price_mean) / price_std
        )
    return {
        "train": all_days[:train_end],
        "val": all_days[train_end:val_end],
        "test": all_days[val_end : val_end + config.test_days],
    }