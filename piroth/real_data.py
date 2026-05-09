from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DiagnosticsConfig
from .simulator import SyntheticDay


REQUIRED_DAY_FILES = ("ask.csv", "bid.csv", "price.csv", "msg.csv")


class RealMarketDataLoader:
    def __init__(self, config: DiagnosticsConfig) -> None:
        self.config = config
        self.root = Path(config.real_data_root)
        self.symbol_root = self.root / config.symbol
        if not self.symbol_root.exists() and config.symbol == "GOOG":
            self.symbol_root = self.root / "GOOGL"

    def available_days(self) -> list[str]:
        if not self.symbol_root.exists():
            raise FileNotFoundError(f"Real data directory does not exist: {self.symbol_root}")
        days = []
        for day_dir in sorted(self.symbol_root.iterdir()):
            if not day_dir.is_dir():
                continue
            if all((day_dir / name).exists() for name in REQUIRED_DAY_FILES):
                days.append(day_dir.name)
        if not days:
            raise FileNotFoundError(f"No real data days found under {self.symbol_root}")
        return days

    def train_day_names(self) -> list[str]:
        return self.available_days()[: self.config.train_days]

    def test_day_names(self) -> list[str]:
        days = self.available_days()
        start = self.config.train_days
        return days[start : start + self.config.test_days]

    def train_days(self, *, lightweight: bool = False, skip_msg: bool = False) -> list[SyntheticDay]:
        return [self.load_day(day, lightweight=lightweight, skip_msg=skip_msg) for day in self.train_day_names()]

    def test_days(self, *, lightweight: bool = False, skip_msg: bool = False) -> list[SyntheticDay]:
        return [self.load_day(day, lightweight=lightweight, skip_msg=skip_msg) for day in self.test_day_names()]

    def load_day(self, day: str, *, lightweight: bool = False, skip_msg: bool = False) -> SyntheticDay:
        day_root = self.symbol_root / day
        max_rows = self.config.events_per_day_override
        windows = [(self.config.real_start_time, self.config.real_end_time)]
        price = _read_event_frame(day_root / "price.csv", self.config, max_rows=max_rows, windows=windows)
        if price.empty:
            raise ValueError(f"No price rows loaded for {day_root}; check real_start_time/real_end_time")
        timestamps = pd.Index(price["timestamp"])
        ask = _read_event_frame(day_root / "ask.csv", self.config, max_rows=max_rows, windows=windows)
        bid = _read_event_frame(day_root / "bid.csv", self.config, max_rows=max_rows, windows=windows)
        ask, bid = [_align_to_timestamps(frame, timestamps, label, day_root) for frame, label in ((ask, "ask"), (bid, "bid"))]
        if lightweight or skip_msg:
            msg = pd.DataFrame({"timestamp": price["timestamp"]})
        else:
            msg = _read_event_frame(day_root / "msg.csv", self.config, max_rows=max_rows, windows=windows)
            msg = _align_to_timestamps(msg, timestamps, "msg", day_root)
        price = _normalize_price(price, ask, bid, self.config.symbol_spec.tick_size)
        trades = _empty_trades() if lightweight else _read_trades(day_root / "trades.csv", self.config, price["timestamp"].min(), price["timestamp"].max())
        event_log = pd.DataFrame(columns=["timestamp", "event_type", "agent_type", "agent_id", "side", "price", "size", "fair_value", "maker_order_id"])
        latent = pd.DataFrame(
            {
                "timestamp": price["timestamp"],
                "fair_value": price["midprice"],
                "fair_value_tick": price["midprice"] / self.config.symbol_spec.tick_size,
                "regime_drift": 0.0,
                "metaorder_direction": 0,
                "metaorder_strength": 0.0,
                "regime_shift": 0,
                "event_kind": "real",
                "top_imbalance": _top_imbalance(ask, bid),
                "queue_pressure": 0.0,
            }
        )
        return SyntheticDay(
            symbol=self.config.symbol,
            day=day,
            ask=ask.reset_index(drop=True),
            bid=bid.reset_index(drop=True),
            price=price.reset_index(drop=True),
            trades=trades.reset_index(drop=True),
            msg=msg.reset_index(drop=True),
            event_log=event_log,
            latent=latent.reset_index(drop=True),
            depth_cube=(
                _depth_cube_from_lob(ask, bid, price, radius=self.config.export_depth_radius_ticks)
                if self.config.real_build_depth_cube
                else np.zeros((0, 0), dtype=np.float32)
            ),
        )


def load_market_days(config: DiagnosticsConfig, split: str, *, lightweight: bool = False, skip_msg: bool = False) -> list[SyntheticDay]:
    if config.data_source == "synthetic":
        from .simulator import SyntheticMarketGenerator

        generator = SyntheticMarketGenerator(config)
        names = generator.train_days() if split == "train" else generator.test_days()
        return [generator.generate_day(day) for day in names]
    if config.data_source == "real":
        loader = RealMarketDataLoader(config)
        return (
            loader.train_days(lightweight=lightweight, skip_msg=skip_msg)
            if split == "train"
            else loader.test_days(lightweight=lightweight, skip_msg=skip_msg)
        )
    raise ValueError(f"Unknown data_source: {config.data_source}")


def load_market_day_names(config: DiagnosticsConfig, split: str) -> list[str]:
    if config.data_source == "synthetic":
        from .simulator import SyntheticMarketGenerator

        generator = SyntheticMarketGenerator(config)
        return generator.train_days() if split == "train" else generator.test_days()
    if config.data_source == "real":
        loader = RealMarketDataLoader(config)
        return loader.train_day_names() if split == "train" else loader.test_day_names()
    raise ValueError(f"Unknown data_source: {config.data_source}")


def _read_event_frame(
    path: Path,
    config: DiagnosticsConfig,
    *,
    max_rows: int | None,
    windows: list[tuple[str, str]],
) -> pd.DataFrame:
    chunks = []
    remaining = max_rows
    seen_filtered = 0
    stride = max(int(config.real_event_stride), 1)
    for chunk in pd.read_csv(path, chunksize=config.real_chunk_size, parse_dates=["timestamp"]):
        if not chunk.empty and chunk["timestamp"].iloc[0] > _window_end_timestamp(chunk["timestamp"].iloc[0], windows):
            break
        chunk = _filter_windows(chunk, windows)
        if chunk.empty:
            continue
        filtered_len = len(chunk)
        if stride > 1:
            positions = np.arange(seen_filtered, seen_filtered + len(chunk))
            chunk = chunk.loc[(positions % stride) == 0]
        seen_filtered += filtered_len
        if chunk.empty:
            continue
        if remaining is not None:
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        chunks.append(chunk)
        if remaining is not None and remaining <= 0:
            break
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _read_trades(path: Path, config: DiagnosticsConfig, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not path.exists():
        return _empty_trades()
    chunks = []
    for chunk in pd.read_csv(path, chunksize=config.real_chunk_size, parse_dates=["timestamp"]):
        if not chunk.empty and chunk["timestamp"].iloc[0] > end:
            break
        chunk = chunk[(chunk["timestamp"] >= start) & (chunk["timestamp"] <= end)]
        if not chunk.empty:
            chunks.append(chunk)
    trades = pd.concat(chunks, ignore_index=True) if chunks else _empty_trades()
    if "signed_size" not in trades.columns and not trades.empty:
        trades["signed_size"] = np.where(trades["aggressor_side"] == "A", -trades["size"], trades["size"])
    for column, value in {
        "taker_agent": "real_taker",
        "maker_agent_id": "real_maker",
        "maker_agent": "real_maker",
        "maker_order_id": -1,
        "queue_ahead": 0,
    }.items():
        if column not in trades.columns:
            trades[column] = value
    return trades


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "price",
            "size",
            "signed_size",
            "aggressor_side",
            "taker_agent",
            "maker_agent_id",
            "maker_agent",
            "maker_order_id",
            "queue_ahead",
        ]
    )


def _filter_windows(frame: pd.DataFrame, windows: list[tuple[str, str]]) -> pd.DataFrame:
    clock = frame["timestamp"].dt.strftime("%H:%M:%S")
    mask = np.zeros(len(frame), dtype=bool)
    for start, end in windows:
        mask |= (clock >= start) & (clock <= end)
    return frame.loc[mask]


def _window_end_timestamp(reference: pd.Timestamp, windows: list[tuple[str, str]]) -> pd.Timestamp:
    latest = max(end for _, end in windows)
    return reference.normalize() + pd.to_timedelta(latest)


def _align_to_timestamps(frame: pd.DataFrame, timestamps: pd.Index, label: str, day_root: Path) -> pd.DataFrame:
    if len(frame) < len(timestamps):
        raise ValueError(f"{label}.csv under {day_root} has fewer filtered rows than price.csv")
    frame = frame.iloc[: len(timestamps)].copy()
    if not pd.Index(frame["timestamp"]).equals(timestamps):
        frame = frame.set_index("timestamp").reindex(timestamps).reset_index(names="timestamp")
        if frame.isna().any().any():
            raise ValueError(f"{label}.csv under {day_root} does not align with filtered price timestamps")
    return frame


def _normalize_price(price: pd.DataFrame, ask: pd.DataFrame, bid: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    price = price.copy()
    if "ask1_price" not in price:
        price["ask1_price"] = ask["ask1_price"].to_numpy()
    if "bid1_price" not in price:
        price["bid1_price"] = bid["bid1_price"].to_numpy()
    if "best_ask" not in price:
        price["best_ask"] = price["ask1_price"]
    if "best_bid" not in price:
        price["best_bid"] = price["bid1_price"]
    if "midprice" not in price:
        price["midprice"] = 0.5 * (price["ask1_price"] + price["bid1_price"])
    price["spread"] = price["ask1_price"] - price["bid1_price"]
    price["spread_ticks"] = np.maximum(1, np.rint(price["spread"] / tick_size).astype(int))
    bid_v = bid["bid1_volume"].to_numpy(dtype=np.float64)
    ask_v = ask["ask1_volume"].to_numpy(dtype=np.float64)
    total = np.clip(bid_v + ask_v, 1.0, None)
    price["microprice"] = (price["ask1_price"].to_numpy() * bid_v + price["bid1_price"].to_numpy() * ask_v) / total
    price["return_bp"] = 10_000.0 * price["midprice"].pct_change().fillna(0.0)
    return price


def _top_imbalance(ask: pd.DataFrame, bid: pd.DataFrame) -> np.ndarray:
    bid_v = bid["bid1_volume"].to_numpy(dtype=np.float64)
    ask_v = ask["ask1_volume"].to_numpy(dtype=np.float64)
    return (bid_v - ask_v) / np.clip(bid_v + ask_v, 1.0, None)


def _depth_cube_from_lob(ask: pd.DataFrame, bid: pd.DataFrame, price: pd.DataFrame, radius: int) -> np.ndarray:
    tick = float(np.median(price["spread"] / np.maximum(price["spread_ticks"], 1))) if "spread_ticks" in price else 0.01
    tick = tick if tick > 0 else 0.01
    cube = np.zeros((len(price), radius * 2 + 1), dtype=np.float32)
    for idx in range(len(price)):
        center = round(float(price.iloc[idx]["midprice"]) / tick)
        for level in range(1, 11):
            bid_rel = round(float(bid.iloc[idx][f"bid{level}_price"]) / tick) - center
            ask_rel = round(float(ask.iloc[idx][f"ask{level}_price"]) / tick) - center
            if -radius <= bid_rel <= radius:
                cube[idx, bid_rel + radius] = float(bid.iloc[idx][f"bid{level}_volume"])
            if -radius <= ask_rel <= radius:
                cube[idx, ask_rel + radius] = -float(ask.iloc[idx][f"ask{level}_volume"])
    return cube
