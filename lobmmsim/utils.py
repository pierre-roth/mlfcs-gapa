from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def price_legal_check(ask_price: float, bid_price: float, tick_size: float) -> tuple[float, float]:
    ask = math.ceil(ask_price / tick_size) * tick_size
    bid = math.floor(bid_price / tick_size) * tick_size
    if ask <= bid:
        ask = bid + tick_size
    return round(ask, 6), round(bid, 6)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if is_dataclass(payload):
        payload = asdict(payload)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def rolling_left_bounds(timestamps_ns: np.ndarray, window_seconds: int) -> np.ndarray:
    return np.searchsorted(timestamps_ns, timestamps_ns - window_seconds * 1_000_000_000, side="left")


def cumulative_window_sums(values: np.ndarray, left_bounds: np.ndarray) -> np.ndarray:
    cumsum = np.vstack([np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0, dtype=np.float64)])
    right = np.arange(len(values)) + 1
    return cumsum[right] - cumsum[left_bounds]

