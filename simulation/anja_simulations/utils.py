from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def price_legal_check(ask_price: float, bid_price: float, tick_size: float) -> tuple[float, float]:
    ask = math.ceil(ask_price / tick_size) * tick_size
    bid = math.floor(bid_price / tick_size) * tick_size
    if ask <= bid:
        ask = bid + tick_size
    return round(ask, 6), round(bid, 6)


def parse_windows(values: Iterable[str]) -> list[tuple[str, str]]:
    windows = []
    for item in values:
        start, end = item.split("-", maxsplit=1)
        windows.append((start, end))
    return windows

