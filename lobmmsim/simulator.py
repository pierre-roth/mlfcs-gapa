from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyrallis

from .config import GenerateConfig
from .utils import ensure_dir, price_legal_check, save_json, set_seed


_MSG_COLUMNS = [
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
class SymbolProfile:
    base_price: float
    events_per_day: int
    depth_scale: float
    alpha_persistence: float
    alpha_noise: float
    market_order_bias: float
    add_rate: float
    cancel_rate: float


_SYMBOL_PROFILES: dict[str, SymbolProfile] = {
    "000001": SymbolProfile(base_price=12.5, events_per_day=120_000, depth_scale=1.25, alpha_persistence=0.97, alpha_noise=0.035, market_order_bias=0.08, add_rate=1.05, cancel_rate=0.82),
    "000858": SymbolProfile(base_price=135.0, events_per_day=90_000, depth_scale=1.0, alpha_persistence=0.975, alpha_noise=0.032, market_order_bias=0.07, add_rate=1.0, cancel_rate=0.78),
    "002415": SymbolProfile(base_price=32.0, events_per_day=60_000, depth_scale=0.8, alpha_persistence=0.98, alpha_noise=0.03, market_order_bias=0.06, add_rate=0.96, cancel_rate=0.72),
}


def _day_label(idx: int) -> str:
    return (pd.Timestamp("2019-11-01") + pd.tseries.offsets.BDay(idx)).strftime("%Y%m%d")


def _parse_window(anchor_day: pd.Timestamp, window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = window.split("-", maxsplit=1)
    return anchor_day + pd.to_timedelta(start), anchor_day + pd.to_timedelta(end)


def _session_timestamps(anchor_day: pd.Timestamp, session_windows: list[str], events_per_day: int, rng: np.random.Generator) -> pd.DatetimeIndex:
    segments = [_parse_window(anchor_day, window) for window in session_windows]
    durations = np.array([(end - start).total_seconds() for start, end in segments], dtype=np.float64)
    total_seconds = durations.sum()
    counts = np.floor(events_per_day * durations / total_seconds).astype(int)
    counts[-1] += events_per_day - int(counts.sum())
    pieces: list[pd.DatetimeIndex] = []
    for (start, end), count, seconds in zip(segments, counts, durations, strict=True):
        if count <= 0:
            continue
        intervals = rng.gamma(shape=2.0, scale=1.0, size=count)
        intervals = np.cumsum(intervals)
        intervals = intervals / intervals[-1] * max(seconds - 1e-6, 1e-6)
        ns = np.round(intervals * 1_000_000_000).astype(np.int64)
        pieces.append(pd.DatetimeIndex(start.value + ns))
    if not pieces:
        return pd.DatetimeIndex([])
    return pieces[0].append(pieces[1:]).sort_values()


def _weighted_level(rng: np.random.Generator, levels: int = 10) -> int:
    weights = np.exp(-0.55 * np.arange(levels))
    weights = weights / weights.sum()
    return int(rng.choice(np.arange(levels), p=weights))


class SyntheticOrderBook:
    def __init__(
        self,
        profile: SymbolProfile,
        tick_size: float,
        trade_unit: int,
        rng: np.random.Generator,
        signal_scale: float,
        noise_scale: float,
        market_order_impact_scale: float,
        flow_reversion_scale: float,
        spread_widen_prob: float,
        spread_imbalance_threshold: float,
        spread_alpha_threshold: float,
        recenter_follow_scale: float,
    ) -> None:
        self.profile = profile
        self.tick_size = tick_size
        self.trade_unit = trade_unit
        self.rng = rng
        self.signal_scale = signal_scale
        self.noise_scale = noise_scale
        self.market_order_impact_scale = market_order_impact_scale
        self.flow_reversion_scale = flow_reversion_scale
        self.spread_widen_prob = spread_widen_prob
        self.spread_imbalance_threshold = spread_imbalance_threshold
        self.spread_alpha_threshold = spread_alpha_threshold
        self.recenter_follow_scale = recenter_follow_scale
        self.mid = round(profile.base_price / tick_size) * tick_size
        self.efficient_price = self.mid
        self.spread_ticks = 1
        self.bid_volumes = self._init_depth(profile.depth_scale)
        self.ask_volumes = self._init_depth(profile.depth_scale)
        self.alpha = 0.0
        self.signed_flow_state = 0.0
        self.regime = 0
        self._regime_clock = 0

    def _init_depth(self, scale: float) -> np.ndarray:
        base = np.linspace(14, 6, 10) * self.trade_unit * scale
        noise = self.rng.uniform(0.7, 1.3, size=10)
        return np.maximum(np.round(base * noise / self.trade_unit) * self.trade_unit, self.trade_unit).astype(np.float32)

    @property
    def bid1(self) -> float:
        return round(self.mid - self.spread_ticks * self.tick_size / 2.0, 6)

    @property
    def ask1(self) -> float:
        return round(self.mid + self.spread_ticks * self.tick_size / 2.0, 6)

    def bid_prices(self) -> np.ndarray:
        return np.asarray([self.bid1 - level * self.tick_size for level in range(10)], dtype=np.float32)

    def ask_prices(self) -> np.ndarray:
        return np.asarray([self.ask1 + level * self.tick_size for level in range(10)], dtype=np.float32)

    def _step_latent(self) -> tuple[bool, float]:
        self._regime_clock += 1
        regime_shift = False
        if self._regime_clock > 350 and self.rng.random() < 0.004:
            self.regime = int(self.rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3]))
            self._regime_clock = 0
            regime_shift = True
        target = self.signal_scale * 0.35 * self.regime
        self.alpha = (
            self.profile.alpha_persistence * self.alpha
            + (1.0 - self.profile.alpha_persistence) * target
            + self.rng.normal(0.0, 0.55 * self.profile.alpha_noise * self.signal_scale)
        )
        efficient_move = 0.0035 * self.alpha + self.rng.normal(0.0, 0.28 * self.noise_scale)
        self.efficient_price = max(self.tick_size, self.efficient_price + efficient_move)
        return regime_shift, efficient_move

    def _top_imbalance(self) -> float:
        return float((self.bid_volumes[0] - self.ask_volumes[0]) / max(self.bid_volumes[0] + self.ask_volumes[0], 1e-6))

    def _desired_spread_ticks(self) -> int:
        imbalance = abs(self._top_imbalance())
        if (
            imbalance > self.spread_imbalance_threshold
            or abs(self.alpha) > self.spread_alpha_threshold * self.signal_scale
            or self.rng.random() < self.spread_widen_prob
        ):
            return 2
        return 1

    def _choose_event(self) -> tuple[str, str]:
        imbalance = self._top_imbalance()
        alpha = self.alpha
        flow_pressure = float(np.tanh(self.signed_flow_state / 18.0))
        flow_term = self.flow_reversion_scale * flow_pressure
        # Positive latent alpha should favor profitable bid-side fills before the future upward move,
        # rather than immediate same-direction adverse selection.
        mb = np.exp(self.profile.market_order_bias - 0.14 * alpha - 0.08 * imbalance - 0.35 * flow_term)
        ms = np.exp(self.profile.market_order_bias + 0.14 * alpha + 0.08 * imbalance + 0.35 * flow_term)
        lb = np.exp(self.profile.add_rate - 0.10 * alpha + 0.25 * max(imbalance, 0.0) + 0.10 * flow_term)
        ls = np.exp(self.profile.add_rate + 0.10 * alpha + 0.25 * max(-imbalance, 0.0) - 0.10 * flow_term)
        cb = np.exp(self.profile.cancel_rate + 0.08 * alpha - 0.10 * max(imbalance, 0.0) - 0.06 * flow_term)
        cs = np.exp(self.profile.cancel_rate - 0.08 * alpha - 0.10 * max(-imbalance, 0.0) + 0.06 * flow_term)
        weights = np.asarray([mb, ms, lb, ls, cb, cs], dtype=np.float64)
        weights = weights / weights.sum()
        event_idx = int(self.rng.choice(np.arange(6), p=weights))
        mapping = [
            ("market", "buy"),
            ("market", "sell"),
            ("limit", "buy"),
            ("limit", "sell"),
            ("cancel", "buy"),
            ("cancel", "sell"),
        ]
        return mapping[event_idx]

    def _draw_size(self, side_scale: float = 1.0) -> float:
        lots = int(self.rng.choice([1, 2, 3, 4], p=[0.55, 0.25, 0.12, 0.08]))
        size = max(self.trade_unit, int(round(lots * self.trade_unit * side_scale)))
        return float(size)

    def _recenter(self) -> None:
        target_mid = round(self.efficient_price / self.tick_size) * self.tick_size
        gap_ticks = int(round((target_mid - self.mid) / self.tick_size))
        move_mid = 0
        if gap_ticks != 0:
            gap_abs = abs(gap_ticks)
            follow_prob = min(0.75, self.recenter_follow_scale * (0.06 + 0.12 * gap_abs + 0.05 * min(abs(self.alpha), 1.0)))
            if self.rng.random() < follow_prob:
                move_mid = 1 if gap_ticks > 0 else -1
        if move_mid != 0:
            if move_mid > 0:
                self.bid_volumes = np.roll(self.bid_volumes, -1)
                self.ask_volumes = np.roll(self.ask_volumes, -1)
                self.bid_volumes[-1] = self._init_depth(self.profile.depth_scale)[0]
                self.ask_volumes[-1] = self._init_depth(self.profile.depth_scale)[0]
            else:
                self.bid_volumes = np.roll(self.bid_volumes, 1)
                self.ask_volumes = np.roll(self.ask_volumes, 1)
                self.bid_volumes[0] = self._init_depth(self.profile.depth_scale)[0]
                self.ask_volumes[0] = self._init_depth(self.profile.depth_scale)[0]
            self.mid = round(self.mid + move_mid * self.tick_size, 6)
        self.spread_ticks = self._desired_spread_ticks()
        if self.bid_volumes[0] <= 0:
            self.bid_volumes[:-1] = self.bid_volumes[1:]
            self.bid_volumes[-1] = self._init_depth(self.profile.depth_scale)[0]
            self.mid -= self.tick_size
        if self.ask_volumes[0] <= 0:
            self.ask_volumes[:-1] = self.ask_volumes[1:]
            self.ask_volumes[-1] = self._init_depth(self.profile.depth_scale)[0]
            self.mid += self.tick_size
        self.mid = max(self.tick_size, self.mid)
        ask, bid = price_legal_check(self.ask1, self.bid1, self.tick_size)
        self.mid = round((ask + bid) / 2.0, 6)
        self.spread_ticks = max(1, int(round((ask - bid) / self.tick_size)))

    def step(self) -> tuple[dict[str, float | int | str], dict[str, float], dict[str, float]]:
        regime_shift, efficient_move = self._step_latent()
        event_type, side = self._choose_event()
        msg = {column: 0.0 for column in _MSG_COLUMNS}
        trade = {"price": np.nan, "size": np.nan, "aggressor_side": ""}
        queue_pressure = float((self.ask_volumes[:3].sum() - self.bid_volumes[:3].sum()) / max(self.ask_volumes[:3].sum() + self.bid_volumes[:3].sum(), 1e-6))
        size = self._draw_size(1.0 + 0.3 * abs(self.alpha))
        if event_type == "market":
            if side == "buy":
                trade["price"] = self.ask1
                trade["size"] = size
                trade["aggressor_side"] = "B"
                self.ask_volumes[0] -= size
                msg["market_buy_volume"] = size
                msg["market_buy_n"] = 1.0
                self.signed_flow_state = 0.985 * self.signed_flow_state + size / self.trade_unit
                self.efficient_price += self.market_order_impact_scale * (0.0015 * self.tick_size + 0.0008 * self.alpha)
            else:
                trade["price"] = self.bid1
                trade["size"] = size
                trade["aggressor_side"] = "A"
                self.bid_volumes[0] -= size
                msg["market_sell_volume"] = size
                msg["market_sell_n"] = 1.0
                self.signed_flow_state = 0.985 * self.signed_flow_state - size / self.trade_unit
                self.efficient_price -= self.market_order_impact_scale * (0.0015 * self.tick_size - 0.0008 * self.alpha)
        elif event_type == "limit":
            level = _weighted_level(self.rng)
            if side == "buy":
                self.bid_volumes[level] += size
                msg["limit_buy_volume"] = size
                msg["limit_buy_n"] = 1.0
            else:
                self.ask_volumes[level] += size
                msg["limit_sell_volume"] = size
                msg["limit_sell_n"] = 1.0
            self.signed_flow_state *= 0.995
        else:
            level = _weighted_level(self.rng)
            if side == "buy":
                removed = min(size, float(self.bid_volumes[level] - self.trade_unit))
                removed = max(0.0, removed)
                self.bid_volumes[level] -= removed
                msg["withdraw_buy_volume"] = removed
                msg["withdraw_buy_n"] = 1.0 if removed > 0 else 0.0
            else:
                removed = min(size, float(self.ask_volumes[level] - self.trade_unit))
                removed = max(0.0, removed)
                self.ask_volumes[level] -= removed
                msg["withdraw_sell_volume"] = removed
                msg["withdraw_sell_n"] = 1.0 if removed > 0 else 0.0
            self.signed_flow_state *= 0.99
        self.bid_volumes = np.maximum(self.bid_volumes, self.trade_unit).astype(np.float32)
        self.ask_volumes = np.maximum(self.ask_volumes, self.trade_unit).astype(np.float32)
        self._recenter()
        latent = {
            "efficient_price": float(self.efficient_price),
            "latent_alpha": float(self.alpha),
            "regime": int(self.regime),
            "signed_flow_state": float(self.signed_flow_state),
            "spread_ticks": int(self.spread_ticks),
            "top_imbalance": float(self._top_imbalance()),
            "queue_pressure": queue_pressure,
            "event_type": event_type,
            "event_side": side,
            "regime_shift": int(regime_shift),
            "efficient_move": float(efficient_move),
        }
        return latent, msg, trade

    def book_snapshot(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        ask = {"timestamp": None}
        bid = {"timestamp": None}
        for level, (ask_p, ask_v, bid_p, bid_v) in enumerate(zip(self.ask_prices(), self.ask_volumes, self.bid_prices(), self.bid_volumes, strict=True), start=1):
            ask[f"ask{level}_price"] = float(ask_p)
            ask[f"ask{level}_volume"] = float(ask_v)
            bid[f"bid{level}_price"] = float(bid_p)
            bid[f"bid{level}_volume"] = float(bid_v)
        price = {"timestamp": None, "ask1_price": float(self.ask1), "bid1_price": float(self.bid1), "midprice": float((self.ask1 + self.bid1) / 2.0)}
        return ask, bid, price


def _simulate_day(config: GenerateConfig, symbol: str, day: str) -> dict[str, pd.DataFrame]:
    profile = _SYMBOL_PROFILES[symbol]
    profile = SymbolProfile(
        base_price=config.base_prices.get(symbol, profile.base_price),
        events_per_day=config.events_per_day.get(symbol, profile.events_per_day),
        depth_scale=profile.depth_scale,
        alpha_persistence=profile.alpha_persistence,
        alpha_noise=profile.alpha_noise,
        market_order_bias=profile.market_order_bias,
        add_rate=profile.add_rate,
        cancel_rate=profile.cancel_rate,
    )
    digest = hashlib.blake2b(f"{symbol}|{day}|{config.seed}".encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="big", signed=False) % 1_000_000_000
    rng = np.random.default_rng(seed)
    timestamps = _session_timestamps(pd.Timestamp(day), config.session_windows, profile.events_per_day, rng)
    book = SyntheticOrderBook(
        profile,
        config.tick_size,
        config.trade_unit,
        rng,
        config.alpha_signal_scale,
        config.price_noise_scale,
        config.market_order_impact_scale,
        config.flow_reversion_scale,
        config.spread_widen_prob,
        config.spread_imbalance_threshold,
        config.spread_alpha_threshold,
        config.recenter_follow_scale,
    )

    ask_rows = []
    bid_rows = []
    price_rows = []
    msg_rows = []
    trade_rows = []
    latent_rows = []
    for timestamp in timestamps:
        latent, msg, trade = book.step()
        ask_row, bid_row, price_row = book.book_snapshot()
        ask_row["timestamp"] = timestamp
        bid_row["timestamp"] = timestamp
        price_row["timestamp"] = timestamp
        ask_rows.append(ask_row)
        bid_rows.append(bid_row)
        price_rows.append(price_row)
        msg_rows.append({"timestamp": timestamp, **msg})
        latent_rows.append(
            {
                "timestamp": timestamp,
                **latent,
                "bid1_price": price_row["bid1_price"],
                "ask1_price": price_row["ask1_price"],
                "midprice": price_row["midprice"],
                "bid1_volume": bid_row["bid1_volume"],
                "ask1_volume": ask_row["ask1_volume"],
            }
        )
        if trade["aggressor_side"]:
            trade_rows.append({"timestamp": timestamp, **trade})

    ask_df = pd.DataFrame(ask_rows)
    bid_df = pd.DataFrame(bid_rows)
    price_df = pd.DataFrame(price_rows)
    msg_df = pd.DataFrame(msg_rows)
    trades_df = pd.DataFrame(trade_rows, columns=["timestamp", "price", "size", "aggressor_side"])
    latent_df = pd.DataFrame(latent_rows)
    return {
        "ask": ask_df,
        "bid": bid_df,
        "price": price_df,
        "msg": msg_df,
        "trades": trades_df,
        "latent": latent_df,
    }


def generate_dataset(config: GenerateConfig) -> dict[str, dict[str, int]]:
    config.apply_mode_defaults()
    set_seed(config.seed)
    root = Path(config.data_dir)
    summary: dict[str, dict[str, int]] = {}
    for symbol in config.symbols:
        summary[symbol] = {}
        for day_idx in range(config.num_days):
            day = _day_label(day_idx)
            day_root = root / symbol / day
            if day_root.exists() and config.overwrite:
                shutil.rmtree(day_root)
            ensure_dir(day_root)
            frames = _simulate_day(config, symbol, day)
            frames["ask"].to_csv(day_root / "ask.csv", index=False)
            frames["bid"].to_csv(day_root / "bid.csv", index=False)
            frames["price"].to_csv(day_root / "price.csv", index=False)
            frames["msg"].to_csv(day_root / "msg.csv", index=False)
            frames["trades"].to_csv(day_root / "trades.csv", index=False)
            frames["latent"].to_csv(day_root / "latent.csv", index=False)
            summary[symbol][day] = int(len(frames["price"]))
    save_json(root / "simulation_summary.json", summary)
    save_json(
        root / "simulation_config.json",
        {
            "symbols": config.symbols,
            "num_days": config.num_days,
            "session_windows": config.session_windows,
            "stable_windows": config.stable_windows,
            "tick_size": config.tick_size,
            "trade_unit": config.trade_unit,
            "alpha_signal_scale": config.alpha_signal_scale,
        },
    )
    return summary


@pyrallis.wrap()
def main(config: GenerateConfig) -> None:
    generate_dataset(config)


if __name__ == "__main__":
    main()
