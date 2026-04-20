from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyrallis

from .config import GenerateConfig
from .utils import ensure_dir, price_legal_check, save_json, set_seed


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
class SymbolProfile:
    base_price: float
    depth_scale: float
    alpha_persistence: float
    alpha_noise: float
    market_order_bias: float
    add_rate: float
    cancel_rate: float


def _day_label(idx: int) -> str:
    return (pd.Timestamp("2019-11-01") + pd.tseries.offsets.BDay(idx)).strftime("%Y%m%d")


def _session_timestamps(anchor_day: pd.Timestamp, session_windows: list[str], event_count: int, rng: np.random.Generator) -> pd.DatetimeIndex:
    windows = []
    for raw in session_windows:
        start, end = raw.split("-", maxsplit=1)
        windows.append((anchor_day + pd.to_timedelta(start), anchor_day + pd.to_timedelta(end)))
    durations = np.asarray([(end - start).total_seconds() for start, end in windows], dtype=np.float64)
    counts = np.floor(event_count * durations / durations.sum()).astype(int)
    counts[-1] += event_count - int(counts.sum())
    parts: list[pd.DatetimeIndex] = []
    for (start, end), count, seconds in zip(windows, counts, durations, strict=True):
        if count <= 0:
            continue
        waits = np.cumsum(rng.gamma(shape=2.0, scale=1.0, size=count))
        waits = waits / waits[-1] * max(seconds - 1e-6, 1e-6)
        ns = np.round(waits * 1_000_000_000).astype(np.int64)
        parts.append(pd.DatetimeIndex(start.value + ns))
    return parts[0].append(parts[1:]).sort_values() if parts else pd.DatetimeIndex([])


def _weighted_level(rng: np.random.Generator, levels: int = 10) -> int:
    weights = np.exp(-0.55 * np.arange(levels))
    weights = weights / weights.sum()
    return int(rng.choice(np.arange(levels), p=weights))


class SyntheticOrderBook:
    def __init__(self, config: GenerateConfig, profile: SymbolProfile, rng: np.random.Generator) -> None:
        self.config = config
        self.profile = profile
        self.rng = rng
        self.mid = round(profile.base_price / config.tick_size) * config.tick_size
        self.efficient_price = self.mid
        self.spread_ticks = 1
        self.bid_volumes = self._init_depth(profile.depth_scale)
        self.ask_volumes = self._init_depth(profile.depth_scale)
        self.alpha = 0.0
        self.regime = 0
        self.signed_flow_state = 0.0
        self.regime_clock = 0

    def _init_depth(self, scale: float) -> np.ndarray:
        base = np.linspace(14, 6, 10) * self.config.trade_unit * scale
        noise = self.rng.uniform(0.7, 1.3, size=10)
        return np.maximum(np.round(base * noise / self.config.trade_unit) * self.config.trade_unit, self.config.trade_unit).astype(np.float32)

    @property
    def bid1(self) -> float:
        return round(self.mid - self.spread_ticks * self.config.tick_size / 2.0, 6)

    @property
    def ask1(self) -> float:
        return round(self.mid + self.spread_ticks * self.config.tick_size / 2.0, 6)

    def bid_prices(self) -> np.ndarray:
        return np.asarray([self.bid1 - i * self.config.tick_size for i in range(10)], dtype=np.float32)

    def ask_prices(self) -> np.ndarray:
        return np.asarray([self.ask1 + i * self.config.tick_size for i in range(10)], dtype=np.float32)

    def _step_latent(self) -> tuple[int, float]:
        self.regime_clock += 1
        regime_shift = 0
        if self.regime_clock > 350 and self.rng.random() < 0.004:
            self.regime = int(self.rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3]))
            self.regime_clock = 0
            regime_shift = 1
        target = self.config.alpha_signal_scale * 0.35 * self.regime
        self.alpha = (
            self.profile.alpha_persistence * self.alpha
            + (1.0 - self.profile.alpha_persistence) * target
            + self.rng.normal(0.0, 0.55 * self.profile.alpha_noise * self.config.alpha_signal_scale)
        )
        efficient_move = 0.0035 * self.alpha + self.rng.normal(0.0, 0.28 * self.config.price_noise_scale)
        self.efficient_price = max(self.config.tick_size, self.efficient_price + efficient_move)
        return regime_shift, efficient_move

    def _top_imbalance(self) -> float:
        return float((self.bid_volumes[0] - self.ask_volumes[0]) / max(self.bid_volumes[0] + self.ask_volumes[0], 1e-8))

    def _desired_spread_ticks(self) -> int:
        imbalance = abs(self._top_imbalance())
        if (
            imbalance > self.config.spread_imbalance_threshold
            or abs(self.alpha) > self.config.spread_alpha_threshold * self.config.alpha_signal_scale
            or self.rng.random() < self.config.spread_widen_prob
        ):
            return 2
        return 1

    def _choose_event(self) -> tuple[str, str]:
        imbalance = self._top_imbalance()
        flow_pressure = float(np.tanh(self.signed_flow_state / 18.0))
        flow_term = self.config.flow_reversion_scale * flow_pressure
        alpha = self.alpha
        market_buy = np.exp(
            self.profile.market_order_bias
            - self.config.market_order_alpha_sensitivity * alpha
            - self.config.market_order_imbalance_sensitivity * imbalance
            - self.config.market_order_flow_sensitivity * flow_term
        )
        market_sell = np.exp(
            self.profile.market_order_bias
            + self.config.market_order_alpha_sensitivity * alpha
            + self.config.market_order_imbalance_sensitivity * imbalance
            + self.config.market_order_flow_sensitivity * flow_term
        )
        limit_buy = np.exp(
            self.profile.add_rate
            - self.config.limit_alpha_sensitivity * alpha
            + 0.25 * max(imbalance, 0.0)
            + self.config.limit_alpha_sensitivity * flow_term
        )
        limit_sell = np.exp(
            self.profile.add_rate
            + self.config.limit_alpha_sensitivity * alpha
            + 0.25 * max(-imbalance, 0.0)
            - self.config.limit_alpha_sensitivity * flow_term
        )
        cancel_buy = np.exp(
            self.profile.cancel_rate
            + self.config.cancel_alpha_sensitivity * alpha
            - 0.10 * max(imbalance, 0.0)
            - 0.06 * flow_term
        )
        cancel_sell = np.exp(
            self.profile.cancel_rate
            - self.config.cancel_alpha_sensitivity * alpha
            - 0.10 * max(-imbalance, 0.0)
            + 0.06 * flow_term
        )
        weights = np.asarray([market_buy, market_sell, limit_buy, limit_sell, cancel_buy, cancel_sell], dtype=np.float64)
        weights = weights / weights.sum()
        idx = int(self.rng.choice(np.arange(6), p=weights))
        mapping = [
            ("market", "buy"),
            ("market", "sell"),
            ("limit", "buy"),
            ("limit", "sell"),
            ("cancel", "buy"),
            ("cancel", "sell"),
        ]
        return mapping[idx]

    def _draw_size(self) -> float:
        lots = int(self.rng.choice([1, 2, 3, 4], p=[0.55, 0.25, 0.12, 0.08]))
        return float(max(self.config.trade_unit, lots * self.config.trade_unit))

    def _recenter(self) -> None:
        target_mid = round(self.efficient_price / self.config.tick_size) * self.config.tick_size
        gap_ticks = int(round((target_mid - self.mid) / self.config.tick_size))
        if gap_ticks != 0:
            gap_abs = abs(gap_ticks)
            follow_prob = min(
                0.75,
                self.config.recenter_follow_scale
                * (
                    self.config.recenter_base_prob
                    + self.config.recenter_gap_scale * gap_abs
                    + self.config.recenter_alpha_scale * min(abs(self.alpha), 1.0)
                ),
            )
            if self.rng.random() < follow_prob:
                move = 1 if gap_ticks > 0 else -1
                if move > 0:
                    self.bid_volumes = np.roll(self.bid_volumes, -1)
                    self.ask_volumes = np.roll(self.ask_volumes, -1)
                    self.bid_volumes[-1] = self._init_depth(self.profile.depth_scale)[0]
                    self.ask_volumes[-1] = self._init_depth(self.profile.depth_scale)[0]
                else:
                    self.bid_volumes = np.roll(self.bid_volumes, 1)
                    self.ask_volumes = np.roll(self.ask_volumes, 1)
                    self.bid_volumes[0] = self._init_depth(self.profile.depth_scale)[0]
                    self.ask_volumes[0] = self._init_depth(self.profile.depth_scale)[0]
                self.mid = round(self.mid + move * self.config.tick_size, 6)
        self.spread_ticks = self._desired_spread_ticks()
        ask, bid = price_legal_check(self.ask1, self.bid1, self.config.tick_size)
        self.mid = round((ask + bid) / 2.0, 6)
        self.spread_ticks = max(1, int(round((ask - bid) / self.config.tick_size)))

    def snapshot(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        ask_row = {"timestamp": None}
        bid_row = {"timestamp": None}
        for level, (ask_p, ask_v, bid_p, bid_v) in enumerate(zip(self.ask_prices(), self.ask_volumes, self.bid_prices(), self.bid_volumes, strict=True), start=1):
            ask_row[f"ask{level}_price"] = float(ask_p)
            ask_row[f"ask{level}_volume"] = float(ask_v)
            bid_row[f"bid{level}_price"] = float(bid_p)
            bid_row[f"bid{level}_volume"] = float(bid_v)
        price_row = {"timestamp": None, "ask1_price": float(self.ask1), "bid1_price": float(self.bid1), "midprice": float((self.ask1 + self.bid1) / 2.0)}
        return ask_row, bid_row, price_row

    def step(self) -> tuple[dict[str, float | int | str], dict[str, float], dict[str, float]]:
        regime_shift, efficient_move = self._step_latent()
        event_type, side = self._choose_event()
        msg = {key: 0.0 for key in MSG_COLUMNS}
        trade = {"price": np.nan, "size": np.nan, "aggressor_side": ""}
        queue_pressure = float((self.ask_volumes[:3].sum() - self.bid_volumes[:3].sum()) / max(self.ask_volumes[:3].sum() + self.bid_volumes[:3].sum(), 1e-8))
        size = self._draw_size()
        if event_type == "market":
            if side == "buy":
                trade = {"price": float(self.ask1), "size": size, "aggressor_side": "B"}
                self.ask_volumes[0] -= size
                msg["market_buy_volume"] = size
                msg["market_buy_n"] = 1.0
                self.signed_flow_state = 0.985 * self.signed_flow_state + size / self.config.trade_unit
                self.efficient_price += self.config.market_order_impact_scale * (
                    self.config.market_order_tick_impact * self.config.tick_size
                    + self.config.market_order_alpha_impact * self.alpha
                )
                if self.config.touch_replenish_fraction > 0:
                    self.ask_volumes[0] += self.config.touch_replenish_fraction * size
            else:
                trade = {"price": float(self.bid1), "size": size, "aggressor_side": "A"}
                self.bid_volumes[0] -= size
                msg["market_sell_volume"] = size
                msg["market_sell_n"] = 1.0
                self.signed_flow_state = 0.985 * self.signed_flow_state - size / self.config.trade_unit
                self.efficient_price -= self.config.market_order_impact_scale * (
                    self.config.market_order_tick_impact * self.config.tick_size
                    - self.config.market_order_alpha_impact * self.alpha
                )
                if self.config.touch_replenish_fraction > 0:
                    self.bid_volumes[0] += self.config.touch_replenish_fraction * size
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
                removed = max(0.0, min(size, float(self.bid_volumes[level] - self.config.trade_unit)))
                self.bid_volumes[level] -= removed
                msg["withdraw_buy_volume"] = removed
                msg["withdraw_buy_n"] = 1.0 if removed > 0 else 0.0
            else:
                removed = max(0.0, min(size, float(self.ask_volumes[level] - self.config.trade_unit)))
                self.ask_volumes[level] -= removed
                msg["withdraw_sell_volume"] = removed
                msg["withdraw_sell_n"] = 1.0 if removed > 0 else 0.0
            self.signed_flow_state *= 0.99
        self.bid_volumes = np.maximum(self.bid_volumes, self.config.trade_unit).astype(np.float32)
        self.ask_volumes = np.maximum(self.ask_volumes, self.config.trade_unit).astype(np.float32)
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
            "regime_shift": regime_shift,
            "efficient_move": float(efficient_move),
        }
        return latent, msg, trade


def _symbol_profile(symbol: str, config: GenerateConfig) -> SymbolProfile:
    return SymbolProfile(
        base_price=config.base_prices[symbol],
        depth_scale={"000001": 1.25, "000858": 1.0, "002415": 0.8}.get(symbol, 1.0),
        alpha_persistence={"000001": 0.97, "000858": 0.975, "002415": 0.98}.get(symbol, 0.975),
        alpha_noise={"000001": 0.035, "000858": 0.032, "002415": 0.03}.get(symbol, 0.032),
        market_order_bias={"000001": 0.08, "000858": 0.07, "002415": 0.06}.get(symbol, 0.07),
        add_rate={"000001": 1.05, "000858": 1.0, "002415": 0.96}.get(symbol, 1.0),
        cancel_rate={"000001": 0.82, "000858": 0.78, "002415": 0.72}.get(symbol, 0.78),
    )


def generate_day_frames(symbol: str, day_index: int, config: GenerateConfig) -> dict[str, pd.DataFrame]:
    day = _day_label(day_index)
    seed = abs(hash((config.seed, symbol, day))) % (2**32)
    rng = np.random.default_rng(seed)
    anchor_day = pd.Timestamp(day)
    timestamps = _session_timestamps(anchor_day, config.session_windows, config.events_per_day[symbol], rng)
    profile = _symbol_profile(symbol, config)
    book = SyntheticOrderBook(config, profile, rng)
    ask_rows = []
    bid_rows = []
    price_rows = []
    msg_rows = []
    trade_rows = []
    latent_rows = []
    for ts in timestamps:
        ask_row, bid_row, price_row = book.snapshot()
        ask_row["timestamp"] = ts
        bid_row["timestamp"] = ts
        price_row["timestamp"] = ts
        latent, msg, trade = book.step()
        msg_rows.append({"timestamp": ts, **msg})
        if trade["aggressor_side"]:
            trade_rows.append({"timestamp": ts, "price": trade["price"], "size": trade["size"], "aggressor_side": trade["aggressor_side"]})
        latent_rows.append({"timestamp": ts, **latent})
        ask_rows.append(ask_row)
        bid_rows.append(bid_row)
        price_rows.append(price_row)
    return {
        "ask": pd.DataFrame(ask_rows),
        "bid": pd.DataFrame(bid_rows),
        "price": pd.DataFrame(price_rows),
        "msg": pd.DataFrame(msg_rows),
        "trades": pd.DataFrame(trade_rows),
        "latent": pd.DataFrame(latent_rows),
    }


def generate_dataset(config: GenerateConfig) -> dict[str, object]:
    config.apply_mode_defaults()
    set_seed(config.seed)
    root = ensure_dir(config.data_dir)
    manifest = {"symbols": config.symbols, "days": []}
    for symbol in config.symbols:
        for day_index in range(config.num_days):
            day = _day_label(day_index)
            day_root = root / symbol / day
            if day_root.exists() and not config.overwrite:
                manifest["days"].append({"symbol": symbol, "day": day, "status": "kept"})
                continue
            ensure_dir(day_root)
            frames = generate_day_frames(symbol, day_index, config)
            for name, frame in frames.items():
                frame.to_csv(day_root / f"{name}.csv", index=False)
            manifest["days"].append({"symbol": symbol, "day": day, "status": "generated"})
    save_json(root / "manifest.json", manifest)
    return manifest


@pyrallis.wrap()
def main(config: GenerateConfig) -> None:
    generate_dataset(config)


if __name__ == "__main__":
    main()
