from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import DiagnosticsConfig
from .simulator import SyntheticDay


@dataclass
class QuoteDecision:
    ask_price: float
    bid_price: float
    ask_volume: int
    bid_volume: int
    spread: float
    reservation: float


@dataclass
class AvellanedaStoikovCalibration:
    sigma_event: float
    sigma2_event: float
    kappa: float
    fill_profile: dict[int, float]


@dataclass
class EpisodeResult:
    policy: str
    day: str
    episode_index: int
    pnl: float
    nd_pnl: float
    pnl_map: float
    profit_ratio: float
    avg_position: float
    avg_abs_position: float
    avg_spread: float
    turnover: float
    trades: int
    fill_rate: float


class FixedLevelPolicy:
    name = "Fixed"

    def __init__(self, level: int, config: DiagnosticsConfig) -> None:
        self.level = level
        self.config = config
        self.name = f"Fixed_{level}"

    def quote(self, day: SyntheticDay, quote_idx: int, inventory: int, step: int, total_steps: int) -> QuoteDecision:
        ask = float(day.ask.iloc[quote_idx][f"ask{self.level}_price"])
        bid = float(day.bid.iloc[quote_idx][f"bid{self.level}_price"])
        return QuoteDecision(
            ask_price=ask,
            bid_price=bid,
            ask_volume=self.config.trade_unit,
            bid_volume=self.config.trade_unit,
            spread=ask - bid,
            reservation=0.5 * (ask + bid),
        )


class AvellanedaStoikovPolicy:
    name = "AS"

    def __init__(self, calibration: AvellanedaStoikovCalibration, config: DiagnosticsConfig) -> None:
        self.calibration = calibration
        self.config = config

    def quote(self, day: SyntheticDay, quote_idx: int, inventory: int, step: int, total_steps: int) -> QuoteDecision:
        mid = float(day.price.iloc[quote_idx]["midprice"])
        tau = max(total_steps - step, 1)
        sigma2 = self.calibration.sigma2_event
        gamma = self.config.as_gamma
        kappa = max(self.calibration.kappa, 1e-6)
        inventory_units = inventory / max(self.config.trade_unit, 1)
        reservation = mid - inventory_units * gamma * sigma2 * tau * mid
        total_spread_ticks = gamma * sigma2 * tau + (2.0 / gamma) * np.log1p(gamma / kappa)
        half_spread = max(self.config.symbol_spec.tick_size, 0.5 * total_spread_ticks * self.config.symbol_spec.tick_size)
        ask = _round_up(reservation + half_spread, self.config.symbol_spec.tick_size)
        bid = _round_down(reservation - half_spread, self.config.symbol_spec.tick_size)
        if ask <= bid:
            ask = bid + self.config.symbol_spec.tick_size
        return QuoteDecision(
            ask_price=ask,
            bid_price=bid,
            ask_volume=self.config.trade_unit,
            bid_volume=self.config.trade_unit,
            spread=ask - bid,
            reservation=reservation,
        )


def stable_episode_spans(day: SyntheticDay, config: DiagnosticsConfig) -> list[tuple[int, int]]:
    stable = _stable_mask(day.price["timestamp"], config.stable_windows)
    stable_idx = np.flatnonzero(stable)
    if stable_idx.size == 0:
        return []
    spans: list[tuple[int, int]] = []
    block_start = 0
    for idx in range(1, stable_idx.size + 1):
        boundary = idx == stable_idx.size or stable_idx[idx] != stable_idx[idx - 1] + 1
        if not boundary:
            continue
        block = stable_idx[block_start:idx]
        for offset in range(0, len(block), config.episode_length):
            slice_idx = block[offset : offset + config.episode_length]
            if len(slice_idx) == config.episode_length:
                spans.append((int(slice_idx[0]), int(slice_idx[-1]) + 1))
        block_start = idx
    return spans


def calibrate_avellaneda_stoikov(days: list[SyntheticDay], config: DiagnosticsConfig) -> AvellanedaStoikovCalibration:
    returns = []
    fill_profile: dict[int, float] = {}
    samples_per_distance: dict[int, int] = {}
    hits_per_distance: dict[int, int] = {}
    for day in days:
        stable = _stable_mask(day.price["timestamp"], config.stable_windows)
        mid = day.price["midprice"].to_numpy(dtype=np.float64)
        day_returns = np.diff(mid) / np.clip(mid[:-1], 1e-8, None)
        if day_returns.size:
            returns.append(day_returns[stable[1:]])
        arrays = _CalibrationArrays.from_day(day)
        candidate_indices = np.flatnonzero(stable)
        if candidate_indices.size == 0:
            continue
        max_candidates = 400
        if config.mode == "smoke":
            max_candidates = 80
        elif config.mode == "medium":
            max_candidates = 200
        stride = max(1, len(candidate_indices) // max_candidates)
        for idx in candidate_indices[::stride]:
            quote_idx = max(idx - config.latency, 0)
            best_ask = float(arrays.ask1[quote_idx])
            best_bid = float(arrays.bid1[quote_idx])
            for distance in range(config.as_max_distance_ticks + 1):
                delta = distance * config.symbol_spec.tick_size
                sell_hit = _would_fill_sell_fast(arrays, idx, best_ask + delta, config.as_fill_horizon_events, day.day)
                buy_hit = _would_fill_buy_fast(arrays, idx, best_bid - delta, config.as_fill_horizon_events, day.day)
                hits_per_distance[distance] = hits_per_distance.get(distance, 0) + int(sell_hit) + int(buy_hit)
                samples_per_distance[distance] = samples_per_distance.get(distance, 0) + 2
    sigma_event = float(np.std(np.concatenate(returns))) if returns else 1e-4
    sigma2_event = max(sigma_event**2, 1e-8)
    for distance in range(config.as_max_distance_ticks + 1):
        samples = samples_per_distance.get(distance, 0)
        fill_profile[distance] = hits_per_distance.get(distance, 0) / samples if samples else 0.0
    xs = np.asarray([distance for distance, prob in fill_profile.items() if prob > 0], dtype=np.float64)
    ys = np.asarray([fill_profile[int(distance)] for distance in xs], dtype=np.float64)
    if xs.size >= 2:
        slope, intercept = np.polyfit(xs, np.log(np.clip(ys, 1e-8, None)), 1)
        kappa = max(-slope, 0.05)
    else:
        kappa = 0.8
    return AvellanedaStoikovCalibration(
        sigma_event=sigma_event,
        sigma2_event=sigma2_event,
        kappa=float(kappa),
        fill_profile=fill_profile,
    )


def replay_policy(day: SyntheticDay, policy, config: DiagnosticsConfig) -> list[EpisodeResult]:
    spans = stable_episode_spans(day, config)
    if not spans:
        return []
    grouped_trades = day.trades.groupby("timestamp") if not day.trades.empty else None
    results: list[EpisodeResult] = []
    for episode_index, (start, stop) in enumerate(spans):
        cash = 0.0
        inventory = 0
        turnover = 0.0
        trade_count = 0
        fill_steps = 0
        spreads = []
        inventory_path = []
        for step, event_idx in enumerate(range(start, stop)):
            quote_idx = max(event_idx - config.latency, 0)
            decision = policy.quote(day, quote_idx, inventory, step, stop - start)
            spreads.append(decision.spread)
            event_time = day.price.iloc[event_idx]["timestamp"]
            fills = []
            fills.extend(_match_sell(day, event_idx, decision.ask_price, decision.ask_volume, grouped_trades, quote_key=(policy.name, day.day, episode_index, event_idx, "ask")))
            fills.extend(_match_buy(day, event_idx, decision.bid_price, decision.bid_volume, grouped_trades, quote_key=(policy.name, day.day, episode_index, event_idx, "bid")))
            if fills:
                fill_steps += 1
            for price, signed_volume in fills:
                inventory += signed_volume
                cash -= signed_volume * price
                turnover += abs(signed_volume * price)
                trade_count += 1
            inventory_path.append(float(inventory))
        final_idx = stop - 1
        if inventory != 0:
            close_price = float(day.price.iloc[final_idx]["bid1_price"] if inventory > 0 else day.price.iloc[final_idx]["ask1_price"])
            cash += inventory * close_price
            turnover += abs(inventory * close_price)
            trade_count += 1
            inventory = 0
            inventory_path.append(0.0)
        pnl = cash
        avg_spread = float(np.mean(spreads)) if spreads else config.symbol_spec.tick_size
        inv_series = np.asarray(inventory_path if inventory_path else [0.0], dtype=np.float64)
        results.append(
            EpisodeResult(
                policy=policy.name,
                day=day.day,
                episode_index=episode_index,
                pnl=float(pnl),
                nd_pnl=float(pnl / max(avg_spread, config.symbol_spec.tick_size)),
                pnl_map=float(pnl / max(np.mean(np.abs(inv_series)), 1.0)),
                profit_ratio=float(pnl / max(turnover, 1e-8)),
                avg_position=float(np.mean(inv_series)),
                avg_abs_position=float(np.mean(np.abs(inv_series))),
                avg_spread=float(avg_spread),
                turnover=float(turnover),
                trades=int(trade_count),
                fill_rate=float(fill_steps / max(stop - start, 1)),
            )
        )
    return results


def summarize(results: list[EpisodeResult]) -> dict[str, float]:
    if not results:
        return {}
    frame = pd.DataFrame([asdict(result) for result in results])
    pnl = frame["pnl"].to_numpy(dtype=np.float64)
    sharpe = 0.0
    if pnl.size > 1 and np.std(pnl, ddof=1) > 0:
        sharpe = float(np.mean(pnl) / np.std(pnl, ddof=1))
    return {
        "episodes": float(len(frame)),
        "pnl_mean": float(frame["pnl"].mean()),
        "pnl_std": float(frame["pnl"].std(ddof=0)),
        "nd_pnl_mean": float(frame["nd_pnl"].mean()),
        "pnl_map_mean": float(frame["pnl_map"].mean()),
        "profit_ratio_mean": float(frame["profit_ratio"].mean()),
        "avg_abs_position_mean": float(frame["avg_abs_position"].mean()),
        "avg_spread_mean": float(frame["avg_spread"].mean()),
        "turnover_mean": float(frame["turnover"].mean()),
        "trades_mean": float(frame["trades"].mean()),
        "fill_rate_mean": float(frame["fill_rate"].mean()),
        "sharpe": sharpe,
    }


def calibration_to_dict(calibration: AvellanedaStoikovCalibration) -> dict[str, float | dict[int, float]]:
    payload = asdict(calibration)
    payload["fill_profile"] = {str(key): value for key, value in calibration.fill_profile.items()}
    return payload


def _match_sell(
    day: SyntheticDay,
    event_idx: int,
    ask_price: float,
    ask_volume: int,
    grouped_trades,
    quote_key: tuple[str, str, int, int, str],
) -> list[tuple[float, int]]:
    if ask_volume <= 0:
        return []
    best_bid = float(day.price.iloc[event_idx]["bid1_price"])
    if ask_price <= best_bid:
        return [(best_bid, -ask_volume)]
    block = _trades_at(grouped_trades, day.price.iloc[event_idx]["timestamp"])
    if block.empty:
        return []
    buys = block[block["aggressor_side"] == "B"]
    if buys.empty:
        return []
    best_trade = float(buys["price"].max())
    if best_trade > ask_price:
        return [(ask_price, -ask_volume)]
    if np.isclose(best_trade, ask_price):
        trade_size = int(buys.loc[np.isclose(buys["price"], ask_price), "size"].sum())
        depth = _depth_at(day.ask.iloc[event_idx], "ask", ask_price)
        probability = trade_size / max(trade_size + depth, 1)
        if _stable_uniform(quote_key) < probability:
            return [(ask_price, -ask_volume)]
    return []


def _match_buy(
    day: SyntheticDay,
    event_idx: int,
    bid_price: float,
    bid_volume: int,
    grouped_trades,
    quote_key: tuple[str, str, int, int, str],
) -> list[tuple[float, int]]:
    if bid_volume <= 0:
        return []
    best_ask = float(day.price.iloc[event_idx]["ask1_price"])
    if bid_price >= best_ask:
        return [(best_ask, bid_volume)]
    block = _trades_at(grouped_trades, day.price.iloc[event_idx]["timestamp"])
    if block.empty:
        return []
    sells = block[block["aggressor_side"] == "A"]
    if sells.empty:
        return []
    best_trade = float(sells["price"].min())
    if best_trade < bid_price:
        return [(bid_price, bid_volume)]
    if np.isclose(best_trade, bid_price):
        trade_size = int(sells.loc[np.isclose(sells["price"], bid_price), "size"].sum())
        depth = _depth_at(day.bid.iloc[event_idx], "bid", bid_price)
        probability = trade_size / max(trade_size + depth, 1)
        if _stable_uniform(quote_key) < probability:
            return [(bid_price, bid_volume)]
    return []


def _trades_at(grouped_trades, timestamp: pd.Timestamp) -> pd.DataFrame:
    if grouped_trades is None or timestamp not in grouped_trades.groups:
        return pd.DataFrame(columns=["price", "size", "aggressor_side"])
    return grouped_trades.get_group(timestamp)


def _depth_at(row: pd.Series, prefix: str, price: float) -> int:
    for level in range(1, 11):
        level_price = float(row[f"{prefix}{level}_price"])
        if np.isclose(level_price, price):
            return int(row[f"{prefix}{level}_volume"])
    return 0


def _stable_mask(timestamps: pd.Series, windows: list[str]) -> np.ndarray:
    clock = timestamps.dt.strftime("%H:%M:%S")
    mask = np.zeros(len(clock), dtype=bool)
    for raw in windows:
        start, end = raw.split("-", maxsplit=1)
        mask |= (clock >= start) & (clock <= end)
    return mask


def _would_fill_sell(day: SyntheticDay, start_idx: int, ask_price: float, horizon: int, grouped_trades) -> bool:
    stop = min(start_idx + horizon, len(day.price))
    for idx in range(start_idx, stop):
        fills = _match_sell(day, idx, ask_price, 100, grouped_trades, ("calib", day.day, 0, idx, "ask", ask_price))
        if fills:
            return True
    return False


def _would_fill_buy(day: SyntheticDay, start_idx: int, bid_price: float, horizon: int, grouped_trades) -> bool:
    stop = min(start_idx + horizon, len(day.price))
    for idx in range(start_idx, stop):
        fills = _match_buy(day, idx, bid_price, 100, grouped_trades, ("calib", day.day, 0, idx, "bid", bid_price))
        if fills:
            return True
    return False


class _CalibrationArrays:
    def __init__(
        self,
        *,
        ask1: np.ndarray,
        bid1: np.ndarray,
        ask_prices: np.ndarray,
        bid_prices: np.ndarray,
        ask_volumes: np.ndarray,
        bid_volumes: np.ndarray,
        trade_price_by_event: list[np.ndarray],
        trade_size_by_event: list[np.ndarray],
        trade_side_by_event: list[np.ndarray],
    ) -> None:
        self.ask1 = ask1
        self.bid1 = bid1
        self.ask_prices = ask_prices
        self.bid_prices = bid_prices
        self.ask_volumes = ask_volumes
        self.bid_volumes = bid_volumes
        self.trade_price_by_event = trade_price_by_event
        self.trade_size_by_event = trade_size_by_event
        self.trade_side_by_event = trade_side_by_event

    @classmethod
    def from_day(cls, day: SyntheticDay) -> _CalibrationArrays:
        event_by_time = {timestamp: idx for idx, timestamp in enumerate(day.price["timestamp"])}
        prices: list[list[float]] = [[] for _ in range(len(day.price))]
        sizes: list[list[int]] = [[] for _ in range(len(day.price))]
        sides: list[list[str]] = [[] for _ in range(len(day.price))]
        if not day.trades.empty:
            for row in day.trades[["timestamp", "price", "size", "aggressor_side"]].itertuples(index=False):
                idx = event_by_time.get(row.timestamp)
                if idx is None:
                    continue
                prices[idx].append(float(row.price))
                sizes[idx].append(int(row.size))
                sides[idx].append(str(row.aggressor_side))
        return cls(
            ask1=day.price["ask1_price"].to_numpy(dtype=np.float64),
            bid1=day.price["bid1_price"].to_numpy(dtype=np.float64),
            ask_prices=day.ask[[f"ask{level}_price" for level in range(1, 11)]].to_numpy(dtype=np.float64),
            bid_prices=day.bid[[f"bid{level}_price" for level in range(1, 11)]].to_numpy(dtype=np.float64),
            ask_volumes=day.ask[[f"ask{level}_volume" for level in range(1, 11)]].to_numpy(dtype=np.int64),
            bid_volumes=day.bid[[f"bid{level}_volume" for level in range(1, 11)]].to_numpy(dtype=np.int64),
            trade_price_by_event=[np.asarray(items, dtype=np.float64) for items in prices],
            trade_size_by_event=[np.asarray(items, dtype=np.int64) for items in sizes],
            trade_side_by_event=[np.asarray(items, dtype=object) for items in sides],
        )


def _would_fill_sell_fast(arrays: _CalibrationArrays, start_idx: int, ask_price: float, horizon: int, day: str) -> bool:
    stop = min(start_idx + horizon, len(arrays.ask1))
    for idx in range(start_idx, stop):
        if ask_price <= arrays.bid1[idx]:
            return True
        prices = arrays.trade_price_by_event[idx]
        if prices.size == 0:
            continue
        sides = arrays.trade_side_by_event[idx]
        buy_mask = sides == "B"
        if not np.any(buy_mask):
            continue
        buy_prices = prices[buy_mask]
        best_trade = float(np.max(buy_prices))
        if best_trade > ask_price:
            return True
        if np.isclose(best_trade, ask_price):
            sizes = arrays.trade_size_by_event[idx][buy_mask]
            traded = int(np.sum(sizes[np.isclose(buy_prices, ask_price)]))
            depth = _depth_at_arrays(arrays.ask_prices[idx], arrays.ask_volumes[idx], ask_price)
            probability = traded / max(traded + depth, 1)
            if _stable_uniform(("calib", day, 0, idx, "ask", ask_price)) < probability:
                return True
    return False


def _would_fill_buy_fast(arrays: _CalibrationArrays, start_idx: int, bid_price: float, horizon: int, day: str) -> bool:
    stop = min(start_idx + horizon, len(arrays.bid1))
    for idx in range(start_idx, stop):
        if bid_price >= arrays.ask1[idx]:
            return True
        prices = arrays.trade_price_by_event[idx]
        if prices.size == 0:
            continue
        sides = arrays.trade_side_by_event[idx]
        sell_mask = sides == "A"
        if not np.any(sell_mask):
            continue
        sell_prices = prices[sell_mask]
        best_trade = float(np.min(sell_prices))
        if best_trade < bid_price:
            return True
        if np.isclose(best_trade, bid_price):
            sizes = arrays.trade_size_by_event[idx][sell_mask]
            traded = int(np.sum(sizes[np.isclose(sell_prices, bid_price)]))
            depth = _depth_at_arrays(arrays.bid_prices[idx], arrays.bid_volumes[idx], bid_price)
            probability = traded / max(traded + depth, 1)
            if _stable_uniform(("calib", day, 0, idx, "bid", bid_price)) < probability:
                return True
    return False


def _depth_at_arrays(prices: np.ndarray, volumes: np.ndarray, price: float) -> int:
    matches = np.isclose(prices, price)
    if not np.any(matches):
        return 0
    return int(volumes[np.argmax(matches)])


def _stable_uniform(parts: tuple[object, ...]) -> float:
    payload = json.dumps(parts, separators=(",", ":"), default=str).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) / float(1 << 64)


def _round_up(value: float, tick_size: float) -> float:
    return np.ceil(value / tick_size) * tick_size


def _round_down(value: float, tick_size: float) -> float:
    return np.floor(value / tick_size) * tick_size
