from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import calibrate_avellaneda_stoikov, calibration_to_dict
from .baselines import AvellanedaStoikovCalibration
from .config import DiagnosticsConfig
from .paper_env import EpisodeMetrics
from .simulator import SyntheticDay
from .utils import save_json


def evaluate_paper_baselines(
    train_days: list[SyntheticDay],
    test_days: list[SyntheticDay],
    config: DiagnosticsConfig,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    calibration = calibrate_avellaneda_stoikov(train_days, config)
    all_metrics: list[EpisodeMetrics] = []
    for policy in ["Random", "Fixed_1", "Fixed_2", "Fixed_3", "AS"]:
        for day in test_days:
            all_metrics.extend(_fast_replay_policy(policy, day, config, calibration))
    frame = pd.DataFrame([asdict(metric) for metric in all_metrics])
    frame.to_csv(output_dir / "paper_baseline_episodes.csv", index=False)
    save_json(output_dir / "as_calibration.json", calibration_to_dict(calibration))
    daily = _daily_results(frame)
    daily.to_csv(output_dir / "paper_baseline_daily.csv", index=False)
    return {policy: summarize_metrics(group) for policy, group in frame.groupby("policy")} if not frame.empty else {}


def summarize_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
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
        "reward_mean": float(frame["reward"].mean()),
        "sharpe": sharpe,
    }


def _paper_episode_spans(day: SyntheticDay, config: DiagnosticsConfig) -> list[tuple[int, int]]:
    clock = day.price["timestamp"].dt.strftime("%H:%M:%S")
    mask = np.zeros(len(clock), dtype=bool)
    for raw in config.stable_windows:
        start, end = raw.split("-", maxsplit=1)
        mask |= (clock >= start) & (clock <= end)
    idx = np.flatnonzero(mask)
    spans: list[tuple[int, int]] = []
    block_start = 0
    for pos in range(1, len(idx) + 1):
        boundary = pos == len(idx) or idx[pos] != idx[pos - 1] + 1
        if not boundary:
            continue
        block = idx[block_start:pos]
        for offset in range(0, len(block), config.episode_length):
            window = block[offset : offset + config.episode_length]
            if len(window) == config.episode_length:
                spans.append((int(window[0]), int(window[-1]) + 1))
                if config.max_eval_episodes_per_day is not None and len(spans) >= config.max_eval_episodes_per_day:
                    return spans
        block_start = pos
    return spans


def _fast_replay_policy(
    policy: str,
    day: SyntheticDay,
    config: DiagnosticsConfig,
    calibration: AvellanedaStoikovCalibration,
) -> list[EpisodeMetrics]:
    arrays = _DayArrays.from_day(day)
    results: list[EpisodeMetrics] = []
    for episode_index, (start, stop) in enumerate(_paper_episode_spans(day, config)):
        trade_indices = arrays.trade_event_indices(start, stop, config)
        cash = 0.0
        value = 0.0
        previous_value = 0.0
        inventory = 0
        turnover = 0.0
        trades = 0
        fill_steps = 0
        cumulative_reward = 0.0
        inventory_path: list[float] = []
        spread_path: list[float] = []
        for event_idx in trade_indices:
            previous_value = value
            ask_price, ask_volume, bid_price, bid_volume = _quote(policy, arrays, event_idx, episode_index, stop, inventory, config, calibration)
            ask_price, ask_volume, bid_price, bid_volume = _apply_inventory_guard(ask_price, ask_volume, bid_price, bid_volume, inventory, config)
            fills = arrays.match(event_idx, ask_price, ask_volume, bid_price, bid_volume, config, episode_index)
            matched_pnl = 0.0
            if fills:
                fill_steps += 1
            for trade_price, trade_volume in fills:
                rebate = abs(trade_volume) * config.maker_rebate_per_share
                matched_pnl += (arrays.mid[event_idx] - trade_price) * trade_volume + rebate
                inventory += trade_volume
                cash -= trade_volume * trade_price
                cash += rebate
                turnover += abs(trade_volume * trade_price)
                trades += 1
            value = cash + inventory * arrays.mid[event_idx]
            reward = _reward(
                value=value,
                previous_value=previous_value,
                inventory=inventory,
                trade_price=0.0,
                trade_volume=0,
                mid=arrays.mid[event_idx],
                matched_pnl=matched_pnl,
                ask_price=ask_price,
                bid_price=bid_price,
                config=config,
            )
            cumulative_reward += reward
            inventory_path.append(float(inventory))
            spread_path.append(_quoted_spread(ask_price, bid_price, config))
        if trade_indices:
            final_idx = trade_indices[-1]
            if inventory:
                previous_value = value
                if inventory < 0:
                    close_price = arrays.ask1[max(final_idx - config.latency, 0)]
                    close_volume = -inventory
                else:
                    close_price = arrays.bid1[max(final_idx - config.latency, 0)]
                    close_volume = -inventory
                inventory += close_volume
                cash -= close_volume * close_price
                turnover += abs(close_volume * close_price)
                trades += 1
                value = cash + inventory * arrays.mid[final_idx]
                cumulative_reward += _reward(
                    value=value,
                    previous_value=previous_value,
                    inventory=inventory,
                    trade_price=close_price,
                    trade_volume=close_volume,
                    mid=arrays.mid[final_idx],
                    matched_pnl=None,
                    ask_price=ask_price,
                    bid_price=bid_price,
                    config=config,
                )
        inv = np.asarray(inventory_path if inventory_path else [0.0], dtype=np.float64)
        avg_spread = float(np.mean(spread_path)) if spread_path else config.symbol_spec.tick_size
        pnl = float(value)
        results.append(
            EpisodeMetrics(
                policy=policy,
                day=day.day,
                episode_index=episode_index,
                pnl=pnl,
                nd_pnl=float(pnl / max(avg_spread, config.symbol_spec.tick_size)),
                pnl_map=float(pnl / max(np.mean(np.abs(inv)), 1.0)),
                profit_ratio=float(pnl / max(turnover, 1e-8)),
                avg_position=float(np.mean(inv)),
                avg_abs_position=float(np.mean(np.abs(inv))),
                avg_spread=avg_spread,
                turnover=float(turnover),
                trades=int(trades),
                fill_rate=float(fill_steps / max(len(trade_indices), 1)),
                reward=float(cumulative_reward),
            )
        )
    return results


class _DayArrays:
    def __init__(
        self,
        *,
        day: SyntheticDay,
        mid: np.ndarray,
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
        self.day = day
        self.mid = mid
        self.ask1 = ask1
        self.bid1 = bid1
        self.ask_prices = ask_prices
        self.bid_prices = bid_prices
        self.ask_volumes = ask_volumes
        self.bid_volumes = bid_volumes
        self.trade_price_by_event = trade_price_by_event
        self.trade_size_by_event = trade_size_by_event
        self.trade_side_by_event = trade_side_by_event
        self.trade_events = np.flatnonzero([len(prices) > 0 for prices in trade_price_by_event])

    @classmethod
    def from_day(cls, day: SyntheticDay) -> _DayArrays:
        ask_price_cols = [f"ask{level}_price" for level in range(1, 11)]
        bid_price_cols = [f"bid{level}_price" for level in range(1, 11)]
        ask_volume_cols = [f"ask{level}_volume" for level in range(1, 11)]
        bid_volume_cols = [f"bid{level}_volume" for level in range(1, 11)]
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
            day=day,
            mid=day.price["midprice"].to_numpy(dtype=np.float64),
            ask1=day.price["ask1_price"].to_numpy(dtype=np.float64),
            bid1=day.price["bid1_price"].to_numpy(dtype=np.float64),
            ask_prices=day.ask[ask_price_cols].to_numpy(dtype=np.float64),
            bid_prices=day.bid[bid_price_cols].to_numpy(dtype=np.float64),
            ask_volumes=day.ask[ask_volume_cols].to_numpy(dtype=np.int64),
            bid_volumes=day.bid[bid_volume_cols].to_numpy(dtype=np.int64),
            trade_price_by_event=[np.asarray(items, dtype=np.float64) for items in prices],
            trade_size_by_event=[np.asarray(items, dtype=np.int64) for items in sizes],
            trade_side_by_event=[np.asarray(items, dtype=object) for items in sides],
        )

    def trade_event_indices(self, start: int, stop: int, config: DiagnosticsConfig) -> list[int]:
        lower = max(start, config.lookback + config.latency)
        selected = self.trade_events[(self.trade_events >= lower) & (self.trade_events < stop)]
        if selected.size:
            return [int(idx) for idx in selected]
        return list(range(lower, stop))

    def match(
        self,
        event_idx: int,
        ask_price: float,
        ask_volume: int,
        bid_price: float,
        bid_volume: int,
        config: DiagnosticsConfig,
        episode_index: int,
    ) -> list[tuple[float, int]]:
        fills: list[tuple[float, int]] = []
        previous_idx = max(event_idx - config.latency, 0)
        prices = self.trade_price_by_event[event_idx]
        sizes = self.trade_size_by_event[event_idx]
        sides = self.trade_side_by_event[event_idx]
        if ask_price and ask_volume < 0:
            fill = self._match_sell(ask_price, ask_volume, previous_idx, event_idx, prices, sizes, sides, config, episode_index)
            if fill is not None:
                fills.append(fill)
        if bid_price and bid_volume > 0:
            fill = self._match_buy(bid_price, bid_volume, previous_idx, event_idx, prices, sizes, sides, config, episode_index)
            if fill is not None:
                if config.matching_mode == "author_single":
                    return [fill]
                fills.append(fill)
        if config.matching_mode not in {"author_single", "multi_fill"}:
            raise ValueError(f"Unknown matching_mode: {config.matching_mode}")
        return fills

    def _match_sell(
        self,
        ask_price: float,
        ask_volume: int,
        previous_idx: int,
        event_idx: int,
        prices: np.ndarray,
        sizes: np.ndarray,
        sides: np.ndarray,
        config: DiagnosticsConfig,
        episode_index: int,
    ) -> tuple[float, int] | None:
        if ask_price <= self.bid1[previous_idx]:
            return float(self.bid1[previous_idx]), ask_volume
        side_mask = sides == "B"
        if not np.any(side_mask):
            return None
        buy_prices = prices[side_mask]
        best_trade = float(np.max(buy_prices))
        if best_trade > ask_price:
            return ask_price, ask_volume
        if np.isclose(best_trade, ask_price):
            at_price = np.isclose(buy_prices, ask_price)
            traded = int(np.sum(sizes[side_mask][at_price]))
            depth = _depth_at_arrays(self.ask_prices[event_idx], self.ask_volumes[event_idx], ask_price)
            if _stable_uniform((self.day.day, episode_index, event_idx, "ask", ask_price, config.seed)) < traded / max(traded + depth, 1):
                return ask_price, ask_volume
        return None

    def _match_buy(
        self,
        bid_price: float,
        bid_volume: int,
        previous_idx: int,
        event_idx: int,
        prices: np.ndarray,
        sizes: np.ndarray,
        sides: np.ndarray,
        config: DiagnosticsConfig,
        episode_index: int,
    ) -> tuple[float, int] | None:
        if bid_price >= self.ask1[previous_idx]:
            return float(self.ask1[previous_idx]), bid_volume
        side_mask = sides == "A"
        if not np.any(side_mask):
            return None
        sell_prices = prices[side_mask]
        best_trade = float(np.min(sell_prices))
        if best_trade < bid_price:
            return bid_price, bid_volume
        if np.isclose(best_trade, bid_price):
            at_price = np.isclose(sell_prices, bid_price)
            traded = int(np.sum(sizes[side_mask][at_price]))
            depth = _depth_at_arrays(self.bid_prices[event_idx], self.bid_volumes[event_idx], bid_price)
            if _stable_uniform((self.day.day, episode_index, event_idx, "bid", bid_price, config.seed)) < traded / max(traded + depth, 1):
                return bid_price, bid_volume
        return None


def _quote(
    policy: str,
    arrays: _DayArrays,
    event_idx: int,
    episode_index: int,
    episode_stop: int,
    inventory: int,
    config: DiagnosticsConfig,
    calibration: AvellanedaStoikovCalibration,
) -> tuple[float, int, float, int]:
    quote_idx = max(event_idx - config.latency, 0)
    lot = config.trade_unit
    if policy.startswith("Fixed_"):
        level = int(policy.rsplit("_", maxsplit=1)[1]) - 1
        return float(arrays.ask_prices[quote_idx, level]), -lot, float(arrays.bid_prices[quote_idx, level]), lot
    if policy == "Random":
        seed = _random_policy_seed((arrays.day.day, episode_index, event_idx, config.seed))
        rng = np.random.default_rng(seed)
        ask_level = int(rng.integers(1, 6)) - 1
        bid_level = int(rng.integers(1, 6)) - 1
        return float(arrays.ask_prices[quote_idx, ask_level]), -lot, float(arrays.bid_prices[quote_idx, bid_level]), lot
    if policy == "AS":
        mid = float(arrays.mid[quote_idx])
        tau = max(episode_stop - event_idx, 1)
        gamma = config.as_gamma
        sigma2 = calibration.sigma2_event
        kappa = max(calibration.kappa, 1e-6)
        inventory_units = inventory / max(lot, 1)
        reservation = mid - inventory_units * gamma * sigma2 * tau * mid
        total_spread_ticks = gamma * sigma2 * tau + (2.0 / gamma) * np.log1p(gamma / kappa)
        half_spread = max(config.symbol_spec.tick_size, 0.5 * total_spread_ticks * config.symbol_spec.tick_size)
        ask = _round_up(reservation + half_spread, config.symbol_spec.tick_size)
        bid = _round_down(reservation - half_spread, config.symbol_spec.tick_size)
        if ask <= bid:
            ask = bid + config.symbol_spec.tick_size
        return ask, -lot, bid, lot
    raise ValueError(f"Unknown paper baseline policy: {policy}")


def _apply_inventory_guard(
    ask_price: float,
    ask_volume: int,
    bid_price: float,
    bid_volume: int,
    inventory: int,
    config: DiagnosticsConfig,
) -> tuple[float, int, float, int]:
    max_inventory = config.max_inventory_units * config.trade_unit
    if inventory <= -max_inventory:
        ask_price = 0.0
        ask_volume = 0
    if inventory >= max_inventory:
        bid_price = 0.0
        bid_volume = 0
    return ask_price, ask_volume, bid_price, bid_volume


def _reward(
    *,
    value: float,
    previous_value: float,
    inventory: int,
    trade_price: float,
    trade_volume: int,
    mid: float,
    matched_pnl: float | None,
    ask_price: float,
    bid_price: float,
    config: DiagnosticsConfig,
) -> float:
    pnl = value - previous_value
    if matched_pnl is None:
        matched_pnl = (mid - trade_price) * trade_volume if trade_volume else 0.0
    spread = _quoted_spread(ask_price, bid_price, config)
    spread_penalty = 0.0
    if inventory == 0 and _is_two_sided(ask_price, bid_price) and spread > config.reward_spread_penalty_threshold:
        spread_penalty = config.reward_spread_penalty_scale * spread
    if config.reward_mode == "author_pnl":
        return float(pnl - spread_penalty)
    if config.reward_mode != "hybrid":
        raise ValueError(f"Unknown reward_mode: {config.reward_mode}")
    dampened_pnl = pnl - max(0.0, config.reward_eta * pnl)
    inventory_penalty = config.reward_zeta * (inventory / config.trade_unit) ** 2
    base_pnl = dampened_pnl if config.reward_use_dampened_pnl else pnl
    reward = config.reward_pnl_weight * base_pnl
    reward += config.reward_trading_pnl_weight * matched_pnl if config.reward_use_trading_pnl else 0.0
    reward -= config.reward_inventory_penalty_weight * inventory_penalty if config.reward_use_inventory_penalty else 0.0
    reward -= config.reward_spread_penalty_weight * spread_penalty
    return float(reward)


def _depth_at_arrays(prices: np.ndarray, volumes: np.ndarray, price: float) -> int:
    matches = np.isclose(prices, price)
    if not np.any(matches):
        return 0
    return int(volumes[np.argmax(matches)])


def _quoted_spread(ask_price: float, bid_price: float, config: DiagnosticsConfig) -> float:
    if not _is_two_sided(ask_price, bid_price):
        return config.symbol_spec.tick_size
    return float(max(ask_price - bid_price, config.symbol_spec.tick_size))


def _is_two_sided(ask_price: float, bid_price: float) -> bool:
    return bool(ask_price > 0.0 and bid_price > 0.0 and np.isfinite(ask_price) and np.isfinite(bid_price))


def _stable_uniform(parts: tuple[object, ...]) -> float:
    payload = json.dumps(parts, separators=(",", ":"), default=str).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) / float(1 << 64)


def _random_policy_seed(parts: tuple[object, ...]) -> int:
    digest = hashlib.blake2b(json.dumps(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) % (2**32)


def _round_up(value: float, tick_size: float) -> float:
    return float(np.ceil(value / tick_size) * tick_size)


def _round_down(value: float, tick_size: float) -> float:
    return float(np.floor(value / tick_size) * tick_size)


def _daily_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (policy, day), group in frame.groupby(["policy", "day"]):
        turnover = float(group["turnover"].sum())
        pnl = float(group["pnl"].sum())
        rows.append(
            {
                "policy": policy,
                "day": day,
                "pnl": pnl,
                "nd_pnl": float(group["nd_pnl"].sum()),
                "avg_abs_position": float(group["avg_abs_position"].mean()),
                "profit_ratio": pnl / max(turnover, 1e-8),
                "turnover": turnover,
                "episodes": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy", "day"])
