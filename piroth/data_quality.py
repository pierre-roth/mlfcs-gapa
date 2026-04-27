from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .config import DiagnosticsConfig
from .simulator import SyntheticDay


@dataclass(frozen=True)
class SyntheticQualitySummary:
    score: float
    events_per_day_mean: float
    trades_per_event: float
    spread_mean_ticks: float
    spread_p95_ticks: float
    event_return_std_bp: float
    window_abs_move_p50_ticks: float
    window_abs_move_p90_ticks: float
    order_flow_autocorr: float
    depth_imbalance_std: float
    flags: list[str]


def assess_synthetic_quality(days: list[SyntheticDay], config: DiagnosticsConfig) -> dict[str, object]:
    if not days:
        return asdict(SyntheticQualitySummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ["no days"]))
    price = pd.concat([day.price for day in days], ignore_index=True)
    msg = pd.concat([day.msg for day in days], ignore_index=True)
    trades_per_event = sum(len(day.trades) for day in days) / max(sum(len(day.price) for day in days), 1)
    window_moves = _window_moves(days, config)
    signed_flow = (msg.get("market_buy_volume", 0) - msg.get("market_sell_volume", 0)).to_numpy(dtype=np.float64)
    depth_imbalance = _depth_imbalance(days)
    summary = SyntheticQualitySummary(
        score=0.0,
        events_per_day_mean=float(np.mean([len(day.price) for day in days])),
        trades_per_event=float(trades_per_event),
        spread_mean_ticks=float(price["spread_ticks"].mean()),
        spread_p95_ticks=float(price["spread_ticks"].quantile(0.95)),
        event_return_std_bp=float(price["return_bp"].std(ddof=0)),
        window_abs_move_p50_ticks=float(np.quantile(np.abs(window_moves), 0.50)) if window_moves.size else 0.0,
        window_abs_move_p90_ticks=float(np.quantile(np.abs(window_moves), 0.90)) if window_moves.size else 0.0,
        order_flow_autocorr=_autocorr(signed_flow),
        depth_imbalance_std=float(np.std(depth_imbalance)) if depth_imbalance.size else 0.0,
        flags=[],
    )
    flags = _quality_flags(summary)
    score = _quality_score(summary, flags)
    return asdict(
        SyntheticQualitySummary(
            score=score,
            events_per_day_mean=summary.events_per_day_mean,
            trades_per_event=summary.trades_per_event,
            spread_mean_ticks=summary.spread_mean_ticks,
            spread_p95_ticks=summary.spread_p95_ticks,
            event_return_std_bp=summary.event_return_std_bp,
            window_abs_move_p50_ticks=summary.window_abs_move_p50_ticks,
            window_abs_move_p90_ticks=summary.window_abs_move_p90_ticks,
            order_flow_autocorr=summary.order_flow_autocorr,
            depth_imbalance_std=summary.depth_imbalance_std,
            flags=flags,
        )
    )


def _quality_flags(summary: SyntheticQualitySummary) -> list[str]:
    flags = []
    if summary.trades_per_event < 0.15:
        flags.append("too few trades per event for paper-style matching")
    if summary.trades_per_event > 1.25:
        flags.append("very high trade density; fills may be too easy")
    if summary.spread_mean_ticks < 1.0:
        flags.append("spread is unrealistically pinned at one tick")
    if summary.spread_p95_ticks > 8.0:
        flags.append("spread tail is too wide for liquid top-10 LOB data")
    if summary.window_abs_move_p50_ticks < 2.0:
        flags.append("median 2000-event window is too flat")
    if summary.window_abs_move_p90_ticks > 80.0:
        flags.append("2000-event windows include overly directional moves")
    if summary.order_flow_autocorr < 0.02:
        flags.append("market order flow has little persistence")
    if summary.depth_imbalance_std < 0.03:
        flags.append("top-of-book depth imbalance is too static")
    return flags


def _quality_score(summary: SyntheticQualitySummary, flags: list[str]) -> float:
    score = 100.0
    score -= 9.0 * len(flags)
    score -= _distance_penalty(summary.spread_mean_ticks, 1.2, 3.0, 5.0)
    score -= _distance_penalty(summary.window_abs_move_p50_ticks, 4.0, 30.0, 10.0)
    score -= _distance_penalty(summary.window_abs_move_p90_ticks, 12.0, 75.0, 12.0)
    score -= _distance_penalty(summary.trades_per_event, 0.25, 1.05, 8.0)
    return float(np.clip(score, 0.0, 100.0))


def _distance_penalty(value: float, lower: float, upper: float, scale: float) -> float:
    if lower <= value <= upper:
        return 0.0
    if value < lower:
        return scale * (lower - value) / max(lower, 1e-8)
    return scale * (value - upper) / max(upper, 1e-8)


def _window_moves(days: list[SyntheticDay], config: DiagnosticsConfig) -> np.ndarray:
    moves = []
    for day in days:
        clock = day.price["timestamp"].dt.strftime("%H:%M:%S")
        mask = np.zeros(len(clock), dtype=bool)
        for raw in config.stable_windows:
            start, end = raw.split("-", maxsplit=1)
            mask |= (clock >= start) & (clock <= end)
        idx = np.flatnonzero(mask)
        for offset in range(0, len(idx), config.episode_length):
            window = idx[offset : offset + config.episode_length]
            if len(window) == config.episode_length:
                start_mid = float(day.price.iloc[int(window[0])]["midprice"])
                end_mid = float(day.price.iloc[int(window[-1])]["midprice"])
                moves.append((end_mid - start_mid) / config.symbol_spec.tick_size)
    return np.asarray(moves, dtype=np.float64)


def _depth_imbalance(days: list[SyntheticDay]) -> np.ndarray:
    values = []
    for day in days:
        bid = day.bid["bid1_volume"].to_numpy(dtype=np.float64)
        ask = day.ask["ask1_volume"].to_numpy(dtype=np.float64)
        values.extend(((bid - ask) / np.clip(bid + ask, 1.0, None)).tolist())
    return np.asarray(values, dtype=np.float64)


def _autocorr(values: np.ndarray) -> float:
    if values.size < 3 or np.std(values) == 0:
        return 0.0
    x = values[:-1]
    y = values[1:]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
