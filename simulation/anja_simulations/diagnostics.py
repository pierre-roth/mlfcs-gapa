"""Stylized-facts validator for the synthetic LOB output.

Loads one or more days of generated CSVs (ask/bid/price/trades) and computes
a handful of micro-structural diagnostics. The key metric for RL learnability
is `lob_imbalance_future_return_corr` — if it is ≈ 0, the book shape carries
no information about future price, and PPO with a LOB feature extractor
cannot beat AS.

CLI:

    python -m anja_simulations.diagnostics --data-dir <path> --symbol 000001 \
        --days 3 --horizons 10 50 200

Output: JSON printed to stdout (and optionally written via --output).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _log_returns(midprice: np.ndarray, horizon: int) -> np.ndarray:
    mid = np.asarray(midprice, dtype=np.float64)
    if horizon <= 0 or len(mid) <= horizon:
        return np.array([], dtype=np.float64)
    future = mid[horizon:]
    past = mid[:-horizon]
    mask = (past > 0) & (future > 0)
    return np.log(future[mask]) - np.log(past[mask])


def _kurtosis(x: np.ndarray) -> float:
    if x.size < 4:
        return float("nan")
    mean = float(x.mean())
    centered = x - mean
    m2 = float((centered ** 2).mean())
    m4 = float((centered ** 4).mean())
    if m2 <= 0.0:
        return float("nan")
    return m4 / (m2 ** 2)


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < max_lag + 2:
        return np.full(max_lag + 1, np.nan, dtype=np.float64)
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0:
        return np.full(max_lag + 1, 0.0, dtype=np.float64)
    out = np.empty(max_lag + 1, dtype=np.float64)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = float(np.dot(x[lag:], x[:-lag]) / (n - lag) / var)
    return out


def _trade_sign_series(trades: pd.DataFrame) -> np.ndarray:
    if trades.empty or "aggressor_side" not in trades.columns:
        return np.array([], dtype=np.float64)
    sides = trades["aggressor_side"].astype(str).str.upper().to_numpy()
    signs = np.where(sides == "B", 1.0, np.where(sides == "A", -1.0, 0.0))
    return signs


def _top_imbalance_series(ask: pd.DataFrame, bid: pd.DataFrame) -> np.ndarray:
    ask_vol = ask["ask1_volume"].to_numpy(dtype=np.float64)
    bid_vol = bid["bid1_volume"].to_numpy(dtype=np.float64)
    total = ask_vol + bid_vol
    return np.where(total > 0, (bid_vol - ask_vol) / total, 0.0)


def _load_day(root: Path) -> dict[str, object]:
    ask = pd.read_csv(root / "ask.csv")
    bid = pd.read_csv(root / "bid.csv")
    price = pd.read_csv(root / "price.csv", parse_dates=["timestamp"])
    trades_path = root / "trades.csv"
    trades = pd.read_csv(trades_path, parse_dates=["timestamp"]) if trades_path.exists() else pd.DataFrame()
    latent_path = root / "latent.csv"
    latent = pd.read_csv(latent_path) if latent_path.exists() else pd.DataFrame()
    return {"ask": ask, "bid": bid, "price": price, "trades": trades, "latent": latent}


def _depth_top3(ask: pd.DataFrame, bid: pd.DataFrame) -> np.ndarray:
    cols_ask = [f"ask{i}_volume" for i in (1, 2, 3)]
    cols_bid = [f"bid{i}_volume" for i in (1, 2, 3)]
    a = ask[cols_ask].to_numpy(dtype=np.float64).sum(axis=1)
    b = bid[cols_bid].to_numpy(dtype=np.float64).sum(axis=1)
    return a + b


def _inter_event_seconds(timestamps: pd.Series) -> np.ndarray:
    diffs = np.diff(pd.to_datetime(timestamps).astype("int64").to_numpy()) / 1e9
    diffs = diffs[diffs > 0]
    return diffs


def diagnose_day(root: Path, horizons: list[int], max_acf_lag: int) -> dict[str, object]:
    day = _load_day(root)
    ask, bid, price, trades = day["ask"], day["bid"], day["price"], day["trades"]
    midprice = price["midprice"].to_numpy(dtype=np.float64)
    ask1 = price["ask1_price"].to_numpy(dtype=np.float64)
    bid1 = price["bid1_price"].to_numpy(dtype=np.float64)
    spread = ask1 - bid1

    out: dict[str, object] = {"day": root.name, "events": int(len(price))}

    # 1. Kurtosis at multiple horizons
    kurts: dict[str, float] = {}
    for h in (1, 10, 100):
        rets = _log_returns(midprice, h)
        kurts[str(h)] = _kurtosis(rets)
    out["log_return_kurtosis"] = kurts

    # 2. Trade-sign ACF
    signs = _trade_sign_series(trades)
    if signs.size > max_acf_lag + 2:
        acf = _autocorr(signs, max_acf_lag)
        out["trade_sign_acf_lag1"] = float(acf[1])
        out["trade_sign_acf_lag10"] = float(acf[10]) if max_acf_lag >= 10 else None
        out["trade_sign_acf_lag50"] = float(acf[50]) if max_acf_lag >= 50 else None
    else:
        out["trade_sign_acf_lag1"] = None

    # 3. |return| ACF — volatility clustering
    rets1 = _log_returns(midprice, 1)
    if rets1.size > max_acf_lag + 2:
        abs_acf = _autocorr(np.abs(rets1), max_acf_lag)
        out["abs_return_acf_lag1"] = float(abs_acf[1])
        out["abs_return_acf_lag10"] = float(abs_acf[10]) if max_acf_lag >= 10 else None
        out["abs_return_acf_lag50"] = float(abs_acf[50]) if max_acf_lag >= 50 else None

    # 4. Spread–depth correlation
    depth = _depth_top3(ask, bid)
    if len(spread) == len(depth) and len(spread) > 2:
        out["spread_depth_corr"] = float(np.corrcoef(spread, depth)[0, 1])

    # 5. Inter-event duration stats
    durations = _inter_event_seconds(price["timestamp"])
    if durations.size:
        mean = float(durations.mean())
        var = float(durations.var())
        out["inter_event_mean_s"] = mean
        out["inter_event_cv2"] = var / (mean ** 2) if mean > 0 else None

    # 6. THE KEY METRIC: LOB imbalance → future mid return correlation
    imbalance = _top_imbalance_series(ask, bid)
    fut_corrs: dict[str, float] = {}
    for h in horizons:
        if len(imbalance) <= h + 1:
            fut_corrs[str(h)] = float("nan")
            continue
        imb = imbalance[:-h]
        fut_ret = np.log(np.clip(midprice[h:], 1e-9, None)) - np.log(np.clip(midprice[:-h], 1e-9, None))
        if imb.std() <= 0 or fut_ret.std() <= 0:
            fut_corrs[str(h)] = float("nan")
        else:
            fut_corrs[str(h)] = float(np.corrcoef(imb, fut_ret)[0, 1])
    out["lob_imbalance_future_return_corr"] = fut_corrs

    # 7. LOB imbalance → next trade direction
    if signs.size and len(trades) > 0 and "timestamp" in trades.columns:
        trade_ts = pd.to_datetime(trades["timestamp"]).astype("int64").to_numpy()
        price_ts = pd.to_datetime(price["timestamp"]).astype("int64").to_numpy()
        idx = np.searchsorted(price_ts, trade_ts, side="right") - 1
        idx = np.clip(idx, 0, len(imbalance) - 1)
        imb_before_trade = imbalance[idx]
        if imb_before_trade.std() > 0 and np.std(signs) > 0:
            out["lob_imbalance_next_trade_corr"] = float(
                np.corrcoef(imb_before_trade, signs)[0, 1]
            )

    # 8. Informed-vs-noise trade shares (simulator-specific diagnostic)
    if "taker_agent" in trades.columns and len(trades) > 0:
        taker = trades["taker_agent"].astype(str)
        out["informed_trade_share"] = float((taker == "informed_taker").mean())
        out["noise_trade_share"] = float((taker == "noise_taker").mean())

    return out


def _pass_fail(report: dict[str, object]) -> dict[str, str]:
    checks: dict[str, str] = {}
    # LOB-imbalance↔future-return correlation at horizon 50: target ≥ 0.1
    fut = report.get("lob_imbalance_future_return_corr", {})
    if isinstance(fut, dict) and "50" in fut:
        val = fut["50"]
        if val is None or np.isnan(val):
            checks["lob_imbalance_h50"] = "UNAVAILABLE"
        elif val >= 0.1:
            checks["lob_imbalance_h50"] = f"PASS ({val:+.3f})"
        elif val >= 0.03:
            checks["lob_imbalance_h50"] = f"WEAK ({val:+.3f})"
        else:
            checks["lob_imbalance_h50"] = f"FAIL ({val:+.3f}) — Attn-LOB will have no signal"

    # Return kurtosis: want >3 at 1-event scale (heavier tails than Gaussian)
    kurt = report.get("log_return_kurtosis", {}).get("1")
    if kurt is not None and not np.isnan(kurt):
        checks["return_kurtosis_h1"] = f"PASS ({kurt:.2f})" if kurt > 3.0 else f"FAIL ({kurt:.2f}) — too Gaussian"

    # Trade-sign ACF lag 10: want > 0 (flow persistence)
    acf = report.get("trade_sign_acf_lag10")
    if acf is not None:
        checks["trade_sign_acf_lag10"] = f"PASS ({acf:+.3f})" if acf > 0.02 else f"WEAK ({acf:+.3f})"

    # Volatility clustering: |return| ACF at lag 10 > 0.05
    abs_acf = report.get("abs_return_acf_lag10")
    if abs_acf is not None:
        checks["vol_clustering_lag10"] = f"PASS ({abs_acf:+.3f})" if abs_acf > 0.05 else f"WEAK ({abs_acf:+.3f})"

    # Spread-depth: want negative correlation
    sd = report.get("spread_depth_corr")
    if sd is not None:
        checks["spread_depth_inverse"] = f"PASS ({sd:+.3f})" if sd < -0.05 else f"FAIL ({sd:+.3f})"

    return checks


def _aggregate(day_reports: list[dict[str, object]]) -> dict[str, object]:
    """Average numeric fields across days. Dicts of horizons are merged per-key."""
    if not day_reports:
        return {}
    agg: dict[str, object] = {"n_days": len(day_reports)}
    numeric_keys = [
        "events",
        "trade_sign_acf_lag1", "trade_sign_acf_lag10", "trade_sign_acf_lag50",
        "abs_return_acf_lag1", "abs_return_acf_lag10", "abs_return_acf_lag50",
        "spread_depth_corr", "inter_event_mean_s", "inter_event_cv2",
        "lob_imbalance_next_trade_corr",
        "informed_trade_share", "noise_trade_share",
    ]
    for key in numeric_keys:
        vals = [r.get(key) for r in day_reports if isinstance(r.get(key), (int, float)) and not (isinstance(r.get(key), float) and np.isnan(r.get(key)))]
        if vals:
            agg[key] = float(np.mean(vals))
    # Nested dicts: log_return_kurtosis, lob_imbalance_future_return_corr
    for outer in ("log_return_kurtosis", "lob_imbalance_future_return_corr"):
        horizon_vals: dict[str, list[float]] = {}
        for r in day_reports:
            d = r.get(outer)
            if isinstance(d, dict):
                for h, v in d.items():
                    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                        horizon_vals.setdefault(h, []).append(float(v))
        if horizon_vals:
            agg[outer] = {h: float(np.mean(vs)) for h, vs in horizon_vals.items()}
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", required=True, help="Path to the generated data root (contains <symbol>/<day>/*.csv)")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--days", type=int, default=0, help="Number of days to scan (0 = all found)")
    parser.add_argument("--horizons", type=int, nargs="*", default=[10, 50, 200])
    parser.add_argument("--max-acf-lag", type=int, default=50)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    symbol_root = Path(args.data_dir) / args.symbol
    if not symbol_root.exists():
        raise SystemExit(f"No such directory: {symbol_root}")
    day_dirs = sorted(p for p in symbol_root.iterdir() if p.is_dir())
    if args.days > 0:
        day_dirs = day_dirs[: args.days]
    if not day_dirs:
        raise SystemExit(f"No day directories under {symbol_root}")

    per_day = [diagnose_day(d, args.horizons, args.max_acf_lag) for d in day_dirs]
    aggregate = _aggregate(per_day)
    aggregate["checks"] = _pass_fail(aggregate)
    report = {"symbol": args.symbol, "data_dir": str(args.data_dir), "aggregate": aggregate, "per_day": per_day}

    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
