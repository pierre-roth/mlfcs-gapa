from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan pretraining label balance across thresholds.")
    parser.add_argument("--real-data-root", default="/cluster/work/math/piroth/mlfcs-gapa/data/processed")
    parser.add_argument("--symbols", default="AAPL,GOOGL")
    parser.add_argument("--thresholds", default="0,0.0000025,0.000005,0.00001,0.00002,0.00005,0.0001")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=50)
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--test-days", type=int, default=4)
    parser.add_argument("--start-time", default="09:30:00")
    parser.add_argument("--end-time", default="16:00:00")
    parser.add_argument("--stable-windows", default="10:00:00-11:30:00,13:00:00-14:30:00")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    windows = [tuple(item.split("-", maxsplit=1)) for item in args.stable_windows.split(",") if item.strip()]
    rows = []
    for symbol in symbols:
        symbol_root = Path(args.real_data_root) / symbol
        days = [path.name for path in sorted(symbol_root.iterdir()) if path.is_dir() and (path / "price.csv").exists()]
        splits = {
            "train": days[: args.train_days],
            "eval": days[args.train_days : args.train_days + args.test_days],
        }
        for split, split_days in splits.items():
            prices = []
            for day in split_days:
                print(f"reading {symbol} {split} {day}", flush=True)
                prices.append(_read_price(symbol_root / day / "price.csv", args, windows))
            for threshold in thresholds:
                counts = np.zeros(3, dtype=np.int64)
                samples = 0
                for midprice in prices:
                    labels = _midprice_direction_labels(midprice, args.horizon, threshold)
                    labels = labels.iloc[args.lookback:].dropna().astype(int)
                    counts += np.bincount(labels.to_numpy(), minlength=3)
                    samples += int(labels.size)
                fractions = counts / max(int(counts.sum()), 1)
                rows.append(
                    {
                        "symbol": symbol,
                        "split": split,
                        "threshold": threshold,
                        "samples": samples,
                        "up": int(counts[0]),
                        "stationary": int(counts[1]),
                        "down": int(counts[2]),
                        "up_frac": float(fractions[0]),
                        "stationary_frac": float(fractions[1]),
                        "down_frac": float(fractions[2]),
                        "minority_frac": float(fractions.min()),
                        "majority_frac": float(fractions.max()),
                    }
                )

    frame = pd.DataFrame(rows)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".json":
            output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        else:
            frame.to_csv(output, index=False)
    print(frame.to_csv(index=False), end="")


def _read_price(path: Path, args, windows: list[tuple[str, str]]) -> pd.Series:
    chunks = []
    usecols = ["timestamp", "midprice"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunk_size, parse_dates=["timestamp"]):
        if not chunk.empty and chunk["timestamp"].iloc[0].strftime("%H:%M:%S") > args.end_time:
            break
        clock = chunk["timestamp"].dt.strftime("%H:%M:%S")
        mask = (clock >= args.start_time) & (clock <= args.end_time)
        stable = np.zeros(len(chunk), dtype=bool)
        for start, end in windows:
            stable |= (clock >= start) & (clock <= end)
        chunk = chunk.loc[mask & stable, "midprice"]
        if not chunk.empty:
            chunks.append(chunk.reset_index(drop=True))
    if not chunks:
        return pd.Series(dtype="float64")
    return pd.concat(chunks, ignore_index=True)


def _midprice_direction_labels(midprice: pd.Series, horizon: int, threshold: float) -> pd.Series:
    past = midprice.rolling(window=horizon, min_periods=horizon).mean()
    future = past.shift(-horizon)
    pct_change = (future - past) / past.clip(lower=1e-8)
    labels = pd.Series(np.nan, index=midprice.index, dtype="float64")
    labels[pct_change >= threshold] = 0
    labels[(pct_change < threshold) & (pct_change > -threshold)] = 1
    labels[pct_change <= -threshold] = 2
    return labels


if __name__ == "__main__":
    main()
