from __future__ import annotations

import argparse

import numpy as np

from lobmm.config import ExperimentConfig
from lobmm.data import discover_days, load_day_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare simple LOB microstructure statistics across symbols.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "GOOGL"])
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--mode", default="smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        data_dir=args.data_dir,
        mode=args.mode,
        train_days=max(args.days, 1),
        val_days=1,
        test_days=1,
    ).apply_mode_defaults()

    for symbol in args.symbols:
        days = discover_days(args.data_dir, symbol)
        spreads_bps: list[float] = []
        trade_rates: list[float] = []
        mid_prices: list[float] = []

        for day in days[: args.days]:
            day_data = load_day_data(symbol, day, config)
            mid = day_data.midprice
            spread_bps = 1e4 * (day_data.ask1 - day_data.bid1) / np.maximum(mid, 1e-8)
            spreads_bps.extend(spread_bps.tolist())
            mid_prices.extend(mid.tolist())
            n_trades = sum(len(batch.price) for batch in day_data.trades_by_index.values())
            trade_rates.append(n_trades / max(len(mid), 1))

        print(f"\n{'=' * 50}")
        print(f"Symbol: {symbol} ({len(days)} days available, analyzed {min(args.days, len(days))})")
        print(f"  Mid price:   mean=${np.mean(mid_prices):.2f} range=[${np.min(mid_prices):.2f}, ${np.max(mid_prices):.2f}]")
        print(
            "  Spread:      "
            f"mean={np.mean(spreads_bps):.3f} bps "
            f"median={np.median(spreads_bps):.3f} bps "
            f"p95={np.percentile(spreads_bps, 95):.3f} bps"
        )
        print(f"  Trade rate:  mean={np.mean(trade_rates) * 100:.2f}% of events are trades")


if __name__ == "__main__":
    main()
