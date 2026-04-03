"""
Compare GOOGL vs AAPL microstructure:
- typical spread in bps
- trade rate (fraction of events that are trades)
- mid-price range
- fill rate with random policy under lobmm config
"""
import sys
import numpy as np
from lobmm.config import ExperimentConfig
from lobmm.data import discover_days, load_day_data

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
symbols = sys.argv[2:] if len(sys.argv) > 2 else ["AAPL", "GOOGL"]

config = ExperimentConfig(data_dir=data_dir, mode="smoke", train_days=6, val_days=2, test_days=2)
config.apply_mode_defaults()

for symbol in symbols:
    days = discover_days(data_dir, symbol)
    spreads_bps = []
    trade_rates = []
    mid_prices = []

    for day in days[:5]:
        d = load_day_data(symbol, day, config)
        mid = d.midprice
        ask1 = d.ask1
        bid1 = d.bid1
        spread_bps = 1e4 * (ask1 - bid1) / np.maximum(mid, 1e-8)
        spreads_bps.extend(spread_bps.tolist())
        mid_prices.extend(mid.tolist())
        n_trades = sum(len(v.price) for v in d.trades_by_index.values())
        trade_rates.append(n_trades / max(len(mid), 1))

    print(f"\n{'='*50}")
    print(f"Symbol: {symbol}  ({len(days)} days available, analyzed {min(5,len(days))})")
    print(f"  Mid price:   mean=${np.mean(mid_prices):.2f}  range=[${np.min(mid_prices):.2f}, ${np.max(mid_prices):.2f}]")
    print(f"  Spread:      mean={np.mean(spreads_bps):.3f} bps  median={np.median(spreads_bps):.3f} bps  p95={np.percentile(spreads_bps,95):.3f} bps")
    print(f"  Trade rate:  mean={np.mean(trade_rates)*100:.2f}% of events are trades")
    print(f"  Quote reach: max_bias=2bps max_spread=8bps → agent can quote {2+4:.1f} bps from mid")
    print(f"  Fill likely? {'YES' if np.mean(spreads_bps) < 6 else 'MARGINAL' if np.mean(spreads_bps) < 10 else 'NO - spread too wide'}")
