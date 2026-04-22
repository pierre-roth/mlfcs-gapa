"""
Simulator diagnostic — run before and after parameter changes to compare.

Usage:
    python diag_simulator.py
    python diag_simulator.py --symbol 000001 --events 3000
"""
import sys
import argparse
import numpy as np

from piroth.config import GenerateConfig, ReportConfig
from piroth.simulator import AgentBasedLOB, _symbol_profile, generate_dataset
from piroth.data import load_splits
from piroth.baselines import AvellanedaStoikovPolicy, calibrate_avellaneda_stoikov, FixedLevelPolicy
from piroth.env import ContinuousMarketEnv


def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


def run_baseline(policy, days, config):
    pnls = []
    for day in days:
        env = ContinuousMarketEnv(day, config)
        for i, span in enumerate(env.selected_episodes(config.max_eval_episodes_per_day)):
            env.set_eval_context(i)
            env.reset(span)
            done = False
            while not done:
                quote_idx = max(int(env.episode_decisions[env.cursor] - config.latency), config.lookback - 1)
                decision = policy.act(day, quote_idx, env.inventory, env.cursor, len(env.episode_decisions))
                _, _, done, _ = env.step({
                    "ask_price": decision.ask_price,
                    "ask_volume": decision.ask_volume,
                    "bid_price": decision.bid_price,
                    "bid_volume": decision.bid_volume,
                    "spread": decision.spread,
                    "reservation": 0.5 * (decision.ask_price + decision.bid_price),
                })
            result = env.episode_result(policy.name, i)
            pnls.append(result["pnl"])
    mean = float(np.mean(pnls)) if pnls else 0.0
    sharpe = float(np.mean(pnls) / max(np.std(pnls), 1e-9)) if len(pnls) > 1 else 0.0
    return mean, sharpe, len(pnls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="000001")
    parser.add_argument("--events", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    symbol = args.symbol

    # --- 1. Raw book diagnostics (no pipeline, just the simulator) ---
    section("1. Book & price diagnostics")

    cfg = GenerateConfig(seed=args.seed)
    cfg.apply_mode_defaults()
    cfg.events_per_day = {symbol: args.events}

    profile = _symbol_profile(symbol, cfg)
    rng = np.random.default_rng(args.seed)
    book = AgentBasedLOB(cfg, profile, rng)

    spreads = []
    mid_moves = []
    prev_mid = book.midprice
    touch_depths = []
    market_order_events = 0
    walk_through_events = 0  # market order consumed full touch and went deeper
    trades_at_touch = 0
    trades_beyond_touch = 0

    for step_i in range(args.events):
        if step_i % 100 == 0:
            print(f"  step {step_i}/{args.events}  book levels: bid={len(book.bids)} ask={len(book.asks)}  mid={book.midprice:.4f}", flush=True)
        touch_bid_before = book.best_bid
        touch_ask_before = book.best_ask
        touch_ask_depth_before = sum(o.size for o in book.asks.get(touch_ask_before, []))
        touch_bid_depth_before = sum(o.size for o in book.bids.get(touch_bid_before, []))

        prev_mid = book.midprice
        lat, msg, trades = book.step()

        spreads.append(book.spread_ticks)
        mid_moves.append(book.midprice - prev_mid)
        touch_depths.append((touch_ask_depth_before + touch_bid_depth_before) / 2)

        if trades:
            market_order_events += 1
            for t in trades:
                if t.price == touch_ask_before or t.price == touch_bid_before:
                    trades_at_touch += 1
                else:
                    trades_beyond_touch += 1
                    walk_through_events += 1

    spreads = np.array(spreads)
    mid_moves = np.array(mid_moves)
    touch_depths = np.array(touch_depths)

    print(f"Symbol: {symbol}  Events: {args.events}")
    print(f"\nSpread distribution:")
    print(f"  1 tick : {(spreads == 1).mean()*100:.1f}%")
    print(f"  2 ticks: {(spreads == 2).mean()*100:.1f}%")
    print(f"  3+ticks: {(spreads >= 3).mean()*100:.1f}%")
    print(f"  mean   : {spreads.mean():.2f} ticks")

    print(f"\nTouch depth (avg shares at best bid/ask): {touch_depths.mean():.0f}")

    print(f"\nMarket order events: {market_order_events} ({market_order_events/args.events*100:.1f}% of events)")
    print(f"Walk-through events (market order went past touch): {walk_through_events} ({walk_through_events/max(market_order_events,1)*100:.1f}% of market orders)")
    print(f"Trades at touch: {trades_at_touch}   Trades beyond touch: {trades_beyond_touch}")

    print(f"\nMid-price step variance: {np.var(mid_moves):.2e}  (std={mid_moves.std():.5f})")
    print(f"Mid-price drift over {args.events} events: {mid_moves.sum():.4f}  (start={profile.base_price:.2f})")

    # --- 2. Fill probability decay (what AS calibration uses) ---
    section("2. Fill probability decay")

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        gen_cfg = GenerateConfig(
            mode="smoke",
            data_dir=tmpdir,
            symbols=[symbol],
            seed=args.seed,
        )
        gen_cfg.apply_mode_defaults()
        gen_cfg.events_per_day = {symbol: args.events}
        generate_dataset(gen_cfg)

        rep_cfg = ReportConfig(
            mode="smoke",
            data_dir=tmpdir,
            symbols=[symbol],
            seed=args.seed,
        )
        rep_cfg.apply_mode_defaults()
        rep_cfg.events_per_day = {symbol: args.events}
        splits = load_splits(rep_cfg, symbol)

        from piroth.baselines import _fill_probability_at_distance
        tick = rep_cfg.tick_size
        print(f"\n  distance    fill probability")
        for ticks in range(5):
            prob = _fill_probability_at_distance(splits["train"], rep_cfg, ticks)
            bar = "█" * int(prob * 40)
            print(f"  {ticks} ticks    {prob*100:5.1f}%  {bar}")

        # --- 3. AS calibration results ---
        section("3. AS calibration")
        calibration = calibrate_avellaneda_stoikov(splits["train"], rep_cfg)
        print(f"  gamma (inventory risk aversion): {calibration.gamma:.4f}")
        print(f"  kappa (fill intensity):          {calibration.kappa:.2f}")
        print(f"  step variance:                   {calibration.step_variance:.2e}")
        print(f"  base spread:                     {calibration.base_spread/tick:.2f} ticks  ({calibration.base_spread:.4f})")

        # --- 4. Baseline PnL on test data ---
        section("4. Baseline PnL on test episodes")
        as_policy = AvellanedaStoikovPolicy(rep_cfg, calibration)
        fixed1 = FixedLevelPolicy(rep_cfg, 1)
        fixed2 = FixedLevelPolicy(rep_cfg, 2)

        for policy in [as_policy, fixed1, fixed2]:
            mean, sharpe, n = run_baseline(policy, splits["test"], rep_cfg)
            flag = " ✓ profitable" if mean > 0 else " ✗ losing"
            print(f"  {policy.name:<10}  pnl_mean={mean:+.5f}  sharpe={sharpe:+.3f}  episodes={n}{flag}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
