from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < max_lag + 2:
        return np.full(max_lag + 1, np.nan)
    x = x - x.mean()
    var = np.dot(x, x) / x.size
    if var <= 0:
        return np.zeros(max_lag + 1)
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = np.dot(x[lag:], x[:-lag]) / (x.size - lag) / var
    return out


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _numeric_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out: dict[str, float] = {}
            for key, val in row.items():
                try:
                    out[key] = float(val)
                except (TypeError, ValueError):
                    pass
            rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--symbol", default="000001")
    parser.add_argument("--days", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_dir) / args.symbol
    artifacts = Path(args.artifacts_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    day_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if args.days > 0:
        day_dirs = day_dirs[: args.days]
    if not day_dirs:
        raise SystemExit(f"No day directories under {data_root}")

    price_paths = []
    latent_paths = []
    trade_paths = []
    daily = []
    event_mix = Counter()
    actor_mix = Counter()
    all_returns = []
    all_abs_returns = []
    all_spreads = []
    all_imbalance = []
    all_fut_ret_50 = []
    all_trade_signs = []
    label_counts_by_split: dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}

    for day_index, day_dir in enumerate(day_dirs):
        price = pd.read_csv(day_dir / "price.csv")
        latent = pd.read_csv(day_dir / "latent.csv")
        ask = pd.read_csv(day_dir / "ask.csv", usecols=["ask1_volume"])
        bid = pd.read_csv(day_dir / "bid.csv", usecols=["bid1_volume"])
        trades_path = day_dir / "trades.csv"
        trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
        price_paths.append(price)
        latent_paths.append(latent)
        trade_paths.append(trades)

        mid = price["midprice"].to_numpy(dtype=np.float64)
        returns = np.diff(np.log(np.clip(mid, 1e-9, None)))
        spread_ticks = latent["spread_ticks"].to_numpy(dtype=np.float64)
        ask_vol = ask["ask1_volume"].to_numpy(dtype=np.float64)
        bid_vol = bid["bid1_volume"].to_numpy(dtype=np.float64)
        total_vol = ask_vol + bid_vol
        imbalance = np.where(total_vol > 0, (bid_vol - ask_vol) / total_vol, 0.0)
        h = 50
        if mid.size > h:
            fut_ret = np.log(np.clip(mid[h:], 1e-9, None)) - np.log(np.clip(mid[:-h], 1e-9, None))
            all_imbalance.append(imbalance[:-h])
            all_fut_ret_50.append(fut_ret)
        alpha = 1e-5
        label_h = 10
        if mid.size > label_h:
            label_ret = np.log(np.clip(mid[label_h:], 1e-9, None)) - np.log(np.clip(mid[:-label_h], 1e-9, None))
            labels = np.where(label_ret > alpha, "up", np.where(label_ret < -alpha, "down", "flat"))
            split = "train" if day_index < 8 else "val" if day_index < 10 else "test"
            label_counts_by_split[split].update(labels.tolist())
        all_returns.append(returns)
        all_abs_returns.append(np.abs(returns))
        all_spreads.append(spread_ticks)
        if "event_type" in latent:
            event_mix.update(latent["event_type"].astype(str).tolist())
        if "event_actor" in latent:
            actor_mix.update(latent["event_actor"].astype(str).tolist())
        if not trades.empty and "aggressor_side" in trades:
            sides = trades["aggressor_side"].astype(str).str.upper().to_numpy()
            signs = np.where(sides == "B", 1.0, np.where(sides == "A", -1.0, 0.0))
            all_trade_signs.append(signs)
        daily.append(
            {
                "day": day_dir.name,
                "events": int(len(price)),
                "mid_start": float(mid[0]),
                "mid_end": float(mid[-1]),
                "log_return_sum": float(np.log(mid[-1] / mid[0])),
                "spread_mean": float(np.mean(spread_ticks)),
                "spread_gt1_frac": float(np.mean(spread_ticks > 1.0)),
                "trades": int(len(trades)),
                "top_imbalance_mean": float(np.mean(imbalance)),
                "top_imbalance_std": float(np.std(imbalance)),
            }
        )

    returns_all = np.concatenate(all_returns)
    abs_returns_all = np.concatenate(all_abs_returns)
    spreads_all = np.concatenate(all_spreads)
    imbalance_all = np.concatenate(all_imbalance)
    fut_ret_50_all = np.concatenate(all_fut_ret_50)
    trade_signs_all = np.concatenate(all_trade_signs) if all_trade_signs else np.array([])

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11, 5))
    for day_dir, price in zip(day_dirs[:8], price_paths[:8]):
        mid = price["midprice"].to_numpy(dtype=np.float64)
        step = max(1, len(mid) // 4000)
        ax.plot(np.arange(0, len(mid), step), mid[::step] / mid[0] - 1.0, lw=1.1, label=day_dir.name)
    ax.set_title("Normalized intraday midprice paths")
    ax.set_xlabel("event index")
    ax.set_ylabel("mid / open - 1")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "midprice_paths.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(spreads_all, bins=np.arange(0.5, max(5, spreads_all.max() + 1.5), 1), color="#4062bb", alpha=0.85)
    axes[0].set_title("Spread distribution")
    axes[0].set_xlabel("spread ticks")
    axes[0].set_ylabel("events")
    axes[1].hist(returns_all * 1e4, bins=120, color="#2a9d8f", alpha=0.85)
    axes[1].set_title("One-event log returns")
    axes[1].set_xlabel("return x 1e4")
    fig.tight_layout()
    fig.savefig(output / "spread_return_distributions.png", dpi=160)
    plt.close(fig)

    sample_step = max(1, len(imbalance_all) // 120_000)
    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(
        imbalance_all[::sample_step],
        fut_ret_50_all[::sample_step] * 1e4,
        gridsize=45,
        mincnt=1,
        cmap="viridis",
    )
    ax.set_title("Top imbalance vs future 50-event return")
    ax.set_xlabel("(bid1_volume - ask1_volume) / total")
    ax.set_ylabel("future log return x 1e4")
    fig.colorbar(hb, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(output / "imbalance_future_return_h50.png", dpi=160)
    plt.close(fig)

    def _bar(counter: Counter, path: Path, title: str) -> None:
        items = counter.most_common()
        labels = [k for k, _ in items]
        vals = np.array([v for _, v in items], dtype=np.float64)
        vals = vals / vals.sum()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(labels, vals, color="#9a6b4f")
        ax.set_title(title)
        ax.set_ylabel("share")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    _bar(event_mix, output / "event_type_mix.png", "Event type mix")
    _bar(actor_mix, output / "event_actor_mix.png", "Event actor mix")

    max_lag = 50
    ret_acf = _acf(abs_returns_all, max_lag)
    sign_acf = _acf(trade_signs_all, max_lag) if trade_signs_all.size else np.full(max_lag + 1, np.nan)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(max_lag + 1), ret_acf, label="abs return ACF", lw=2)
    ax.plot(np.arange(max_lag + 1), sign_acf, label="trade sign ACF", lw=2)
    ax.set_title("Persistence diagnostics")
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "acf_diagnostics.png", dpi=160)
    plt.close(fig)

    ppo_rows = _numeric_rows(artifacts / args.symbol / "ppo" / "episodes.csv")
    if ppo_rows:
        pnl = np.array([r.get("pnl", np.nan) for r in ppo_rows], dtype=np.float64)
        trades = np.array([r.get("trades", np.nan) for r in ppo_rows], dtype=np.float64)
        pos = np.array([r.get("avg_abs_position", np.nan) for r in ppo_rows], dtype=np.float64)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].hist(pnl[~np.isnan(pnl)], bins=40, color="#264653")
        axes[0].set_title("PPO episode PnL")
        axes[1].hist(trades[~np.isnan(trades)], bins=30, color="#e76f51")
        axes[1].set_title("PPO trades per episode")
        axes[2].hist(pos[~np.isnan(pos)], bins=30, color="#6a994e")
        axes[2].set_title("PPO avg abs inventory")
        fig.tight_layout()
        fig.savefig(output / "ppo_episode_distributions.png", dpi=160)
        plt.close(fig)

    summary = {
        "n_days": len(day_dirs),
        "daily": daily,
        "spread_mean": float(np.mean(spreads_all)),
        "spread_gt1_frac": float(np.mean(spreads_all > 1.0)),
        "return_std_1event": float(np.std(returns_all)),
        "return_kurtosis_proxy": float(np.mean((returns_all - returns_all.mean()) ** 4) / (np.var(returns_all) ** 2)),
        "imbalance_future_return_corr_50": float(np.corrcoef(imbalance_all, fut_ret_50_all)[0, 1]),
        "trade_sign_acf_lag10": float(sign_acf[10]) if trade_signs_all.size else None,
        "abs_return_acf_lag10": float(ret_acf[10]),
        "event_mix": dict(event_mix),
        "actor_mix": dict(actor_mix),
        "label_counts_h10_alpha1e-5": {split: dict(counts) for split, counts in label_counts_by_split.items()},
    }
    (output / "plot_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
