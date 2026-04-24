from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_partial_analysis(data_root: Path, output_root: Path, run_name: str, symbol: str) -> Path:
    base = data_root / run_name / symbol
    days = sorted([p for p in base.iterdir() if p.is_dir()])
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = output_root / run_name / f"partial_analysis_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for day in days:
        latent_path = day / "latent.csv"
        price_path = day / "price.csv"
        trades_path = day / "trades.csv"
        if not (latent_path.exists() and price_path.exists() and trades_path.exists()):
            continue
        latent = pd.read_csv(latent_path)
        price = pd.read_csv(price_path)
        trades = pd.read_csv(trades_path)
        if latent.empty or price.empty:
            continue

        spread_ticks = latent["spread_ticks"].astype(float)
        event_actor = latent.get("event_actor", pd.Series([""] * len(latent)))
        top_imb = latent.get("top_imbalance", pd.Series(np.zeros(len(latent))))
        queue_pressure = latent.get("queue_pressure", pd.Series(np.zeros(len(latent))))
        vol_state = latent.get("vol_state", pd.Series(np.ones(len(latent))))
        fair = latent.get("fair_value", pd.Series(price["midprice"].values[: len(latent)]))

        mid = price["midprice"].astype(float)
        ret = mid.diff().fillna(0.0)
        n_trades = len(trades)
        informed_trade_share = float((trades["taker_agent"] == "informed_taker").mean()) if n_trades > 0 and "taker_agent" in trades.columns else 0.0
        noise_trade_share = float((trades["taker_agent"] == "noise_taker").mean()) if n_trades > 0 and "taker_agent" in trades.columns else 0.0

        fair_series = pd.to_numeric(fair, errors="coerce").ffill().bfill()
        fair_series = fair_series.iloc[: len(mid)] if hasattr(fair_series, "iloc") else pd.Series(fair_series[: len(mid)])

        rows.append(
            {
                "day": day.name,
                "events": int(len(latent)),
                "trades": int(n_trades),
                "spread_mean": float(spread_ticks.mean()),
                "spread_std": float(spread_ticks.std(ddof=0)),
                "spread_gt1_frac": float((spread_ticks > 1.5).mean()),
                "informed_event_share": float((event_actor == "informed_taker").mean()),
                "noise_event_share": float((event_actor == "noise_taker").mean()),
                "maker_event_share": float((event_actor == "competing_mm").mean()),
                "informed_trade_share": informed_trade_share,
                "noise_trade_share": noise_trade_share,
                "top_imbalance_mean": float(pd.to_numeric(top_imb, errors="coerce").fillna(0).mean()),
                "queue_pressure_mean": float(pd.to_numeric(queue_pressure, errors="coerce").fillna(0).mean()),
                "vol_state_mean": float(pd.to_numeric(vol_state, errors="coerce").fillna(1).mean()),
                "fair_mid_abs_ticks": float(np.abs(fair_series.values - mid.values).mean() / 0.01),
                "rv_per_event": float(np.sqrt((ret**2).mean())),
            }
        )

    if not rows:
        raise RuntimeError("No day data available to analyze.")

    df = pd.DataFrame(rows).sort_values("day")
    df.to_csv(out / "day_metrics.csv", index=False)

    summary = {
        "days_available": int(len(df)),
        "events_total": int(df["events"].sum()),
        "trades_total": int(df["trades"].sum()),
        "spread_mean_avg": float(df["spread_mean"].mean()),
        "spread_gt1_frac_avg": float(df["spread_gt1_frac"].mean()),
        "informed_event_share_avg": float(df["informed_event_share"].mean()),
        "noise_event_share_avg": float(df["noise_event_share"].mean()),
        "maker_event_share_avg": float(df["maker_event_share"].mean()),
        "informed_trade_share_avg": float(df["informed_trade_share"].mean()),
        "rv_per_event_avg": float(df["rv_per_event"].mean()),
        "fair_mid_abs_ticks_avg": float(df["fair_mid_abs_ticks"].mean()),
    }
    pd.Series(summary).to_json(out / "quick_summary.json", indent=2)

    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["day"], df["spread_mean"], marker="o", label="spread_mean")
    ax.plot(df["day"], df["spread_gt1_frac"], marker="o", label="spread_gt1_frac")
    ax.set_title("Spread Diagnostics by Day")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "spread_by_day.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["day"], df["informed_event_share"], marker="o", label="informed_event_share")
    ax.plot(df["day"], df["noise_event_share"], marker="o", label="noise_event_share")
    ax.plot(df["day"], df["maker_event_share"], marker="o", label="maker_event_share")
    ax.set_title("Event Mix by Day")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "event_mix_by_day.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["day"], df["trades"], marker="o", label="trades")
    scale = max(df["trades"].max(), 1.0)
    ax.plot(df["day"], df["informed_trade_share"] * scale, marker="o", label="informed_trade_share_scaled")
    ax.set_title("Trades and Informed Share by Day")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "trades_by_day.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["day"], df["fair_mid_abs_ticks"], marker="o", label="fair_mid_abs_ticks")
    ax.plot(df["day"], df["rv_per_event"] * 1000, marker="o", label="rv_per_event_x1000")
    ax.set_title("Price Process Stability by Day")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "stability_by_day.png", dpi=180)
    plt.close(fig)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--symbol", default="000001")
    parser.add_argument("--data-root", default="/cluster/scratch/apetric/data")
    parser.add_argument("--output-root", default="/cluster/scratch/apetric/artifacts_anja")
    args = parser.parse_args()

    out = build_partial_analysis(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        run_name=args.run_name,
        symbol=args.symbol,
    )
    print(out)


if __name__ == "__main__":
    main()
