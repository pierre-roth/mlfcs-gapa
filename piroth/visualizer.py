from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DiagnosticsConfig
from .data_quality import assess_synthetic_quality
from .paper_features import MSG_COLUMNS
from .simulator import SyntheticDay


def build_synthetic_data_report(days: list[SyntheticDay], config: DiagnosticsConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "assets"
    assets.mkdir(exist_ok=True)
    sections = [
        ("Market Overview", _plot_market_overview(days, assets / "market_overview.png")),
        ("2000-Event Episodes", _plot_episode_gallery(days, config, assets / "episode_gallery.png")),
        ("Spread And Returns", _plot_spread_return_distribution(days, assets / "spread_returns.png")),
        ("Order Flow", _plot_order_flow(days, assets / "order_flow.png")),
        ("LOB Heatmap", _plot_depth_heatmap(days[0], config, assets / "depth_heatmap.png")),
        ("LOB Snapshots", _plot_depth_snapshots(days[0], assets / "depth_snapshots.png")),
        ("Fill Curves", _plot_fill_curves(days, config, assets / "fill_curves.png")),
    ]
    baseline_html = _baseline_tables(output_dir.parent)
    summary_html = _summary_cards(days, config)
    html = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Synthetic LOB Report</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:0;background:#f8fafc;color:#0f172a}",
            "header{padding:28px 36px;background:#0f172a;color:white}",
            "main{max-width:1240px;margin:0 auto;padding:28px 24px 48px}",
            "section{margin:0 0 30px}",
            "h1{font-size:28px;margin:0 0 8px}",
            "h2{font-size:20px;margin:0 0 12px}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:18px 0 28px}",
            ".metric{background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px}",
            ".metric b{display:block;font-size:12px;color:#475569;text-transform:uppercase}",
            ".metric span{font-size:22px;font-weight:650}",
            "img{width:100%;background:white;border:1px solid #e2e8f0;border-radius:8px}",
            "table{border-collapse:collapse;background:white;border:1px solid #e2e8f0;width:100%;font-size:13px}",
            "td,th{padding:8px;border-bottom:1px solid #e2e8f0;text-align:right}",
            "th:first-child,td:first-child{text-align:left}",
            "</style></head><body>",
            f"<header><h1>Synthetic LOB Report: {config.symbol}</h1><div>Paper replication diagnostics for synthetic market-making data</div></header>",
            "<main>",
            summary_html,
            *[f"<section><h2>{title}</h2><img src='{path.relative_to(output_dir)}'></section>" for title, path in sections],
            baseline_html,
            "</main></body></html>",
        ]
    )
    report_path = output_dir / "index.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def load_exported_days(root: Path, symbol: str) -> list[SyntheticDay]:
    days = []
    for day_dir in sorted((root / symbol).iterdir()):
        if not day_dir.is_dir():
            continue
        ask = pd.read_csv(day_dir / "ask.csv", parse_dates=["timestamp"])
        bid = pd.read_csv(day_dir / "bid.csv", parse_dates=["timestamp"])
        price = pd.read_csv(day_dir / "price.csv", parse_dates=["timestamp"])
        trades = pd.read_csv(day_dir / "trades.csv", parse_dates=["timestamp"]) if (day_dir / "trades.csv").exists() else pd.DataFrame()
        msg = pd.read_csv(day_dir / "msg.csv", parse_dates=["timestamp"])
        latent = pd.read_csv(day_dir / "latent.csv", parse_dates=["timestamp"]) if (day_dir / "latent.csv").exists() else pd.DataFrame()
        event_log = pd.read_csv(day_dir / "event_log.csv", parse_dates=["timestamp"]) if (day_dir / "event_log.csv").exists() else pd.DataFrame()
        days.append(SyntheticDay(symbol=symbol, day=day_dir.name, ask=ask, bid=bid, price=price, trades=trades, msg=msg, event_log=event_log, latent=latent, depth_cube=_depth_cube_from_lob(ask, bid, price)))
    return days


def _summary_cards(days: list[SyntheticDay], config: DiagnosticsConfig) -> str:
    quality = assess_synthetic_quality(days, config)
    price = pd.concat([day.price for day in days], ignore_index=True)
    trades = sum(len(day.trades) for day in days)
    windows = sum(len(_episode_spans(day, config)) for day in days)
    cards = [
        ("Days", len(days)),
        ("Events", f"{len(price):,}"),
        ("2000-Event Windows", windows),
        ("Trades/Event", f"{trades / max(len(price), 1):.3f}"),
        ("Mean Spread Ticks", f"{price['spread_ticks'].mean():.2f}"),
        ("P95 Spread Ticks", f"{price['spread_ticks'].quantile(0.95):.2f}"),
        ("Return Std Bp", f"{price['return_bp'].std():.3f}"),
        ("Abs Window Move P50", f"{_window_move_quantile(days, config, 0.50):.1f} ticks"),
        ("Quality Score", f"{quality['score']:.1f}/100"),
    ]
    flags = quality.get("flags", [])
    flag_html = ""
    if flags:
        flag_html = "<section><h2>Quality Flags</h2><table>" + "".join(f"<tr><td>{flag}</td></tr>" for flag in flags) + "</table></section>"
    return "<div class='grid'>" + "".join(f"<div class='metric'><b>{name}</b><span>{value}</span></div>" for name, value in cards) + "</div>" + flag_html


def _plot_market_overview(days: list[SyntheticDay], path: Path) -> Path:
    fig, axes = plt.subplots(len(days), 1, figsize=(15, max(3.2, 2.8 * len(days))), squeeze=False)
    for ax, day in zip(axes.flat, days, strict=False):
        ax.plot(day.price["timestamp"], day.price["midprice"], color="#0f172a", linewidth=0.9, label="mid")
        if not day.latent.empty and "fair_value" in day.latent:
            ax.plot(day.latent["timestamp"], day.latent["fair_value"], color="#ef4444", linewidth=0.8, alpha=0.7, label="latent fair")
        ax.set_title(f"{day.symbol} {day.day}")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_episode_gallery(days: list[SyntheticDay], config: DiagnosticsConfig, path: Path) -> Path:
    windows = [(day, start, stop) for day in days for start, stop in _episode_spans(day, config)]
    windows = windows[: min(len(windows), 12)]
    cols = 3
    rows = max(1, int(np.ceil(len(windows) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows), squeeze=False)
    for ax, (day, start, stop) in zip(axes.flat, windows, strict=False):
        segment = day.price.iloc[start:stop].reset_index(drop=True)
        base = float(segment.iloc[0]["midprice"])
        ax.plot(10_000.0 * (segment["midprice"] / base - 1.0), color="#2563eb", linewidth=1.0)
        ax.set_title(f"{day.day} {start}:{stop}")
        ax.set_xlabel("event")
        ax.set_ylabel("bp")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(windows):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_spread_return_distribution(days: list[SyntheticDay], path: Path) -> Path:
    price = pd.concat([day.price for day in days], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(price["spread_ticks"], bins=np.arange(0.5, max(8, price["spread_ticks"].max()) + 1.5), color="#64748b")
    axes[0].set_title("Spread ticks")
    axes[1].hist(price["return_bp"].clip(-10, 10), bins=80, color="#2563eb")
    axes[1].set_title("Event returns bp clipped")
    axes[2].plot(price["return_bp"].rolling(2000).std().to_numpy(), color="#dc2626", linewidth=0.8)
    axes[2].set_title("Rolling 2000-event return std")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_order_flow(days: list[SyntheticDay], path: Path) -> Path:
    msg = pd.concat([day.msg for day in days], ignore_index=True)
    for column in MSG_COLUMNS:
        if column not in msg:
            msg[column] = 0
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    volumes = {
        "market buy": msg["market_buy_volume"].sum(),
        "market sell": msg["market_sell_volume"].sum(),
        "limit buy": msg["limit_buy_volume"].sum(),
        "limit sell": msg["limit_sell_volume"].sum(),
        "withdraw buy": msg["withdraw_buy_volume"].sum(),
        "withdraw sell": msg["withdraw_sell_volume"].sum(),
    }
    axes[0].bar(volumes.keys(), volumes.values(), color=["#2563eb", "#dc2626", "#38bdf8", "#fb7185", "#0f766e", "#f97316"])
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("Aggregate event volume")
    signed_market = (msg["market_buy_volume"] - msg["market_sell_volume"]).rolling(2000).sum()
    signed_limit = (msg["limit_buy_volume"] - msg["limit_sell_volume"]).rolling(2000).sum()
    axes[1].plot(signed_market.to_numpy(), label="market", color="#2563eb")
    axes[1].plot(signed_limit.to_numpy(), label="limit", color="#0f766e")
    axes[1].set_title("Rolling 2000-event signed flow")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_depth_heatmap(day: SyntheticDay, config: DiagnosticsConfig, path: Path) -> Path:
    spans = _episode_spans(day, config)
    start, stop = spans[0] if spans else (0, min(len(day.price), config.episode_length))
    cube = day.depth_cube[start:stop]
    vmax = np.percentile(np.abs(cube), 99) if cube.size else 1.0
    fig, ax = plt.subplots(figsize=(15, 5.5))
    im = ax.imshow(cube.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Depth around mid {day.day} {start}:{stop}")
    ax.set_xlabel("event")
    ax.set_ylabel("relative ticks")
    fig.colorbar(im, ax=ax, label="bid + / ask -")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_depth_snapshots(day: SyntheticDay, path: Path) -> Path:
    indices = np.linspace(0, len(day.price) - 1, 5, dtype=int)
    fig, axes = plt.subplots(1, len(indices), figsize=(16, 4.6), squeeze=False)
    for ax, idx in zip(axes.flat, indices, strict=False):
        ask = day.ask.iloc[idx]
        bid = day.bid.iloc[idx]
        ask_p = [ask[f"ask{level}_price"] for level in range(1, 11)]
        ask_v = [ask[f"ask{level}_volume"] for level in range(1, 11)]
        bid_p = [bid[f"bid{level}_price"] for level in range(1, 11)]
        bid_v = [bid[f"bid{level}_volume"] for level in range(1, 11)]
        ax.barh(bid_p, bid_v, color="#2563eb", alpha=0.8)
        ax.barh(ask_p, [-v for v in ask_v], color="#dc2626", alpha=0.8)
        ax.axhline(day.price.iloc[idx]["midprice"], color="#111827", linewidth=0.9)
        ax.set_title(f"event {idx}")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _plot_fill_curves(days: list[SyntheticDay], config: DiagnosticsConfig, path: Path) -> Path:
    rows = []
    for day in days:
        grouped = day.trades.groupby("timestamp") if not day.trades.empty else None
        candidate_indices = np.arange(config.lookback, len(day.price), max(1, len(day.price) // 600))
        for idx in candidate_indices:
            quote_idx = max(idx - config.latency, 0)
            ask1 = float(day.price.iloc[quote_idx]["ask1_price"])
            bid1 = float(day.price.iloc[quote_idx]["bid1_price"])
            for distance in range(config.as_max_distance_ticks + 1):
                delta = distance * config.symbol_spec.tick_size
                rows.append({"distance": distance, "side": "ask", "hit": _future_hit(day, grouped, idx, ask1 + delta, "B", config.as_fill_horizon_events)})
                rows.append({"distance": distance, "side": "bid", "hit": _future_hit(day, grouped, idx, bid1 - delta, "A", config.as_fill_horizon_events)})
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not frame.empty:
        summary = frame.groupby(["distance", "side"])["hit"].mean().reset_index()
        for side, group in summary.groupby("side"):
            ax.plot(group["distance"], group["hit"], marker="o", label=side)
    ax.set_title("Synthetic fill probability by quote distance")
    ax.set_xlabel("distance from touch, ticks")
    ax.set_ylabel("fill probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _baseline_tables(root: Path) -> str:
    candidates = [root / "paper_baseline_daily.csv", root / "paper_baseline_episodes.csv"]
    chunks = []
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "policy" in frame:
            columns = [column for column in ["pnl", "nd_pnl", "pnl_map", "profit_ratio", "avg_abs_position"] if column in frame.columns]
            summary = frame.groupby("policy")[columns].mean(numeric_only=True).round(4)
            chunks.append(f"<section><h2>{path.name}</h2>{summary.reset_index().to_html(index=False)}</section>")
    return "\n".join(chunks)


def _episode_spans(day: SyntheticDay, config: DiagnosticsConfig) -> list[tuple[int, int]]:
    clock = day.price["timestamp"].dt.strftime("%H:%M:%S")
    mask = np.zeros(len(clock), dtype=bool)
    for raw in config.stable_windows:
        start, end = raw.split("-", maxsplit=1)
        mask |= (clock >= start) & (clock <= end)
    idx = np.flatnonzero(mask)
    return [(int(idx[pos]), int(idx[pos + config.episode_length - 1]) + 1) for pos in range(0, max(len(idx) - config.episode_length + 1, 0), config.episode_length)]


def _window_move_quantile(days: list[SyntheticDay], config: DiagnosticsConfig, q: float) -> float:
    moves = []
    for day in days:
        for start, stop in _episode_spans(day, config):
            segment = day.price.iloc[start:stop]
            moves.append(abs(float(segment.iloc[-1]["midprice"] - segment.iloc[0]["midprice"])) / config.symbol_spec.tick_size)
    return float(np.quantile(moves, q)) if moves else 0.0


def _future_hit(day: SyntheticDay, grouped, idx: int, price: float, aggressor: str, horizon: int) -> bool:
    stop = min(idx + horizon, len(day.price))
    for event_idx in range(idx, stop):
        timestamp = day.price.iloc[event_idx]["timestamp"]
        if grouped is None or timestamp not in grouped.groups:
            continue
        block = grouped.get_group(timestamp)
        side = block[block["aggressor_side"] == aggressor]
        if side.empty:
            continue
        if aggressor == "B" and float(side["price"].max()) >= price:
            return True
        if aggressor == "A" and float(side["price"].min()) <= price:
            return True
    return False


def _depth_cube_from_lob(ask: pd.DataFrame, bid: pd.DataFrame, price: pd.DataFrame, radius: int = 15) -> np.ndarray:
    cube = np.zeros((len(price), radius * 2 + 1), dtype=np.float32)
    tick = float(np.median(np.diff(np.sort(price["ask1_price"].unique())))) if price["ask1_price"].nunique() > 1 else 0.01
    tick = tick if tick > 0 else 0.01
    for idx in range(len(price)):
        center = round(float(price.iloc[idx]["midprice"]) / tick)
        for level in range(1, 11):
            bid_rel = round(float(bid.iloc[idx][f"bid{level}_price"]) / tick) - center
            ask_rel = round(float(ask.iloc[idx][f"ask{level}_price"]) / tick) - center
            if -radius <= bid_rel <= radius:
                cube[idx, bid_rel + radius] = float(bid.iloc[idx][f"bid{level}_volume"])
            if -radius <= ask_rel <= radius:
                cube[idx, ask_rel + radius] = -float(ask.iloc[idx][f"ask{level}_volume"])
    return cube
