from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DiagnosticsConfig
from .simulator import SyntheticDay


def plot_midprice_days(days: list[SyntheticDay], output_path: Path) -> None:
    count = len(days)
    cols = min(2, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows), squeeze=False)
    for ax, day in zip(axes.flat, days, strict=False):
        frame = day.price
        ax.plot(frame["timestamp"], frame["midprice"], color="#0f172a", linewidth=1.0)
        ax.set_title(f"{day.symbol} {day.day}")
        ax.set_ylabel("Midprice")
        ax.grid(alpha=0.2)
    for ax in axes.flat[count:]:
        ax.axis("off")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_episode_windows(
    days: list[SyntheticDay],
    windows: list[tuple[SyntheticDay, int, int]],
    output_path: Path,
) -> None:
    cols = 3
    rows = int(np.ceil(len(windows) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.8 * rows), squeeze=False)
    for ax, (day, start, stop) in zip(axes.flat, windows, strict=False):
        segment = day.price.iloc[start:stop].reset_index(drop=True)
        base = float(segment.loc[0, "midprice"])
        series = 10_000.0 * (segment["midprice"] / base - 1.0)
        ax.plot(series.to_numpy(), color="#1d4ed8", linewidth=1.1)
        ax.axhline(0.0, color="#94a3b8", linewidth=0.8)
        ax.set_title(f"{day.day} [{start}:{stop}]")
        ax.set_ylabel("bp from start")
        ax.set_xlabel("event")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(windows):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_lob_heatmap(day: SyntheticDay, start: int, stop: int, output_path: Path) -> None:
    cube = day.depth_cube[start:stop]
    fig, ax = plt.subplots(figsize=(15, 6))
    vmax = np.percentile(np.abs(cube), 99)
    im = ax.imshow(
        cube.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title(f"Depth heatmap {day.symbol} {day.day} [{start}:{stop}]")
    ax.set_xlabel("event")
    ax.set_ylabel("relative ticks around mid")
    fig.colorbar(im, ax=ax, label="bid depth (+) / ask depth (-)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_lob_snapshots(day: SyntheticDay, indices: list[int], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(indices), figsize=(5.5 * len(indices), 5), squeeze=False)
    for ax, idx in zip(axes.flat, indices, strict=False):
        ask = day.ask.iloc[idx]
        bid = day.bid.iloc[idx]
        ask_prices = np.asarray([ask[f"ask{level}_price"] for level in range(1, 11)], dtype=np.float64)
        ask_sizes = np.asarray([ask[f"ask{level}_volume"] for level in range(1, 11)], dtype=np.float64)
        bid_prices = np.asarray([bid[f"bid{level}_price"] for level in range(1, 11)], dtype=np.float64)
        bid_sizes = np.asarray([bid[f"bid{level}_volume"] for level in range(1, 11)], dtype=np.float64)
        ax.barh(bid_prices, bid_sizes, color="#2563eb", alpha=0.8)
        ax.barh(ask_prices, -ask_sizes, color="#dc2626", alpha=0.8)
        ax.axhline(day.price.iloc[idx]["midprice"], color="#111827", linestyle="--", linewidth=1.0)
        ax.set_title(f"event {idx}")
        ax.set_xlabel("depth (+bid / -ask)")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_window_summary(windows: list[tuple[SyntheticDay, int, int]], output_path: Path) -> None:
    rows = []
    for day, start, stop in windows:
        segment = day.price.iloc[start:stop]
        start_mid = float(segment.iloc[0]["midprice"])
        end_mid = float(segment.iloc[-1]["midprice"])
        tick_size = float(np.median(np.diff(np.unique(day.ask["ask1_price"].to_numpy(dtype=np.float64)[:1000])))) if len(day.ask) > 1 else 0.01
        if tick_size <= 0 or np.isnan(tick_size):
            tick_size = 0.01
        rows.append(
            {
                "day": day.day,
                "start_event": start,
                "stop_event": stop,
                "start_midprice": start_mid,
                "end_midprice": end_mid,
                "move_ticks": round((end_mid - start_mid) / tick_size, 6),
                "return_bp": 10_000.0 * (end_mid / start_mid - 1.0),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)
