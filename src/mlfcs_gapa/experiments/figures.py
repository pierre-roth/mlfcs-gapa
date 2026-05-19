"""Paper-style figures from replication artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_latency_figure(
    metrics: pl.DataFrame,
    output_path: Path,
    *,
    paper_scale: bool = True,
    methods: list[str] | None = None,
) -> None:
    """Create a Figure-2-style latency plot.

    Expected columns: `method`, `latency_events`, `nd_pnl`, `pnl_map`,
    `profit_ratio`.
    """

    required = {"method", "latency_events", "nd_pnl", "pnl_map", "profit_ratio"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"missing latency metric columns: {sorted(missing)}")

    methods = methods or _ordered_methods(metrics["method"].unique().to_list())
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.2), sharex=True)
    if len(methods) == 1:
        axes = [axes]

    for axis, method in zip(axes, methods, strict=True):
        subset = (
            metrics.filter(pl.col("method") == method)
            .group_by("latency_events")
            .agg(
                pl.col("nd_pnl").mean().alias("nd_pnl"),
                pl.col("pnl_map").mean().alias("pnl_map"),
                pl.col("profit_ratio").mean().alias("profit_ratio"),
            )
            .sort("latency_events")
        )
        if subset.is_empty():
            axis.set_axis_off()
            continue
        x = subset["latency_events"].to_numpy()
        nd_pnl = subset["nd_pnl"].to_numpy()
        profit_ratio = subset["profit_ratio"].to_numpy()
        if paper_scale:
            nd_pnl = nd_pnl / 1e5
            profit_ratio = profit_ratio * 1e4

        axis.plot(x, nd_pnl, label="ND-PnL", linewidth=2.0)
        axis.plot(x, subset["pnl_map"].to_numpy(), label="PnLMAP", linewidth=2.0)
        axis.plot(x, profit_ratio, label="Profit Ratio", linewidth=2.0)
        axis.set_title(method)
        axis.set_xlabel("Latency")
        axis.grid(alpha=0.2)

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_decision_trace(trades: pl.DataFrame, output_path: Path) -> None:
    """Create a Figure-4-style quote/price/inventory trace."""

    required = {"index", "mid_price", "ask_price", "bid_price", "inventory"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"missing decision trace columns: {sorted(missing)}")

    x = trades["index"].to_numpy()
    mid = trades["mid_price"].to_numpy()
    ask = trades["ask_price"].to_numpy()
    bid = trades["bid_price"].to_numpy()
    inventory = trades["inventory"].to_numpy()

    fig, axis_price = plt.subplots(figsize=(10, 4))
    axis_price.plot(x, mid, color="black", linewidth=1.4, label="Mid")
    active_ask = ask > 0
    active_bid = bid > 0
    axis_price.scatter(x[active_ask], ask[active_ask], s=12, color="#d62728", label="Ask")
    axis_price.scatter(x[active_bid], bid[active_bid], s=12, color="#2ca02c", label="Bid")
    axis_price.set_xlabel("Event index")
    axis_price.set_ylabel("Price")
    axis_price.grid(alpha=0.2)

    axis_inventory = axis_price.twinx()
    axis_inventory.plot(x, inventory, color="#1f77b4", alpha=0.45, linewidth=1.2, label="Inventory")
    axis_inventory.axhline(0, color="#1f77b4", alpha=0.25, linewidth=0.8)
    axis_inventory.set_ylabel("Inventory")

    handles_price, labels_price = axis_price.get_legend_handles_labels()
    handles_inv, labels_inv = axis_inventory.get_legend_handles_labels()
    axis_price.legend(handles_price + handles_inv, labels_price + labels_inv, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_attention_heatmap(attention_weights: np.ndarray, output_path: Path) -> None:
    """Create a Figure-3-style attention heatmap.

    Input shape may be `(heads, window)` or `(window,)`.
    """

    weights = np.asarray(attention_weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.ndim != 2:
        raise ValueError(f"attention weights must be 1D or 2D, got {weights.shape}")

    fig, axis = plt.subplots(figsize=(8, 2.8))
    image = axis.imshow(weights, aspect="auto", cmap="viridis", interpolation="nearest")
    axis.set_xlabel("Event offset in 50-event window")
    axis.set_ylabel("Attention head")
    fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _ordered_methods(methods: list[str]) -> list[str]:
    preferred = ["C-PPO", "D-DQN", "AS", "Random", "Fixed", "Fixed_1", "Fixed_2", "Fixed_3"]
    ordered = [method for method in preferred if method in methods]
    ordered.extend(sorted(method for method in methods if method not in ordered))
    return ordered
