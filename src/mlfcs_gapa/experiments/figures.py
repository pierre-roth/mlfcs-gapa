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
    """Create a paper Figure-4-style price/value/inventory trace."""

    required = {"index", "mid_price", "ask_price", "bid_price", "inventory", "value"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"missing decision trace columns: {sorted(missing)}")

    x = trades["index"].to_numpy()
    mid = trades["mid_price"].to_numpy()
    ask = trades["ask_price"].to_numpy()
    bid = trades["bid_price"].to_numpy()
    value = trades["value"].to_numpy()
    inventory = trades["inventory"].to_numpy()

    ask_line = np.where(ask > 0, ask, np.nan)
    bid_line = np.where(bid > 0, bid, np.nan)

    fig, (axis_price, axis_inventory) = plt.subplots(
        2,
        1,
        figsize=(10, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.18},
    )
    value_axis = axis_price.twinx()

    price_line = axis_price.step(x, mid, where="post", color="black", linewidth=1.1, label="price")[0]
    bid_plot = axis_price.plot(x, bid_line, color="#ff7f7f", linewidth=1.2, alpha=0.9, label="bid")[0]
    ask_plot = axis_price.plot(x, ask_line, color="#7fbf7f", linewidth=1.2, alpha=0.9, label="ask")[0]
    value_line = value_axis.step(
        x, value, where="post", color="blue", linewidth=1.8, alpha=0.9, label="value"
    )[0]

    axis_price.set_ylabel("Price")
    value_axis.set_ylabel("Value")
    axis_price.set_xlim(float(x.min()), float(x.max()))
    axis_price.grid(False)

    axis_inventory.step(x, inventory, where="post", color="#1f77b4", linewidth=1.3)
    axis_inventory.axhline(0, color="green", linestyle="--", linewidth=1.1)
    axis_inventory.set_ylabel("Inventory")
    axis_inventory.set_xlabel("Time (s)")
    axis_inventory.grid(False)

    axis_price.legend(
        [price_line, value_line, bid_plot, ask_plot],
        ["price", "value", "bid", "ask"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        fancybox=False,
    )

    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.92, hspace=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    output_path: Path,
    *,
    lob_window: np.ndarray | None = None,
) -> None:
    """Create a paper Figure-3-style attention and LOB-state visualization.

    `attention_weights` may have shape `(heads, window)` or `(window,)`.
    When `lob_window` is provided, it must have shape `(50, 40)` and is shown
    below the attention bar with volume rows above price rows.
    """

    weights = np.asarray(attention_weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.ndim != 2:
        raise ValueError(f"attention weights must be 1D or 2D, got {weights.shape}")

    attention = weights.sum(axis=0)
    x = np.arange(attention.shape[0])

    if lob_window is None:
        fig, axis = plt.subplots(figsize=(8, 2.8))
        axis.bar(x, attention, width=0.8, color="#1f77b4")
        axis.set_xlim(0, attention.shape[0])
        axis.set_xlabel("Timestamps")
        axis.set_ylabel("Attn")
    else:
        state = _paper_lob_state_matrix(lob_window)
        fig, (axis_attention, axis_state) = plt.subplots(
            2,
            1,
            figsize=(8.5, 6.2),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 4.0], "hspace": 0.18},
        )
        axis_attention.bar(x, attention, width=0.8, color="#1f77b4")
        axis_attention.axhline(0, color="black", linewidth=0.8)
        axis_attention.set_ylabel("Attn", fontsize=20)
        axis_attention.set_xlim(0, attention.shape[0])

        axis_state.imshow(
            state,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            extent=[0, attention.shape[0], state.shape[0], 0],
        )
        axis_state.set_xlabel("Timestamps", fontsize=22)
        axis_state.set_xlim(0, attention.shape[0])
        axis_state.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
        axis_state.set_yticklabels(["10", "ask", "0", "bid", "10", "ask", "0", "bid", "10"])
        axis_state.text(
            -0.13,
            0.75,
            "Volume",
            transform=axis_state.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=22,
        )
        axis_state.text(
            -0.13,
            0.25,
            "Price",
            transform=axis_state.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=22,
        )
    if lob_window is None:
        fig.tight_layout()
    else:
        fig.subplots_adjust(top=0.97, bottom=0.12, left=0.14, right=0.99, hspace=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _paper_lob_state_matrix(lob_window: np.ndarray) -> np.ndarray:
    window = np.asarray(lob_window, dtype=np.float64)
    if window.shape != (50, 40):
        raise ValueError(f"lob_window must have shape (50, 40), got {window.shape}")

    ask_price_cols = [4 * level for level in range(10)]
    ask_volume_cols = [4 * level + 1 for level in range(10)]
    bid_price_cols = [4 * level + 2 for level in range(10)]
    bid_volume_cols = [4 * level + 3 for level in range(10)]

    volume = np.vstack(
        [
            window[:, ask_volume_cols[::-1]].T,
            window[:, bid_volume_cols].T,
        ]
    )
    price = np.vstack(
        [
            window[:, ask_price_cols[::-1]].T,
            window[:, bid_price_cols].T,
        ]
    )
    return np.vstack([volume, price])


def _ordered_methods(methods: list[str]) -> list[str]:
    preferred = ["C-PPO", "D-DQN", "AS", "Random", "Fixed", "Fixed_1", "Fixed_2", "Fixed_3"]
    ordered = [method for method in preferred if method in methods]
    ordered.extend(sorted(method for method in methods if method not in ordered))
    return ordered
