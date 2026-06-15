"""Table aggregation helpers for paper-style reports."""

from __future__ import annotations

import polars as pl


PAPER_TABLE_METRICS = ("nd_pnl_table", "pnl_map", "profit_ratio_table", "sharpe")


def add_paper_table_columns(metrics: pl.DataFrame) -> pl.DataFrame:
    """Add columns matching the paper's displayed metric scales."""

    expressions = []
    if "nd_pnl" in metrics.columns:
        expressions.append((pl.col("nd_pnl") / 1e5).alias("nd_pnl_table"))
    if "profit_ratio" in metrics.columns:
        expressions.append((pl.col("profit_ratio") * 1e4).alias("profit_ratio_table"))
    if "sharpe" not in metrics.columns:
        expressions.append(pl.lit(None, dtype=pl.Float64).alias("sharpe"))
    if not expressions:
        return metrics
    return metrics.with_columns(expressions)


def summarize_paper_table(
    metrics: pl.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("method", "stock"),
) -> pl.DataFrame:
    """Aggregate repeated runs into mean/std columns for Table II-style output."""

    metrics = add_paper_table_columns(metrics)
    available_groups = [column for column in group_columns if column in metrics.columns]
    if not available_groups:
        available_groups = ["method"]
    metric_columns = [column for column in PAPER_TABLE_METRICS if column in metrics.columns]
    if not metric_columns:
        raise ValueError("no paper metric columns found")

    expressions = []
    for column in metric_columns:
        expressions.extend(
            [
                pl.col(column).mean().alias(f"{column}_mean"),
                pl.col(column).std().alias(f"{column}_std"),
            ]
        )
    return metrics.group_by(available_groups).agg(expressions).sort(available_groups)


def aggregate_period_table(
    metrics: pl.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("method", "stock"),
    seed_column: str = "train_seed",
) -> pl.DataFrame:
    """Aggregate per-episode metrics into the paper's Table II convention.

    The paper reports test-period aggregates: total PnL over the held-out
    period divided by the average quoted spread (ND-PnL, shown x1e5), by the
    mean absolute position (PnLMAP), and by total traded buy notional
    (Profit Ratio, shown x1e-4); the Sharpe ratio is computed across episode
    PnLs. The +/- std is across training seeds, so deterministic baselines
    (no seed column entry) report mean values only.
    """

    required = {"pnl", "mean_quoted_spread", "mean_abs_inventory", "buy_notional"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"missing period metric columns: {sorted(missing)}")

    groups = [column for column in group_columns if column in metrics.columns]
    if not groups:
        groups = ["method"]
    seed_groups = [*groups, seed_column] if seed_column in metrics.columns else groups

    per_seed = (
        metrics.group_by(seed_groups)
        .agg(
            pl.col("pnl").sum().alias("pnl_total"),
            pl.col("mean_quoted_spread").mean().alias("mean_spread"),
            pl.col("mean_abs_inventory").mean().alias("mean_abs_position"),
            pl.col("buy_notional").sum().alias("buy_notional_total"),
            pl.col("pnl").mean().alias("pnl_episode_mean"),
            pl.col("pnl").std().alias("pnl_episode_std"),
            pl.len().alias("episodes"),
        )
        .with_columns(
            (pl.col("pnl_total") / (pl.col("mean_spread") + 1e-7) / 1e5).alias("nd_pnl_e5"),
            (pl.col("pnl_total") / (pl.col("mean_abs_position") + 1e-7)).alias("pnl_map"),
            (pl.col("pnl_total") / (pl.col("buy_notional_total") + 1e-7) * 1e4).alias(
                "profit_ratio_e4"
            ),
            (
                pl.col("pnl_episode_mean")
                / (pl.col("pnl_episode_std") + 1e-7)
                * pl.col("episodes").sqrt()
            ).alias("sharpe"),
        )
    )

    expressions = []
    for column in ("nd_pnl_e5", "pnl_map", "profit_ratio_e4", "sharpe"):
        expressions.append(pl.col(column).mean().alias(f"{column}_mean"))
        expressions.append(pl.col(column).std().alias(f"{column}_std"))
    expressions.extend(
        [
            pl.col("pnl_total").mean().alias("pnl_total_mean"),
            pl.col("episodes").sum().alias("episodes"),
            pl.len().alias("seeds"),
        ]
    )
    return per_seed.group_by(groups).agg(expressions).sort(groups)


def format_mean_std(mean: float | None, std: float | None, *, digits: int = 2) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"
