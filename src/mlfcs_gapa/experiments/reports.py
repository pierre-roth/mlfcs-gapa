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


def format_mean_std(mean: float | None, std: float | None, *, digits: int = 2) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"
