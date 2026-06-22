"""Aggregate AS-guided extension runs into report artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import typer


app = typer.Typer(help="Aggregate AS-guided extension artifacts.")


def build_as_guided_report_artifacts(
    *,
    extension_root: Path,
    output_dir: Path,
    baseline_replication_root: Path | None = None,
) -> dict[str, Path]:
    """Build combined tables and figures from completed extension run dirs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paper_tables = _read_many(extension_root, "extension_paper_table.csv")
    metrics = _read_many(extension_root, "extension_metrics.csv")
    diagnostics = _read_many(extension_root, "extension_trade_diagnostics.csv")
    if paper_tables.is_empty():
        raise ValueError(f"no extension_paper_table.csv files found under {extension_root}")

    combined_paper = paper_tables
    if baseline_replication_root is not None:
        baseline_path = baseline_replication_root / "table_ii_overall" / "overall_paper_table.csv"
        if baseline_path.exists():
            baseline = (
                pl.read_csv(baseline_path, schema_overrides={"stock": pl.Utf8})
                .filter(pl.col("method") == "C-PPO")
                .with_columns(
                    pl.lit("paper_cppo_original").alias("method"),
                    pl.lit("paper_replication_3272020").alias("variant"),
                )
            )
            combined_paper = pl.concat([baseline, combined_paper], how="diagonal_relaxed")

    combined_paper_path = output_dir / "combined_paper_table.csv"
    metrics_path = output_dir / "combined_metrics.csv"
    diagnostics_path = output_dir / "combined_trade_diagnostics.csv"
    ranking_path = output_dir / "variant_ranking.csv"
    combined_paper.write_csv(combined_paper_path)
    metrics.write_csv(metrics_path)
    diagnostics.write_csv(diagnostics_path)

    ranking = _variant_ranking(combined_paper)
    ranking.write_csv(ranking_path)

    pnl_figure = output_dir / "extension_pnl_by_stock.png"
    participation_figure = output_dir / "extension_participation.png"
    action_figure = output_dir / "extension_action_spread.png"
    _plot_pnl_by_stock(combined_paper, pnl_figure)
    if not diagnostics.is_empty():
        _plot_participation(diagnostics, participation_figure)
        _plot_action_spread(diagnostics, action_figure)

    report_path = output_dir / "as_guided_extension_report.md"
    report_path.write_text(
        _markdown_report(
            combined_paper=combined_paper,
            ranking=ranking,
            diagnostics=diagnostics,
            include_action_figure=action_figure.exists(),
        ),
        encoding="utf-8",
    )

    return {
        "combined_paper_table": combined_paper_path,
        "combined_metrics": metrics_path,
        "combined_trade_diagnostics": diagnostics_path,
        "variant_ranking": ranking_path,
        "pnl_figure": pnl_figure,
        "participation_figure": participation_figure,
        "action_figure": action_figure,
        "report": report_path,
    }


def _read_many(root: Path, filename: str) -> pl.DataFrame:
    frames = []
    for path in sorted(root.glob(f"*/{filename}")):
        frame = pl.read_csv(path, schema_overrides={"stock": pl.Utf8}, infer_schema_length=None)
        frame = frame.with_columns(pl.lit(path.parent.name).alias("run_label"))
        frames.append(frame)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _variant_ranking(paper: pl.DataFrame) -> pl.DataFrame:
    return (
        paper.group_by("method")
        .agg(
            pl.col("pnl_total_mean").mean().alias("mean_pnl_total_by_stock"),
            pl.col("nd_pnl_e5_mean").mean().alias("mean_nd_pnl_e5_by_stock"),
            pl.col("profit_ratio_e4_mean").mean().alias("mean_profit_ratio_e4_by_stock"),
            pl.col("sharpe_mean").mean().alias("mean_sharpe_by_stock"),
            (pl.col("pnl_total_mean") > 0).sum().alias("positive_stocks"),
            pl.len().alias("stock_rows"),
        )
        .sort("mean_pnl_total_by_stock", descending=True)
    )


def _plot_pnl_by_stock(paper: pl.DataFrame, output_path: Path) -> None:
    pivot = (
        paper.select(["method", "stock", "pnl_total_mean"])
        .pivot(index="method", on="stock", values="pnl_total_mean", aggregate_function="first")
        .sort("method")
    )
    methods = pivot["method"].to_list()
    stocks = [column for column in pivot.columns if column != "method"]
    x = range(len(methods))
    width = 0.8 / max(1, len(stocks))
    fig, axis = plt.subplots(figsize=(max(10, 0.7 * len(methods)), 5.2))
    for i, stock in enumerate(stocks):
        offsets = [value + (i - (len(stocks) - 1) / 2) * width for value in x]
        axis.bar(offsets, pivot[stock].fill_null(0.0).to_numpy(), width=width, label=stock)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Period PnL")
    axis.set_xticks(list(x))
    axis.set_xticklabels(methods, rotation=35, ha="right")
    axis.legend(title="Stock")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_participation(diagnostics: pl.DataFrame, output_path: Path) -> None:
    frame = (
        diagnostics.group_by(["method", "stock"])
        .agg(
            pl.col("fills").mean().alias("fills_mean"),
            (pl.col("fills") == 0).mean().alias("zero_fill_share"),
        )
        .sort(["stock", "method"])
    )
    methods = frame["method"].unique().sort().to_list()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, column, ylabel in (
        (axes[0], "fills_mean", "Mean fills / episode"),
        (axes[1], "zero_fill_share", "Zero-fill episode share"),
    ):
        values = [
            frame.filter(pl.col("method") == method)[column].mean()
            for method in methods
        ]
        axis.bar(methods, values)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_action_spread(diagnostics: pl.DataFrame, output_path: Path) -> None:
    if "action_spread_mean" not in diagnostics.columns:
        return
    frame = (
        diagnostics.group_by(["method", "stock"])
        .agg(
            pl.col("action_spread_mean").mean().alias("action_spread_mean"),
            *[
                pl.col(column).mean().alias(column)
                for column in ("teacher_action_spread_mean", "raw_action_spread_mean")
                if column in diagnostics.columns
            ],
        )
        .sort(["stock", "method"])
    )
    fig, axis = plt.subplots(figsize=(12, 4.8))
    labels = [f"{row['method']}\\n{row['stock']}" for row in frame.to_dicts()]
    axis.plot(labels, frame["action_spread_mean"].to_numpy(), marker="o", label="executed")
    if "teacher_action_spread_mean" in frame.columns:
        axis.plot(labels, frame["teacher_action_spread_mean"].to_numpy(), marker="o", label="AS")
    if "raw_action_spread_mean" in frame.columns:
        axis.plot(labels, frame["raw_action_spread_mean"].to_numpy(), marker="o", label="raw")
    axis.set_ylabel("Paper action spread")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _markdown_report(
    *,
    combined_paper: pl.DataFrame,
    ranking: pl.DataFrame,
    diagnostics: pl.DataFrame,
    include_action_figure: bool,
) -> str:
    lines = [
        "# AS-Guided Market-Making Extension Results",
        "",
        "This report compares AS-guided C-PPO variants against the paper C-PPO baseline on the same synthetic panel.",
        "",
        "## Figures",
        "",
        "![PnL by stock](extension_pnl_by_stock.png)",
        "",
        "![Participation diagnostics](extension_participation.png)",
        "",
    ]
    if include_action_figure:
        lines.extend(["![Action spread diagnostics](extension_action_spread.png)", ""])
    lines.extend(
        [
            "## Variant Ranking",
            "",
            _to_markdown_table(ranking),
            "",
            "## Paper-Style Table",
            "",
            _to_markdown_table(combined_paper.sort(["stock", "method"])),
            "",
        ]
    )
    if not diagnostics.is_empty():
        participation = (
            diagnostics.group_by(["method", "stock"])
            .agg(
                pl.col("fills").mean().alias("fills_mean"),
                (pl.col("fills") == 0).mean().alias("zero_fill_share"),
                pl.col("max_abs_inventory").mean().alias("max_abs_inventory_mean"),
            )
            .sort(["stock", "method"])
        )
        lines.extend(["## Participation Summary", "", _to_markdown_table(participation), ""])
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- `paper_cppo_original` is the C-PPO row from full replication job 3272020.")
    lines.append("- `paper_cppo_rerun` is the same PPO setup rerun through the extension harness.")
    lines.append("- `bc_as_*` uses AS behavioral cloning before PPO fine-tuning.")
    lines.append("- `soft_as_*` keeps the original action but subtracts a quadratic AS-divergence penalty.")
    lines.append("- `hard_as_*` clips executed actions to a fixed paper-action window around AS.")
    lines.append("")
    return "\n".join(lines)


def _to_markdown_table(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "_No rows._"
    columns = frame.columns
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.to_dicts():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


@app.command("build")
def build_command(
    extension_root: Path = typer.Argument(..., help="Root containing one subdir per extension run."),
    output_dir: Path = typer.Option(..., help="Output directory for combined report artifacts."),
    baseline_replication_root: Path | None = typer.Option(None),
) -> None:
    paths = build_as_guided_report_artifacts(
        extension_root=extension_root,
        output_dir=output_dir,
        baseline_replication_root=baseline_replication_root,
    )
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


if __name__ == "__main__":
    app()

