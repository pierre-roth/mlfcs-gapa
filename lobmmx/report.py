from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyrallis
import seaborn as sns

from .config import ExperimentConfig
from .metrics import sharpe, sharpe_annualized_episodes, sharpe_daily
from .pipeline import prepare_run
from .utils import ensure_dir

OVERALL_METHOD_ORDER = [
    "PPO_full",
    "AS",
    "Random",
    "Fixed_1",
    "Fixed_2",
    "Fixed_3",
]

ABLATION_METHOD_ORDER = [
    "PPO_full",
    "PPO_wo_lob",
    "PPO_wo_attn",
    "PPO_wo_dynamic",
]

DISPLAY_NAME = {
    "PPO_full": "C-PPO",
    "PPO_wo_lob": "w/o LOB state",
    "PPO_wo_attn": "w/o Attn-LOB",
    "PPO_wo_dynamic": "w/o Dynamic state",
    "AS": "AS",
    "Random": "Random",
    "Fixed_1": "Fixed_1",
    "Fixed_2": "Fixed_2",
    "Fixed_3": "Fixed_3",
}

RESULT_COLUMNS = {
    "symbol",
    "day",
    "method",
    "episode_index",
    "pnl",
    "nd_pnl",
    "pnl_map",
    "profit_ratio",
    "avg_position",
    "avg_abs_position",
    "avg_spread",
    "turnover",
    "reward",
    "trades",
    "latency",
    "fill_rate",
    "avg_bias_bps",
    "avg_ask_distance_bps",
    "avg_bid_distance_bps",
    "avg_spread_bps",
}


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def _display_name(method: str) -> str:
    return DISPLAY_NAME.get(method, method)


def _method_sort_key(method: str) -> tuple[int, str]:
    if method in OVERALL_METHOD_ORDER:
        return (OVERALL_METHOD_ORDER.index(method), method)
    if method in ABLATION_METHOD_ORDER:
        return (100 + ABLATION_METHOD_ORDER.index(method), method)
    return (999, method)


def _collect_episode_frames(out_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(out_dir.glob("**/*.csv")):
        if "reports" in path.parts or "latency" in path.parts:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if RESULT_COLUMNS.issubset(frame.columns):
            frame["source"] = str(path.relative_to(out_dir))
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["display_method"] = combined["method"].map(_display_name)
    return combined


def _format_overall_results(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["nd_pnl", "pnl_map", "profit_ratio"]
    summary = (
        combined.groupby(["symbol", "method"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(str(part) for part in column if part)
        for column in summary.columns.to_flat_index()
    ]
    rows = []
    groups = {key: group for key, group in combined.groupby(["symbol", "method"])}
    sharpe_map = {key: sharpe(group["pnl"].tolist()) for key, group in groups.items()}
    sharpe_daily_map = {
        key: sharpe_daily(group["pnl"].tolist(), group["day"].astype(str).tolist())
        for key, group in groups.items()
    }
    sharpe_annual_ep_map = {
        key: sharpe_annualized_episodes(group["pnl"].tolist(), len(group) / max(int(group["day"].nunique()), 1))
        for key, group in groups.items()
    }
    for _, row in summary.iterrows():
        formatted = {
            "symbol": row["symbol"],
            "method": row["method"],
            "display_method": _display_name(row["method"]),
        }
        for metric in metrics:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            if pd.isna(std):
                std = 0.0
            formatted[metric] = f"{mean:.3f}±{std:.3f}"
        formatted["sharpe"] = f"{sharpe_map[(row['symbol'], row['method'])]:.3f}"
        formatted["sharpe_annual_daily"] = f"{sharpe_daily_map.get((row['symbol'], row['method']), 0.0):.2f}"
        formatted["sharpe_annual_ep"] = f"{sharpe_annual_ep_map.get((row['symbol'], row['method']), 0.0):.2f}"
        rows.append(formatted)
    formatted_df = pd.DataFrame(rows).sort_values(["symbol", "method"], key=lambda series: series.map(lambda x: _method_sort_key(x)[0]) if series.name == "method" else series)
    summary["sharpe"] = summary.apply(lambda row: sharpe_map[(row["symbol"], row["method"])], axis=1)
    summary["sharpe_annual_daily"] = summary.apply(lambda row: sharpe_daily_map.get((row["symbol"], row["method"]), 0.0), axis=1)
    summary["sharpe_annual_ep"] = summary.apply(lambda row: sharpe_annual_ep_map.get((row["symbol"], row["method"]), 0.0), axis=1)
    return summary, formatted_df


def _save_overall_table(summary: pd.DataFrame, formatted: pd.DataFrame, report_dir: Path) -> None:
    summary.to_csv(report_dir / "continuous_overall_results.csv", index=False)
    if formatted.empty:
        return
    markdown = formatted[formatted["method"].isin(OVERALL_METHOD_ORDER)].copy()
    markdown["method"] = markdown["display_method"]
    markdown = markdown.drop(columns=["display_method"])
    _save_markdown_table(markdown, report_dir / "continuous_overall_results.md")


def _save_runtime_table(out_dir: Path, report_dir: Path) -> None:
    rows = []
    for timing_file in sorted(out_dir.glob("**/*timing.json")):
        if "latency" in timing_file.parts or any(part.startswith("latency_") for part in timing_file.parts):
            continue
        payload = _load_json(timing_file)
        if not payload:
            continue
        relative = timing_file.relative_to(out_dir)
        symbol = relative.parts[0] if len(relative.parts) > 1 else "ALL"
        rows.append(
            {
                "symbol": symbol,
                "method": payload.get("method", timing_file.stem.replace("_timing", "")),
                "display_method": _display_name(payload.get("method", timing_file.stem.replace("_timing", ""))),
                "inference_ms_per_step": float(payload.get("inference_ms_per_step", 0.0)),
                "train_ms_per_step": float(payload.get("train_ms_per_step", 0.0)),
            }
        )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["symbol", "method"], key=lambda series: series.map(lambda x: _method_sort_key(x)[0]) if series.name == "method" else series)
    frame.to_csv(report_dir / "runtime_summary.csv", index=False)
    display = frame[["symbol", "display_method", "inference_ms_per_step", "train_ms_per_step"]].rename(columns={"display_method": "method"})
    _save_markdown_table(display, report_dir / "runtime_summary.md")


def _save_policy_diagnostics(combined: pd.DataFrame, report_dir: Path) -> None:
    columns = [
        "fill_rate",
        "avg_bias_bps",
        "avg_ask_distance_bps",
        "avg_bid_distance_bps",
        "avg_spread_bps",
    ]
    available = [column for column in columns if column in combined.columns]
    if not available:
        return
    frame = combined.groupby("method")[available].mean().reset_index()
    frame["display_method"] = frame["method"].map(_display_name)
    frame = frame.sort_values("method", key=lambda series: series.map(lambda value: _method_sort_key(value)[0]))
    frame.to_csv(report_dir / "policy_diagnostics.csv", index=False)
    _save_markdown_table(
        frame[["display_method", *available]].rename(columns={"display_method": "method"}),
        report_dir / "policy_diagnostics.md",
    )


def _training_curves(out_dir: Path, report_dir: Path) -> None:
    pretrain_files = sorted(out_dir.glob("*/pretrain/history.csv"))
    ppo_files = sorted(out_dir.glob("*/ppo/full/history.csv"))
    if pretrain_files:
        pretrain_frames = []
        for file in pretrain_files:
            frame = pd.read_csv(file)
            frame["symbol"] = file.parts[-3]
            pretrain_frames.append(frame)
        pretrain_df = pd.concat(pretrain_frames, ignore_index=True)
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=pretrain_df, x="epoch", y="f1", hue="symbol", marker="o")
        plt.title("Pretrain F1")
        plt.tight_layout()
        plt.savefig(report_dir / "pretrain_f1.png")
        plt.close()
    if ppo_files:
        ppo_frames = []
        for file in ppo_files:
            frame = pd.read_csv(file)
            frame["symbol"] = file.parts[-4]
            ppo_frames.append(frame)
        ppo_df = pd.concat(ppo_frames, ignore_index=True)
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=ppo_df, x="epoch", y="reward_mean", hue="symbol", marker="o")
        plt.title("PPO Reward Mean")
        plt.tight_layout()
        plt.savefig(report_dir / "ppo_reward_curve.png")
        plt.close()


def _latency_plot(out_dir: Path, report_dir: Path) -> None:
    latency_files = sorted(out_dir.glob("*/latency/latency_*.csv"))
    if not latency_files:
        return
    frames = []
    for file in latency_files:
        frame = pd.read_csv(file)
        frame["symbol"] = file.parts[-3]
        frame["latency"] = int(file.stem.split("_")[-1])
        frames.append(frame)
    latency_df = pd.concat(frames, ignore_index=True)
    latency_summary = latency_df.groupby(["symbol", "latency", "method"])[["pnl", "nd_pnl"]].mean().reset_index()
    latency_summary.to_csv(report_dir / "latency_summary.csv", index=False)
    plot_df = latency_summary[
        latency_summary["method"].apply(lambda value: value.startswith("PPO_full_latency_") or value in {"AS", "Random", "Fixed_1", "Fixed_2", "Fixed_3"})
    ].copy()
    if not plot_df.empty:
        plot_df["display_method"] = plot_df["method"].replace({f"PPO_full_latency_{latency}": "C-PPO" for latency in plot_df["latency"].unique()})
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=plot_df, x="latency", y="nd_pnl", hue="display_method", style="symbol", marker="o")
        plt.title("Latency Study (ND-PnL)")
        plt.tight_layout()
        plt.savefig(report_dir / "latency_pnl.png")
        plt.close()


def _ablation_plot(out_dir: Path, report_dir: Path) -> None:
    rows = []
    for variant in ["full", "full_wo_lob", "full_simple_backbone", "full_wo_dynamic"]:
        payload = _load_json(out_dir / f"train_ppo_{variant}.json")
        if not payload:
            continue
        for symbol, metrics in payload.items():
            rows.append({"variant": variant, "symbol": symbol, **metrics})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame["display_variant"] = frame["variant"].replace(
        {
            "full": "C-PPO",
            "full_wo_lob": "w/o LOB state",
            "full_simple_backbone": "w/o Attn-LOB",
            "full_wo_dynamic": "w/o Dynamic state",
        }
    )
    frame.to_csv(report_dir / "ablation_summary.csv", index=False)
    _save_markdown_table(
        frame[["symbol", "display_variant", "nd_pnl_mean", "pnl_map_mean", "profit_ratio_mean", "sharpe"]].rename(
            columns={
                "display_variant": "variant",
                "nd_pnl_mean": "ND-PnL",
                "pnl_map_mean": "PnLMAP",
                "profit_ratio_mean": "ProfitRatio",
                "sharpe": "Sharpe",
            }
        ),
        report_dir / "ablation_summary.md",
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=frame, x="display_variant", y="nd_pnl_mean", hue="symbol")
    plt.title("PPO Ablations (ND-PnL)")
    plt.tight_layout()
    plt.savefig(report_dir / "ablation_pnl.png")
    plt.close()


def _decision_and_attention_plots(out_dir: Path, report_dir: Path) -> None:
    trace_files = sorted(out_dir.glob("*/ppo/full/traces/episode_0.csv"))
    for trace_file in trace_files:
        trace = pd.read_csv(trace_file)
        symbol = trace_file.parts[-5]
        if trace.empty:
            continue
        plt.figure(figsize=(10, 5))
        ax1 = plt.gca()
        ax1.plot(trace["midprice"].values, label="midprice", color="#1f77b4")
        ax1.plot(trace["ask_quote"].values, label="ask quote", color="#d62728", alpha=0.8)
        ax1.plot(trace["bid_quote"].values, label="bid quote", color="#2ca02c", alpha=0.8)
        ax2 = ax1.twinx()
        ax2.plot(trace["inventory"].values, label="inventory", color="#9467bd", alpha=0.5)
        plt.title(f"{symbol} PPO Decision History")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
        plt.tight_layout()
        plt.savefig(report_dir / f"{symbol.lower()}_decision_history.png")
        plt.close()

        attention_path = trace_file.with_name("episode_0_attention.csv")
        if attention_path.exists():
            attention = pd.read_csv(attention_path)
            plt.figure(figsize=(10, 4))
            sns.heatmap(attention.T, cmap="viridis")
            plt.title(f"{symbol} Attention Weights")
            plt.xlabel("Decision step")
            plt.ylabel("Lookback index")
            plt.tight_layout()
            plt.savefig(report_dir / f"{symbol.lower()}_attention_heatmap.png")
            plt.close()


def run_report(config: ExperimentConfig) -> Path:
    config.apply_mode_defaults()
    out_dir = prepare_run(config, label="report")
    report_dir = ensure_dir(out_dir / "reports")
    combined = _collect_episode_frames(out_dir)
    if combined.empty:
        return report_dir
    combined.to_csv(report_dir / "combined_results.csv", index=False)

    method_summary = (
        combined.groupby("method")[["pnl", "nd_pnl", "pnl_map", "profit_ratio", "avg_abs_position", "avg_spread", "turnover", "reward"]]
        .mean()
        .reset_index()
    )
    method_summary["sharpe"] = method_summary["method"].map(lambda method: sharpe(combined.loc[combined["method"] == method, "pnl"].tolist()))
    method_summary = method_summary.sort_values("method", key=lambda series: series.map(lambda value: _method_sort_key(value)[0]))
    method_summary["display_method"] = method_summary["method"].map(_display_name)
    method_summary.to_csv(report_dir / "method_summary.csv", index=False)
    _save_markdown_table(
        method_summary[["display_method", "pnl", "nd_pnl", "pnl_map", "profit_ratio", "sharpe", "avg_abs_position", "avg_spread", "turnover", "reward"]].rename(columns={"display_method": "method"}),
        report_dir / "method_summary.md",
    )

    symbol_summary = (
        combined.groupby(["symbol", "method"])[["pnl", "nd_pnl", "pnl_map", "profit_ratio"]]
        .mean()
        .reset_index()
    )
    symbol_summary["sharpe"] = symbol_summary.apply(
        lambda row: sharpe(combined.loc[(combined["symbol"] == row["symbol"]) & (combined["method"] == row["method"]), "pnl"].tolist()),
        axis=1,
    )
    symbol_summary["display_method"] = symbol_summary["method"].map(_display_name)
    symbol_summary.to_csv(report_dir / "symbol_method_summary.csv", index=False)

    paper_table = method_summary[method_summary["method"].isin(OVERALL_METHOD_ORDER)][["display_method", "pnl", "nd_pnl", "pnl_map", "profit_ratio", "sharpe"]].rename(
        columns={
            "display_method": "method",
            "pnl": "PnL",
            "nd_pnl": "ND-PnL",
            "pnl_map": "PnLMAP",
            "profit_ratio": "ProfitRatio",
            "sharpe": "Sharpe",
        }
    )
    paper_table.to_csv(report_dir / "continuous_paper_table.csv", index=False)
    _save_markdown_table(paper_table, report_dir / "continuous_paper_table.md")

    overall_summary, overall_formatted = _format_overall_results(combined)
    _save_overall_table(overall_summary, overall_formatted, report_dir)
    _save_runtime_table(out_dir, report_dir)
    _save_policy_diagnostics(combined, report_dir)

    sns.set_theme(style="whitegrid")
    metrics = ["nd_pnl", "pnl_map", "profit_ratio", "sharpe"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(axes.ravel(), metrics):
        plot_df = method_summary[method_summary["method"].isin(OVERALL_METHOD_ORDER)].copy()
        plot_df["display_method"] = plot_df["method"].map(_display_name)
        sns.barplot(data=plot_df, x="display_method", y=metric, errorbar=None, ax=axis)
        axis.tick_params(axis="x", rotation=45)
        axis.set_title(metric)
    fig.tight_layout()
    fig.savefig(report_dir / "paper_metrics_overview.png")
    plt.close(fig)

    _training_curves(out_dir, report_dir)
    _latency_plot(out_dir, report_dir)
    _ablation_plot(out_dir, report_dir)
    _decision_and_attention_plots(out_dir, report_dir)
    return report_dir


@pyrallis.wrap()
def main(config: ExperimentConfig) -> None:
    run_report(config)


if __name__ == "__main__":
    main()
