from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

SYMBOLS = ["000001", "000858", "002415"]
DEFAULT_GPU_RUNS = {
    "000001": "piroth2_gpu_medium_000001_20260424_214500",
    "000858": "piroth2_gpu_medium_000858_20260424_214500",
    "002415": "piroth2_gpu_medium_002415_20260424_223500",
}
DEFAULT_BASELINE_RUNS = {
    "000001": "piroth2_baseline_medium_calibfast_000001_20260424_222500",
    "000858": "piroth2_baseline_medium_calibfast_000858_20260424_222500",
    "002415": "piroth2_baseline_medium_calibfast_002415_20260424_223500",
}
DEFAULT_LATENCY_RUNS = {
    symbol: f"piroth2_latency_fast_{symbol}_20260424_223500" for symbol in SYMBOLS
}
DEFAULT_ABLATION_PREFIX = "piroth2_ppo_ablation"
DEFAULT_VALIDATION_RUN = "piroth2_validation_post_speed_20260424_230000"
PPO_SEED_RUN_RE = re.compile(r"piroth2_ppo_seed(?P<seed>\d+)_(?P<symbol>\d+)_(?P<stamp>\d{8}_\d{6})")
BASELINE_SEED_RUN_RE = re.compile(r"piroth2_baseline_seed(?P<seed>\d+)_(?P<symbol>\d+)_(?P<stamp>\d{8}_\d{6})")
PPO_TUNED_RUN_RE = re.compile(r"piroth2_ppo_tuned_(?P<symbol>\d+)_seed(?P<seed>\d+)_(?P<stamp>\d{8}_\d{6})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect paper-replication result tables from artifact runs.")
    parser.add_argument("--root", type=Path, default=Path("artifacts_piroth2"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.root / "paper_results_summary"
    out.mkdir(parents=True, exist_ok=True)
    tables = collect_tables(args.root)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "summary.json").write_text(json.dumps({name: frame.to_dict("records") for name, frame in tables.items()}, indent=2) + "\n")
    (out / "summary.md").write_text(_markdown(tables), encoding="utf-8")
    (out / "index.html").write_text(_html(tables), encoding="utf-8")
    print(out / "summary.md")


def collect_tables(root: Path) -> dict[str, pd.DataFrame]:
    ppo_seed = _ppo_seed_table(root)
    paired_baselines = _paired_seed_baseline_table(root)
    ppo_tuned = _ppo_tuned_table(root)
    return {
        "medium_agents": _agent_table(root),
        "medium_baselines": _baseline_table(root),
        "latency": _latency_table(root),
        "ablations": _ablation_table(root),
        "ppo_seed_sweep": ppo_seed,
        "paired_seed_baselines": paired_baselines,
        "ppo_seed_vs_as": _ppo_vs_as_table(ppo_seed, paired_baselines),
        "ppo_tuned": ppo_tuned,
        "ppo_tuned_vs_as": _ppo_vs_as_table(ppo_tuned, paired_baselines),
        "synthetic_validation": _validation_table(root),
    }


def _agent_table(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, run_name in DEFAULT_GPU_RUNS.items():
        run = root / run_name
        for agent, filename in [("C-PPO", "ppo_episodes.csv"), ("D-DQN", "dqn_episodes.csv")]:
            path = run / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            rows.append({"symbol": symbol, "agent": agent, **_episode_means(frame)})
    return pd.DataFrame(rows)


def _baseline_table(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, run_name in DEFAULT_BASELINE_RUNS.items():
        path = root / run_name / "paper_baseline_summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for policy, metrics in payload["paper_baselines"].items():
            rows.append({"symbol": symbol, "agent": policy, **_select_metrics(metrics)})
    return pd.DataFrame(rows)


def _latency_table(root: Path) -> pd.DataFrame:
    frames = []
    for symbol, run_name in DEFAULT_LATENCY_RUNS.items():
        path = root / run_name / "latency_suite.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "symbol", symbol)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ablation_table(root: Path) -> pd.DataFrame:
    rows = []
    for variant in ["full", "no_lob", "no_dynamic", "no_pretrain"]:
        run = root / f"{DEFAULT_ABLATION_PREFIX}_{variant}_000858_20260424_224500"
        for agent, filename in [("C-PPO", "ppo_episodes.csv"), ("D-DQN", "dqn_episodes.csv")]:
            path = run / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            rows.append({"variant": variant, "agent": agent, **_episode_means(frame)})
    return pd.DataFrame(rows)


def _validation_table(root: Path) -> pd.DataFrame:
    path = root / DEFAULT_VALIDATION_RUN / "synthetic_validation_summary.json"
    if not path.exists():
        return pd.DataFrame()
    payload = json.loads(path.read_text())
    rows = [
        {
            "case_count": payload["case_count"],
            "pass_rate": payload["pass_rate"],
            "score_mean": payload["score_mean"],
            "score_min": payload["score_min"],
            "flags_total": payload["flags_total"],
        }
    ]
    return pd.DataFrame(rows)


def _ppo_seed_table(root: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(root.glob("piroth2_ppo_seed*_*_*")):
        match = PPO_SEED_RUN_RE.fullmatch(run.name)
        if match is None:
            continue
        path = run / "ppo_episodes.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        rows.append(
            {
                "symbol": match.group("symbol"),
                "seed": int(match.group("seed")),
                "stamp": match.group("stamp"),
                **_episode_means(frame),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["symbol", "seed", "stamp"], ignore_index=True)


def _paired_seed_baseline_table(root: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(root.glob("piroth2_baseline_seed*_*_*")):
        match = BASELINE_SEED_RUN_RE.fullmatch(run.name)
        if match is None:
            continue
        path = run / "paper_baseline_summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for policy, metrics in payload["paper_baselines"].items():
            rows.append(
                {
                    "symbol": match.group("symbol"),
                    "seed": int(match.group("seed")),
                    "stamp": match.group("stamp"),
                    "agent": policy,
                    **_select_metrics(metrics),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["symbol", "seed", "agent", "stamp"], ignore_index=True)


def _ppo_tuned_table(root: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(root.glob("piroth2_ppo_tuned_*_seed*_*")):
        match = PPO_TUNED_RUN_RE.fullmatch(run.name)
        if match is None:
            continue
        path = run / "ppo_episodes.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        rows.append(
            {
                "symbol": match.group("symbol"),
                "seed": int(match.group("seed")),
                "stamp": match.group("stamp"),
                **_episode_means(frame),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["symbol", "seed", "stamp"], ignore_index=True)


def _ppo_vs_as_table(ppo: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    if ppo.empty or baselines.empty:
        return pd.DataFrame()
    as_rows = baselines[baselines["agent"] == "AS"][
        ["symbol", "seed", "pnl_mean", "reward_mean", "fill_rate_mean", "trades_mean"]
    ].rename(
        columns={
            "pnl_mean": "as_pnl_mean",
            "reward_mean": "as_reward_mean",
            "fill_rate_mean": "as_fill_rate_mean",
            "trades_mean": "as_trades_mean",
        }
    )
    ppo_rows = ppo[
        ["symbol", "seed", "pnl_mean", "reward_mean", "fill_rate_mean", "trades_mean"]
    ].rename(
        columns={
            "pnl_mean": "ppo_pnl_mean",
            "reward_mean": "ppo_reward_mean",
            "fill_rate_mean": "ppo_fill_rate_mean",
            "trades_mean": "ppo_trades_mean",
        }
    )
    merged = ppo_rows.merge(as_rows, on=["symbol", "seed"], how="inner")
    if merged.empty:
        return merged
    merged["pnl_advantage"] = merged["ppo_pnl_mean"] - merged["as_pnl_mean"]
    merged["reward_advantage"] = merged["ppo_reward_mean"] - merged["as_reward_mean"]
    return merged[
        [
            "symbol",
            "seed",
            "ppo_pnl_mean",
            "as_pnl_mean",
            "pnl_advantage",
            "ppo_reward_mean",
            "as_reward_mean",
            "reward_advantage",
            "ppo_fill_rate_mean",
            "as_fill_rate_mean",
            "ppo_trades_mean",
            "as_trades_mean",
        ]
    ].sort_values(["symbol", "seed"], ignore_index=True)


def _episode_means(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "episodes": float(len(frame)),
        "pnl_mean": float(frame["pnl"].mean()),
        "reward_mean": float(frame["reward"].mean()),
        "fill_rate_mean": float(frame["fill_rate"].mean()),
        "trades_mean": float(frame["trades"].mean()),
        "avg_spread_mean": float(frame["avg_spread"].mean()),
    }


def _select_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "episodes": float(metrics.get("episodes", 0.0)),
        "pnl_mean": float(metrics.get("pnl_mean", 0.0)),
        "reward_mean": float(metrics.get("reward_mean", 0.0)),
        "fill_rate_mean": float(metrics.get("fill_rate_mean", 0.0)),
        "trades_mean": float(metrics.get("trades_mean", 0.0)),
        "avg_spread_mean": float(metrics.get("avg_spread_mean", 0.0)),
    }


def _markdown(tables: dict[str, pd.DataFrame]) -> str:
    parts = ["# Paper Replication Result Summary", ""]
    for name, frame in tables.items():
        parts.append(f"## {name.replace('_', ' ').title()}")
        parts.append("")
        parts.append(_frame_to_markdown(frame) if not frame.empty else "_missing_")
        parts.append("")
    return "\n".join(parts)


def _html(tables: dict[str, pd.DataFrame]) -> str:
    sections = []
    for name, frame in tables.items():
        title = name.replace("_", " ").title()
        if frame.empty:
            body = "<p class='missing'>missing</p>"
        else:
            body = frame.round(4).to_html(index=False, classes="result-table", border=0)
        sections.append(f"<section><h2>{title}</h2>{body}</section>")
    return "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Paper Replication Result Summary</title>",
            "<style>",
            "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:0;background:#f8fafc;color:#0f172a}",
            "header{padding:26px 34px;background:#111827;color:#fff}",
            "main{max-width:1360px;margin:0 auto;padding:26px 24px 48px}",
            "h1{font-size:27px;margin:0 0 6px}",
            "h2{font-size:19px;margin:0 0 12px}",
            "section{margin:0 0 28px;overflow-x:auto}",
            ".result-table{border-collapse:collapse;background:#fff;border:1px solid #e5e7eb;font-size:13px;min-width:720px}",
            ".result-table th{background:#eef2f7;color:#334155;font-weight:650}",
            ".result-table th,.result-table td{padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}",
            ".result-table th:first-child,.result-table td:first-child{text-align:left}",
            ".missing{color:#64748b;font-style:italic}",
            "</style></head><body>",
            "<header><h1>Paper Replication Result Summary</h1><div>Synthetic-data market-making experiments</div></header>",
            "<main>",
            *sections,
            "</main></body></html>",
        ]
    )


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[_format_cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]
    header = "| " + " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


if __name__ == "__main__":
    main()
