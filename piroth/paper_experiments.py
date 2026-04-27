from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .config import DiagnosticsConfig
from .data_quality import assess_synthetic_quality
from .paper_evaluation import evaluate_paper_baselines
from .real_data import load_market_days
from .training import evaluate_trained_policy, train_dqn, train_ppo, train_pretrain_classifier
from .utils import ensure_dir, save_json
from .visualizer import build_synthetic_data_report


def run_paper_baseline_suite(config: DiagnosticsConfig) -> dict[str, object]:
    output_dir = ensure_dir(config.output_dir())
    train_days = load_market_days(config, "train")
    test_days = load_market_days(config, "test")
    summary = {
        "synthetic_quality": assess_synthetic_quality(test_days, config),
        "paper_baselines": evaluate_paper_baselines(train_days, test_days, config, output_dir),
    }
    save_json(output_dir / "paper_baseline_summary.json", summary)
    if config.create_plots:
        build_synthetic_data_report(test_days[: min(len(test_days), config.export_day_count)], config, output_dir / "visual_report")
    return summary


def run_latency_suite(config: DiagnosticsConfig, latencies: list[int] | None = None) -> pd.DataFrame:
    latencies = latencies or [0, 1, 5, 10, 20]
    rows = []
    for latency in latencies:
        trial = replace(config, latency=latency, run_name=f"{config.run_name}_latency_{latency}")
        summary = run_paper_baseline_suite(trial)
        for policy, metrics in summary["paper_baselines"].items():
            rows.append({"latency": latency, "policy": policy, **metrics})
    frame = pd.DataFrame(rows)
    root = ensure_dir(config.output_dir())
    frame.to_csv(root / "latency_suite.csv", index=False)
    return frame


def run_ablation_suite(config: DiagnosticsConfig, device: str = "cpu") -> pd.DataFrame:
    output_dir = ensure_dir(config.output_dir())
    train_days = load_market_days(config, "train")
    test_days = load_market_days(config, "test")
    model_dir = ensure_dir(output_dir / "models")
    pretrain_path = model_dir / "attnlob_pretrain.pt"
    if not pretrain_path.exists():
        pretrain_path = train_pretrain_classifier(train_days, config, model_dir, device=device)
    rows = []
    variants = [
        ("full", True, True, True, pretrain_path),
        ("w_o_lob_state", False, True, True, None),
        ("w_o_dynamic_state", True, False, True, pretrain_path),
        ("author_market_state_alias", True, True, True, pretrain_path),
        ("w_o_pretrain", True, True, True, None),
    ]
    for name, include_lob, include_market, include_agent, encoder_path in variants:
        variant_dir = ensure_dir(output_dir / "ablations" / name)
        variant_config = replace(
            config,
            run_name=f"{config.run_name}_{name}",
            include_lob_state=include_lob,
            include_market_state=include_market,
            include_agent_state=include_agent,
            author_market_state_alias=name == "author_market_state_alias",
        )
        ppo_path = train_ppo(train_days, variant_config, variant_dir, pretrain_path=encoder_path, device=device)
        ppo_frame = evaluate_trained_policy(test_days, variant_config, ppo_path, "ppo", variant_dir, device=device)
        rows.append({"variant": name, "agent": "C-PPO", **_mean_metrics(ppo_frame)})
        dqn_path = train_dqn(train_days, variant_config, variant_dir, pretrain_path=encoder_path, device=device)
        dqn_frame = evaluate_trained_policy(test_days, variant_config, dqn_path, "dqn", variant_dir, device=device)
        rows.append({"variant": name, "agent": "D-DQN", **_mean_metrics(dqn_frame)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ablation_suite.csv", index=False)
    return frame


def run_full_paper_suite(config: DiagnosticsConfig, device: str = "cpu") -> dict[str, object]:
    output_dir = ensure_dir(config.output_dir())
    train_days = load_market_days(config, "train")
    test_days = load_market_days(config, "test")
    for synthetic_day in (train_days + test_days)[: config.export_day_count]:
        synthetic_day.export(config.export_dir())
    quality = assess_synthetic_quality(test_days, config)
    baselines = evaluate_paper_baselines(train_days, test_days, config, output_dir)
    model_dir = ensure_dir(output_dir / "models")
    pretrain_path = train_pretrain_classifier(train_days, config, model_dir, device=device)
    ppo_path = train_ppo(train_days, config, model_dir, pretrain_path=pretrain_path, device=device)
    dqn_path = train_dqn(train_days, config, model_dir, pretrain_path=pretrain_path, device=device)
    ppo = evaluate_trained_policy(test_days, config, ppo_path, "ppo", output_dir, device=device)
    dqn = evaluate_trained_policy(test_days, config, dqn_path, "dqn", output_dir, device=device)
    report = None
    if config.create_plots:
        report = build_synthetic_data_report(test_days[: min(len(test_days), config.export_day_count)], config, output_dir / "visual_report")
    summary = {
        "config": asdict(config),
        "synthetic_quality": quality,
        "paper_baselines": baselines,
        "c_ppo": _mean_metrics(ppo),
        "d_dqn": _mean_metrics(dqn),
        "visual_report": str(report) if report is not None else "",
    }
    save_json(output_dir / "paper_suite_summary.json", summary)
    return summary


def _mean_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    columns = ["pnl", "nd_pnl", "pnl_map", "profit_ratio", "avg_abs_position", "avg_spread", "turnover", "fill_rate", "reward"]
    return {f"{column}_mean": float(frame[column].mean()) for column in columns if column in frame}
