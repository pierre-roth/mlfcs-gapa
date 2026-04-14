from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyrallis
import torch
from sklearn.linear_model import LogisticRegression, Ridge

from .config import RLTrainConfig
from .data import DayData
from .env import MarketMakingEnv
from .pipeline import evaluate_baseline_policy, load_symbol_splits, prepare_run, save_episode_results, standard_baselines, summarize_results
from .train_rl import evaluate_rl_model, load_trained_ppo
from .utils import ensure_dir, save_json


def _backbone_feature_table(model, days: list[DayData], config: RLTrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats = []
    regimes = []
    alphas = []
    model.to(config.device)
    model.eval()
    with torch.no_grad():
        for day in days:
            assert day.normalized_lob is not None
            for idx in day.valid_label_indices(config.lookback, config.pretrain_horizon):
                start = idx - config.lookback + 1
                lob = torch.tensor(day.normalized_lob[start : idx + 1][None, :, :], dtype=torch.float32, device=config.device)
                z = model.encoder.backbone.features(lob).squeeze(0).cpu().numpy()
                feats.append(z)
                regimes.append(int(day.latent.iloc[idx]["regime"]) + 1)
                alphas.append(float(day.latent.iloc[idx]["latent_alpha"]))
    return np.asarray(feats), np.asarray(regimes), np.asarray(alphas)


def _probe_metrics(model, train_days: list[DayData], test_days: list[DayData], config: RLTrainConfig) -> dict[str, float]:
    x_train, y_regime_train, y_alpha_train = _backbone_feature_table(model, train_days, config)
    x_test, y_regime_test, y_alpha_test = _backbone_feature_table(model, test_days, config)
    if len(x_train) == 0 or len(x_test) == 0 or len(np.unique(y_regime_train)) < 2:
        return {"regime_probe_accuracy": 0.0, "alpha_probe_r2": 0.0}
    regime_probe = LogisticRegression(max_iter=500).fit(x_train, y_regime_train)
    alpha_probe = Ridge(alpha=1.0).fit(x_train, y_alpha_train)
    return {
        "regime_probe_accuracy": float(regime_probe.score(x_test, y_regime_test)),
        "alpha_probe_r2": float(alpha_probe.score(x_test, y_alpha_test)),
    }


def _attention_alignment(trace_dir: Path) -> dict[str, float]:
    shift_scores = []
    recent_scores = []
    for trace_path in sorted(trace_dir.glob("episode_*.csv")):
        if trace_path.name.endswith("_attention.csv"):
            continue
        attention_path = trace_path.with_name(trace_path.stem + "_attention.csv")
        if not attention_path.exists():
            continue
        trace = pd.read_csv(trace_path)
        attn = pd.read_csv(attention_path).to_numpy(dtype=np.float64)
        if attn.size == 0 or trace.empty:
            continue
        recent_mass = attn[:, -10:].sum(axis=1) / np.maximum(attn.sum(axis=1), 1e-8)
        recent_scores.extend(recent_mass.tolist())
        valid = trace["regime_shift"].to_numpy(dtype=np.int64)
        if len(valid) == len(recent_mass):
            shift_scores.extend(recent_mass[valid > 0].tolist())
    return {
        "attention_recent_mass": float(np.mean(recent_scores)) if recent_scores else 0.0,
        "attention_recent_mass_at_regime_shift": float(np.mean(shift_scores)) if shift_scores else 0.0,
    }


def _counterfactual_sensitivity(model, test_days: list[DayData], config: RLTrainConfig, max_samples: int = 128) -> dict[str, float]:
    deltas = []
    model.to(config.device)
    model.eval()
    with torch.no_grad():
        for day in test_days:
            assert day.normalized_lob is not None
            indices = day.valid_label_indices(config.lookback, config.pretrain_horizon)[:max_samples]
            for idx in indices:
                start = idx - config.lookback + 1
                lob = day.normalized_lob[start : idx + 1].copy()
                flat = np.concatenate([day.dynamic[idx], day.agent_template[idx]]).astype(np.float32)
                up = lob.copy()
                down = lob.copy()
                price_cols = [col for col in range(up.shape[1]) if col % 4 in {0, 2}]
                up[:, price_cols] += config.tick_size
                down[:, price_cols] -= config.tick_size
                up_t = torch.tensor(up[None, :, :], dtype=torch.float32, device=config.device)
                down_t = torch.tensor(down[None, :, :], dtype=torch.float32, device=config.device)
                flat_t = torch.tensor(flat[None, :], dtype=torch.float32, device=config.device)
                up_dist, _ = model.dist_value(up_t, flat_t)
                down_dist, _ = model.dist_value(down_t, flat_t)
                deltas.append(float(up_dist.mean[0, 0] - down_dist.mean[0, 0]))
    return {"counterfactual_action1_shift": float(np.mean(deltas)) if deltas else 0.0}


def run_report(config: RLTrainConfig) -> dict[str, dict[str, float]]:
    config.apply_mode_defaults()
    out_dir = prepare_run(config, label="report")
    report_summary: dict[str, dict[str, float]] = {}
    for symbol in config.symbols:
        splits = load_symbol_splits(config, symbol)
        model = load_trained_ppo(config, symbol, splits["train"])
        symbol_dir = ensure_dir(Path(out_dir) / symbol / "report")
        ppo_results, _ = evaluate_rl_model([MarketMakingEnv(day, config) for day in splits["test"]], model, config, output_dir=symbol_dir / "ppo", method_name=config.method_name())
        ppo_frame = save_episode_results(symbol_dir / "ppo_episodes.csv", ppo_results)
        baseline_summary: dict[str, dict[str, float]] = {}
        for baseline in standard_baselines(config):
            results, runtime = evaluate_baseline_policy(baseline, splits["test"], config)
            frame = save_episode_results(symbol_dir / f"{baseline.name}_episodes.csv", results)
            baseline_summary[baseline.name] = {**summarize_results(frame), **runtime}
        diagnostics = {
            **_probe_metrics(model, splits["train"], splits["test"], config),
            **_attention_alignment(symbol_dir / "ppo" / "traces"),
            **_counterfactual_sensitivity(model, splits["test"], config),
        }
        summary = {
            **summarize_results(ppo_frame),
            **diagnostics,
            "oracle_pnl_mean": float(baseline_summary.get("OraclePaper", {}).get("pnl_mean", 0.0)),
            "fixed1_pnl_mean": float(baseline_summary.get("Fixed_1", {}).get("pnl_mean", 0.0)),
        }
        save_json(symbol_dir / "summary.json", summary)
        save_json(symbol_dir / "baselines.json", baseline_summary)
        report_summary[symbol] = summary
    save_json(Path(out_dir) / "report_summary.json", report_summary)
    return report_summary


@pyrallis.wrap()
def main(config: RLTrainConfig) -> None:
    run_report(config)


if __name__ == "__main__":
    main()
