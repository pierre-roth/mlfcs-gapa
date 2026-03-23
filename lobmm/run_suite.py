from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd
import pyrallis

from .config import PretrainConfig, RLTrainConfig, SuiteConfig
from .evaluate import run_evaluation
from .pipeline import prepare_run
from .pretrain import run_pretrain
from .report import run_report
from .train_rl import evaluate_rl_model, load_trained_ppo, run_rl_training
from .utils import ensure_dir, save_json


def _config_kwargs(target_cls, config: SuiteConfig, **overrides):
    allowed = {field.name for field in fields(target_cls)}
    base = {key: value for key, value in asdict(config).items() if key in allowed}
    base.update(overrides)
    return base


def _run_latency_study(config: SuiteConfig) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    base_cfg = RLTrainConfig(**_config_kwargs(RLTrainConfig, config, algorithm="ppo", state_mode="full"))
    for symbol in config.symbols:
        from .pipeline import evaluate_baseline_policy, load_symbol_splits, standard_baselines
        from .env import MarketMakingEnv

        splits = load_symbol_splits(base_cfg, symbol)
        model = load_trained_ppo(base_cfg, symbol, splits["train"])
        symbol_results: dict[str, object] = {}
        for latency in config.latency_sweep:
            latency_cfg = RLTrainConfig(**_config_kwargs(RLTrainConfig, config, algorithm="ppo", state_mode="full", latency=latency))
            latency_cfg.apply_mode_defaults()
            ppo_envs = [
                MarketMakingEnv(day, latency_cfg, state_mode="full", wo_lob_state=False, wo_dynamic_state=False, reward_mode=latency_cfg.reward_mode)
                for day in splits["test"]
            ]
            ppo_runs = evaluate_rl_model(
                ppo_envs,
                model,
                latency_cfg,
                output_dir=Path(config.output_dir()) / symbol / "ppo" / "full" / f"latency_{latency}",
                method_name=f"PPO_full_latency_{latency}",
            )[0]
            records = [result.to_dict() for result in ppo_runs]
            for baseline in standard_baselines(latency_cfg):
                baseline_runs = evaluate_baseline_policy(baseline, splits["test"], latency_cfg, latency=latency)[0]
                records.extend(result.to_dict() for result in baseline_runs)
            frame = pd.DataFrame(records)
            latency_dir = ensure_dir(Path(config.output_dir()) / symbol / "latency")
            frame.to_csv(latency_dir / f"latency_{latency}.csv", index=False)
            symbol_results[str(latency)] = {
                method: metrics
                for method, metrics in frame.groupby("method")[["pnl", "nd_pnl", "pnl_map", "profit_ratio"]].mean().round(6).to_dict("index").items()
            }
        results[symbol] = symbol_results
    return results


def run_suite(config: SuiteConfig) -> None:
    config.apply_mode_defaults()
    out_dir = prepare_run(config)
    summary: dict[str, object] = {}
    if config.run_pretrain:
        summary["pretrain"] = run_pretrain(PretrainConfig(**_config_kwargs(PretrainConfig, config)))
    if config.run_ablations:
        summary["pretrain_simple"] = run_pretrain(
            PretrainConfig(
                **_config_kwargs(
                    PretrainConfig,
                    config,
                    pretrain_backbone="simple",
                    save_backbone_name="simple_lob.pt",
                )
            )
        )
    if config.run_main_agents:
        summary["ppo"] = run_rl_training(RLTrainConfig(**_config_kwargs(RLTrainConfig, config, algorithm="ppo", state_mode="full")))
    if config.run_rl_baselines:
        summary["inventory_rl"] = run_rl_training(RLTrainConfig(**_config_kwargs(RLTrainConfig, config, algorithm="ppo", state_mode="inventory_only", wo_lob_state=True, wo_dynamic_state=True)))
        summary["handcrafted_rl"] = run_rl_training(RLTrainConfig(**_config_kwargs(RLTrainConfig, config, algorithm="ppo", state_mode="handcrafted", wo_lob_state=True, wo_dynamic_state=True)))
    if config.run_non_rl_baselines:
        summary["non_rl"] = run_evaluation(RLTrainConfig(**_config_kwargs(RLTrainConfig, config)))
    if config.run_ablations:
        summary["ablations"] = {
            "wo_lob": run_rl_training(RLTrainConfig(**_config_kwargs(RLTrainConfig, config, algorithm="ppo", state_mode="full", wo_lob_state=True))),
            "wo_dynamic": run_rl_training(RLTrainConfig(**_config_kwargs(RLTrainConfig, config, algorithm="ppo", state_mode="full", wo_dynamic_state=True))),
            "simple_backbone": run_rl_training(RLTrainConfig(**_config_kwargs(RLTrainConfig, config, algorithm="ppo", state_mode="full", pretrain_backbone="simple", backbone_name="simple_lob.pt", variant_tag="wo_attn"))),
        }
    if config.run_latency:
        summary["latency"] = _run_latency_study(config)
    if config.run_report:
        summary["report_dir"] = str(run_report(config))
    save_json(out_dir / "suite_summary.json", summary)


@pyrallis.wrap()
def main(config: SuiteConfig) -> None:
    run_suite(config)


if __name__ == "__main__":
    main()
