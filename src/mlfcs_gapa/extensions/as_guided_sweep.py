"""Thirty-two run AS-guided extension sweep.

This module is intentionally separate from the paper replication entry points.
It defines a stable index-to-configuration map for the next AS-guided study.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import polars as pl

from mlfcs_gapa.extensions.as_guided_panel import ASGuidedPanelConfig, run_as_guided_panel
from mlfcs_gapa.paper.constants import PAPER


def sweep_configs(
    *,
    output_dir: Path,
    encoder_checkpoint: str | None,
    device: str,
    total_timesteps: int,
    agent_seeds: int,
    n_envs: int,
) -> list[ASGuidedPanelConfig]:
    base = ASGuidedPanelConfig(
        output_dir=output_dir,
        variant="soft_as",
        label="base",
        total_timesteps=total_timesteps,
        agent_seeds=agent_seeds,
        n_envs=n_envs,
        encoder_checkpoint=encoder_checkpoint,
        device=device,
    )

    configs: list[ASGuidedPanelConfig] = [
        _cfg(base, "soft_lam_0p03", soft_penalty=0.03),
        _cfg(base, "soft_lam_0p05", soft_penalty=0.05),
        _cfg(base, "soft_lam_0p075", soft_penalty=0.075),
        _cfg(base, "soft_lam_0p125", soft_penalty=0.125),
        _cfg(base, "soft_lam_0p15", soft_penalty=0.15),
        _cfg(base, "soft_lam_0p20", soft_penalty=0.20),
        _cfg(base, "soft_lam_0p30", soft_penalty=0.30),
        _cfg(base, "soft_lam_0p50", soft_penalty=0.50),
        _cfg(base, "soft_spread_only", soft_penalty=0.10, bias_weight=0.0),
        _cfg(base, "soft_bias_only", soft_penalty=0.10, spread_weight=0.0),
        _cfg(base, "soft_weight_spread", soft_penalty=0.10, bias_weight=0.05, spread_weight=0.15),
        _cfg(base, "soft_weight_bias", soft_penalty=0.10, bias_weight=0.15, spread_weight=0.05),
        _cfg(base, "soft_huber", soft_penalty=0.10, penalty_norm="huber", huber_delta=0.10),
        _cfg(base, "soft_l1", soft_penalty=0.10, penalty_norm="l1"),
        _cfg(base, "soft_quote_space", soft_penalty=0.10, penalty_space="quote"),
        _cfg(base, "soft_adaptive_l2", soft_penalty=0.10, penalty_norm="adaptive_l2"),
        _cfg(
            base,
            "soft_episode_decay",
            soft_penalty=0.30,
            soft_penalty_end=0.05,
            penalty_schedule="episode_decay",
        ),
        _cfg(
            base,
            "soft_episode_warmup",
            soft_penalty=0.0,
            soft_penalty_end=0.10,
            penalty_schedule="episode_warmup",
        ),
        _cfg(base, "soft_as_gamma_0p10", soft_penalty=0.10, as_gamma=0.10),
        _cfg(base, "soft_as_gamma_0p30", soft_penalty=0.10, as_gamma=0.30),
        _cfg(base, "soft_as_gamma_2p00", soft_penalty=0.10, as_gamma=2.00),
        _cfg(base, "soft_as_kappa_25", soft_penalty=0.10, as_kappa=25.0),
        _cfg(base, "soft_as_kappa_50", soft_penalty=0.10, as_kappa=50.0),
        _cfg(base, "soft_as_kappa_200", soft_penalty=0.10, as_kappa=200.0),
        _cfg(base, "soft_as_empirical_kappa", soft_penalty=0.10, as_calibration="empirical_kappa"),
        _cfg(base, "soft_as_stock_specific", soft_penalty=0.10, as_calibration="stock_specific"),
        _cfg(
            base,
            "hard_as_w0p30",
            variant="hard_as",
            hard_window_bias=0.30,
            hard_window_spread=0.30,
        ),
        _cfg(
            base,
            "hard_as_bias0p30_spread0p10",
            variant="hard_as",
            hard_window_bias=0.30,
            hard_window_spread=0.10,
        ),
        _cfg(
            base,
            "hard_as_bias0p10_spread0p30",
            variant="hard_as",
            hard_window_bias=0.10,
            hard_window_spread=0.30,
        ),
        _cfg(base, "profit_ppo_control", variant="profit_ppo"),
        _cfg(
            base,
            "paper_reward_eta0p25_zeta0p005",
            variant="paper_cppo",
            eta=0.25,
            zeta=0.005,
        ),
        _cfg(
            base,
            "paper_cppo_long_400k",
            variant="paper_cppo",
            total_timesteps=max(total_timesteps * 2, 400_000),
            eta=PAPER.eta_dampened_pnl,
            zeta=PAPER.zeta_inventory_penalty,
        ),
    ]
    return [replace(config, output_dir=output_dir / config.label) for config in configs]


def _cfg(base: ASGuidedPanelConfig, label: str, **updates) -> ASGuidedPanelConfig:
    return replace(base, label=label, **updates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one indexed AS-guided sweep config.")
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--agent-seeds", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    configs = sweep_configs(
        output_dir=args.output_dir,
        encoder_checkpoint=args.encoder_checkpoint,
        device=args.device,
        total_timesteps=args.total_timesteps,
        agent_seeds=args.agent_seeds,
        n_envs=args.n_envs,
    )
    if args.list:
        for index, config in enumerate(configs):
            print(f"{index:02d} {config.label} {config.variant}")
        return
    if not 0 <= args.run_index < len(configs):
        raise SystemExit(f"run-index must be in [0, {len(configs) - 1}]")

    config = configs[args.run_index]
    print(f"run_index={args.run_index} label={config.label} variant={config.variant}", flush=True)
    print(config, flush=True)
    metrics, _ = run_as_guided_panel(config)
    print(f"wrote AS-guided sweep run to {config.output_dir}", flush=True)
    print(metrics.group_by("method").agg(pl.len().alias("rows")), flush=True)


if __name__ == "__main__":
    main()
