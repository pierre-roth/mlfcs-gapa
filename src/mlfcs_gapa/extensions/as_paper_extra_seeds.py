"""Extra seed runs for the AS-regularized paper comparison.

This runner is intentionally narrower than the full matched 400k sweep. It
only trains the methods reported in the paper with additional seed indices, so
the extension work stays separate from the paper-replication pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mlfcs_gapa.extensions.as_guided_panel import (
    ASGuidedPanelConfig,
    DEFAULT_STOCKS,
    run_as_guided_panel,
)
from mlfcs_gapa.extensions.as_matched_400k_sweep import MethodSpec
from mlfcs_gapa.paper.constants import PAPER


STOCKS = tuple(DEFAULT_STOCKS.split(","))
SEED_INDICES = (3, 4)


@dataclass(frozen=True)
class ExtraSeedSpec:
    method: MethodSpec
    stock: str
    seed_index: int

    @property
    def task_label(self) -> str:
        return f"{self.method.label}/{self.stock}/seed{self.seed_index}"


def method_specs() -> list[MethodSpec]:
    return [
        MethodSpec("paper_cppo", "paper_cppo", soft_penalty=0.0),
        MethodSpec("profit_ppo", "profit_ppo"),
        MethodSpec(
            "soft_as_low_risk_lam_0p10",
            "soft_as",
            soft_penalty=0.10,
            as_calibration="stock_risk_low",
        ),
    ]


def run_specs() -> list[ExtraSeedSpec]:
    return [
        ExtraSeedSpec(method, stock, seed_index)
        for method in method_specs()
        for stock in STOCKS
        for seed_index in SEED_INDICES
    ]


def config_for_spec(
    spec: ExtraSeedSpec,
    *,
    output_dir: Path,
    encoder_checkpoint: str | None,
    device: str,
    total_timesteps: int,
    n_envs: int,
    seed: int,
) -> ASGuidedPanelConfig:
    method = spec.method
    return ASGuidedPanelConfig(
        output_dir=output_dir / spec.task_label,
        variant=method.variant,
        label=method.label,
        stocks=(spec.stock,),
        total_timesteps=total_timesteps,
        agent_seeds=1,
        agent_seed_offset=spec.seed_index,
        n_envs=n_envs,
        seed=seed,
        soft_penalty=method.soft_penalty,
        hard_window_bias=method.hard_window_bias,
        hard_window_spread=method.hard_window_spread,
        as_gamma=method.as_gamma,
        as_kappa=method.as_kappa,
        as_calibration=method.as_calibration,
        eta=PAPER.eta_dampened_pnl,
        zeta=PAPER.zeta_inventory_penalty,
        encoder_checkpoint=encoder_checkpoint,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one extra-seed AS paper task.")
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-timesteps", type=int, default=400_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    specs = run_specs()
    if args.list:
        for index, spec in enumerate(specs):
            print(
                f"{index:03d} {spec.method.label} {spec.method.variant} "
                f"{spec.stock} seed{spec.seed_index}"
            )
        return
    if not 0 <= args.run_index < len(specs):
        raise SystemExit(f"run-index must be in [0, {len(specs) - 1}]")

    spec = specs[args.run_index]
    config = config_for_spec(
        spec,
        output_dir=args.output_dir,
        encoder_checkpoint=args.encoder_checkpoint,
        device=args.device,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    print(
        " ".join(
            [
                f"run_index={args.run_index}",
                f"label={spec.method.label}",
                f"variant={spec.method.variant}",
                f"stock={spec.stock}",
                f"seed_index={spec.seed_index}",
                f"output_dir={config.output_dir}",
            ]
        ),
        flush=True,
    )
    print(config, flush=True)
    metrics, _ = run_as_guided_panel(config)
    print(f"wrote extra seed task to {config.output_dir}", flush=True)
    print(metrics.group_by("method", "stock", "train_seed").agg(pl.len().alias("rows")), flush=True)


if __name__ == "__main__":
    main()
