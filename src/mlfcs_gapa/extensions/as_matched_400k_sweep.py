"""Paper-faithful 400k matched AS-guided sweep.

Each indexed run is one stock and, for PPO methods, one training seed. This
keeps failures cheap to resubmit while preserving the original paper's
per-stock training setup.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mlfcs_gapa.extensions.as_guided_panel import (
    ASCalibrationName,
    ASGuidedPanelConfig,
    DEFAULT_STOCKS,
    VariantName,
    run_as_baseline_panel,
    run_as_guided_panel,
)
from mlfcs_gapa.paper.constants import PAPER


STOCKS = tuple(DEFAULT_STOCKS.split(","))
SEED_INDICES = (0, 1, 2)


@dataclass(frozen=True)
class MethodSpec:
    label: str
    variant: VariantName
    soft_penalty: float = 0.10
    as_calibration: ASCalibrationName = "stock_specific"
    as_gamma: float = 1.0
    as_kappa: float = 100.0
    hard_window_bias: float = 0.10
    hard_window_spread: float = 0.10


@dataclass(frozen=True)
class RunSpec:
    method: MethodSpec
    stock: str
    seed_index: int | None

    @property
    def task_label(self) -> str:
        if self.seed_index is None:
            return f"{self.method.label}/{self.stock}/as_baseline"
        return f"{self.method.label}/{self.stock}/seed{self.seed_index}"


def method_specs() -> list[MethodSpec]:
    return [
        MethodSpec("as_empirical_matched", "as_baseline"),
        MethodSpec("profit_ppo", "profit_ppo"),
        MethodSpec(
            "paper_cppo",
            "paper_cppo",
            as_calibration="stock_specific",
            soft_penalty=0.0,
        ),
        MethodSpec("soft_as_lam_0p03", "soft_as", soft_penalty=0.03),
        MethodSpec("soft_as_lam_0p05", "soft_as", soft_penalty=0.05),
        MethodSpec("soft_as_lam_0p10", "soft_as", soft_penalty=0.10),
        MethodSpec("soft_as_lam_0p125", "soft_as", soft_penalty=0.125),
        MethodSpec("soft_as_lam_0p20", "soft_as", soft_penalty=0.20),
        MethodSpec(
            "soft_as_low_risk_lam_0p10",
            "soft_as",
            soft_penalty=0.10,
            as_calibration="stock_risk_low",
        ),
        MethodSpec(
            "soft_as_high_risk_lam_0p10",
            "soft_as",
            soft_penalty=0.10,
            as_calibration="stock_risk_high",
        ),
        MethodSpec(
            "soft_as_fill_kappa_lam_0p10",
            "soft_as",
            soft_penalty=0.10,
            as_calibration="fill_kappa",
        ),
        MethodSpec(
            "soft_as_spread_kappa_lam_0p10",
            "soft_as",
            soft_penalty=0.10,
            as_calibration="spread_kappa",
        ),
        MethodSpec(
            "hard_as_w0p30",
            "hard_as",
            as_calibration="stock_specific",
            hard_window_bias=0.30,
            hard_window_spread=0.30,
        ),
    ]


def run_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for method in method_specs():
        if method.variant == "as_baseline":
            specs.extend(RunSpec(method, stock, None) for stock in STOCKS)
            continue
        for stock in STOCKS:
            specs.extend(RunSpec(method, stock, seed_index) for seed_index in SEED_INDICES)
    return specs


def config_for_spec(
    spec: RunSpec,
    *,
    output_dir: Path,
    encoder_checkpoint: str | None,
    device: str,
    total_timesteps: int,
    n_envs: int,
    seed: int,
) -> ASGuidedPanelConfig:
    method = spec.method
    seed_index = 0 if spec.seed_index is None else spec.seed_index
    return ASGuidedPanelConfig(
        output_dir=output_dir / spec.task_label,
        variant=method.variant,
        label=method.label,
        stocks=(spec.stock,),
        total_timesteps=0 if method.variant == "as_baseline" else total_timesteps,
        agent_seeds=1,
        agent_seed_offset=seed_index,
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
    parser = argparse.ArgumentParser(description="Run one indexed matched 400k AS sweep task.")
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
            seed = "baseline" if spec.seed_index is None else f"seed{spec.seed_index}"
            print(
                f"{index:03d} {spec.method.label} {spec.method.variant} "
                f"{spec.stock} {seed} as_calibration={spec.method.as_calibration}"
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
    if spec.method.variant == "as_baseline":
        metrics, _ = run_as_baseline_panel(config)
    else:
        metrics, _ = run_as_guided_panel(config)
    print(f"wrote matched 400k task to {config.output_dir}", flush=True)
    print(metrics.group_by("method", "stock", "train_seed").agg(pl.len().alias("rows")), flush=True)


if __name__ == "__main__":
    main()
