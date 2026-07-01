"""Appendix diagnostics for the AS-regularized extension paper.

The routines here deliberately live under ``extensions``. They aggregate
extension artifacts and run direct-AS evaluation variants without touching the
paper-replication pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from stable_baselines3 import PPO

from mlfcs_gapa.data.features import normalize_lob_window
from mlfcs_gapa.data.schema import LobDataset, lob_columns
from mlfcs_gapa.env.baselines import estimate_episode_volatility
from mlfcs_gapa.extensions.as_guided_panel import (
    ASGuidedPanelConfig,
    DEFAULT_STOCKS,
    PPO_LOG_STD_INIT,
    SYNTHETIC_STOCK_BASE_PRICES,
    _build_panel,
    _estimate_kappa_from_fill_decay,
    _estimate_kappa_from_l1_spread,
    _filter_stable_windows,
    _inventory_target_gamma,
    _merge_lob_datasets,
    _scale_gamma,
    _stock_specific_gamma,
    run_as_baseline_panel,
)
from mlfcs_gapa.paper.constants import PAPER
from mlfcs_gapa.training.ppo import AttnLOBFeatureExtractor


MAIN_METHODS = ("paper_cppo", "profit_ppo", "soft_as_low_risk_lam_0p10")
STOCKS = tuple(DEFAULT_STOCKS.split(","))
METHOD_LABELS = {
    "paper_cppo": "C-PPO",
    "profit_ppo": "Profit PPO",
    "soft_as_low_risk_lam_0p10": "Soft AS",
    "as_empirical_matched": "Direct AS\nstock-specific",
    "as_inventory_target": "Direct AS\ninventory-target",
}


def run_appendix_diagnostics(
    *,
    matched_root: Path,
    strengthening_root: Path,
    paper_dir: Path,
    artifact_dir: Path,
    device: str,
    seed: int,
) -> None:
    tables_dir = paper_dir / "tables"
    figs_dir = paper_dir / "figs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    calibration = calibration_comparison(seed=seed)
    calibration.write_csv(tables_dir / "as_calibration_comparison.csv")

    inventory_target_metrics, inventory_target_trades = run_inventory_target_as(
        artifact_dir=artifact_dir,
        seed=seed,
    )
    inventory_target_metrics.write_csv(tables_dir / "direct_as_inventory_target_metrics.csv")
    inventory_target_trades.write_csv(tables_dir / "direct_as_inventory_target_trade_summary.csv")

    teacher_quality = teacher_quality_table(
        matched_root=matched_root,
        inventory_target_metrics=inventory_target_metrics,
        inventory_target_trades=inventory_target_trades,
    )
    teacher_quality.write_csv(tables_dir / "teacher_quality_by_stock.csv")

    tail_episodes, tail_summary = tail_risk_diagnostics(
        matched_root=matched_root,
        strengthening_root=strengthening_root,
    )
    tail_episodes.write_csv(tables_dir / "tail_risk_episodes.csv")
    tail_summary.write_csv(tables_dir / "tail_risk_summary.csv")

    divergence = teacher_divergence_table(strengthening_root=stage_path(strengthening_root))
    divergence.write_csv(tables_dir / "teacher_divergence_by_method_latency.csv")

    synthetic = synthetic_validation(seed=seed)
    synthetic.write_csv(tables_dir / "synthetic_validation_summary.csv")

    attention = attention_diagnostics(
        matched_root=matched_root,
        stock="000858",
        seed_index=2,
        device=device,
        seed=seed,
    )
    attention.write_csv(tables_dir / "attention_window_diagnostics.csv")

    plot_calibration_comparison(calibration, figs_dir / "fig_as_calibration_comparison.png")
    plot_teacher_quality(teacher_quality, figs_dir / "fig_teacher_quality.png")
    plot_tail_risk(tail_summary, figs_dir / "fig_tail_risk.png")
    plot_synthetic_validation(synthetic, figs_dir / "fig_synthetic_validation.png")
    plot_attention_diagnostics(attention, figs_dir / "fig_attention_diagnostics.png")


def calibration_comparison(*, seed: int) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for stock, base_price in SYNTHETIC_STOCK_BASE_PRICES.items():
        train_dataset = _training_dataset(stock, base_price, seed=seed)
        spread = (
            train_dataset.orderbook["ask1_price"].to_numpy()
            - train_dataset.orderbook["bid1_price"].to_numpy()
        )
        sigma = estimate_episode_volatility(train_dataset, PAPER.episode_events)
        stock_gamma = _stock_specific_gamma(train_dataset, stock=stock)
        spread_kappa = _estimate_kappa_from_l1_spread(train_dataset)
        fill_kappa = _estimate_kappa_from_fill_decay(train_dataset)
        inventory_gamma = _inventory_target_gamma(train_dataset)
        definitions = [
            ("stock_specific", stock_gamma, spread_kappa, "spread kappa + volatility gamma"),
            (
                "stock_risk_low",
                _scale_gamma(stock_gamma, 0.5),
                spread_kappa,
                "paper-selected half-gamma teacher",
            ),
            ("fill_kappa", stock_gamma, fill_kappa, "fill-decay kappa + volatility gamma"),
            (
                "inventory_target",
                inventory_gamma,
                fill_kappa,
                "fill-decay kappa + half-cap spread-skew gamma",
            ),
        ]
        for calibration, gamma, kappa, description in definitions:
            rows.append(
                {
                    "stock": stock,
                    "as_calibration": calibration,
                    "as_gamma": gamma,
                    "as_kappa": kappa,
                    "episode_sigma": sigma,
                    "mean_half_spread": float(spread.mean()) / 2.0,
                    "target_inventory_lots": PAPER.omega_inventory_units / 2.0,
                    "description": description,
                }
            )
    return pl.DataFrame(rows)


def run_inventory_target_as(
    *,
    artifact_dir: Path,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    output_dir = artifact_dir / "direct_as_inventory_target"
    config = ASGuidedPanelConfig(
        output_dir=output_dir,
        variant="as_baseline",
        label="as_inventory_target",
        stocks=STOCKS,
        total_timesteps=0,
        agent_seeds=1,
        n_envs=1,
        seed=seed,
        as_calibration="inventory_target",
        device="cpu",
    )
    metrics, trades = run_as_baseline_panel(config)
    trade_summary = _trade_summary(trades)
    return metrics, trade_summary


def teacher_quality_table(
    *,
    matched_root: Path,
    inventory_target_metrics: pl.DataFrame,
    inventory_target_trades: pl.DataFrame,
) -> pl.DataFrame:
    combined_metrics = _read_csv(matched_root / "summary_tables" / "combined_extension_metrics.csv")
    combined_trades = _read_csv(matched_root / "summary_tables" / "combined_trade_diagnostics.csv")
    direct_metrics = combined_metrics.filter(pl.col("method") == "as_empirical_matched")
    direct_trades = combined_trades.filter(pl.col("method") == "as_empirical_matched")
    inventory_target_trades = inventory_target_trades.with_columns(
        pl.lit("as_inventory_target").alias("method"),
        pl.lit("as_baseline").alias("variant"),
        pl.lit(-1).alias("train_seed"),
    )

    metrics = pl.concat(
        [
            direct_metrics.select(inventory_target_metrics.columns),
            inventory_target_metrics,
        ],
        how="vertical_relaxed",
    ).with_columns(_stock_expr())
    trades = pl.concat(
        [
            direct_trades.select(inventory_target_trades.columns),
            inventory_target_trades,
        ],
        how="vertical_relaxed",
    ).with_columns(_stock_expr())

    metric_summary = metrics.group_by("method", "variant", "stock", "as_calibration").agg(
        pl.col("as_gamma").mean().alias("as_gamma"),
        pl.col("as_kappa").mean().alias("as_kappa"),
        pl.col("pnl").mean().alias("pnl_mean"),
        pl.col("nd_pnl").mean().alias("nd_pnl_mean"),
        pl.col("sharpe").mean().alias("sharpe_mean"),
        pl.col("mean_abs_inventory").mean().alias("mean_abs_inventory"),
        pl.col("mean_quoted_spread").mean().alias("mean_quoted_spread"),
        pl.col("buy_notional").mean().alias("buy_notional"),
    )
    trade_summary = trades.group_by("method", "stock").agg(
        pl.col("fills").mean().alias("fills_per_episode"),
        pl.col("abs_volume").mean().alias("abs_volume_per_episode"),
        pl.col("max_abs_inventory").mean().alias("max_abs_inventory_mean"),
    )
    return (
        metric_summary.join(trade_summary, on=["method", "stock"], how="left")
        .sort("method", "stock")
    )


def tail_risk_diagnostics(
    *,
    matched_root: Path,
    strengthening_root: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    paths: list[Path] = []
    for method in MAIN_METHODS:
        paths.append(matched_root / method / "extension_trades.parquet")
        paths.extend((strengthening_root / "extra-seeds" / method).rglob("extension_trades.parquet"))
    trades = pl.concat([pl.read_parquet(path) for path in paths if path.exists()], how="diagonal_relaxed")
    trades = trades.filter(pl.col("method").is_in(MAIN_METHODS)).with_columns(_stock_expr())

    group_cols = ["method", "variant", "stock", "train_seed", "day", "episode_id"]
    rows: list[dict[str, object]] = []
    for key, episode in trades.partition_by(group_cols, as_dict=True, maintain_order=True).items():
        values = episode["value"].to_numpy().astype(float)
        inventories = episode["inventory"].to_numpy().astype(float)
        trade_volume = episode["trade_volume"].to_numpy().astype(float)
        running_peak = np.maximum.accumulate(values)
        drawdown = running_peak - values
        row = dict(zip(group_cols, key, strict=True))
        row.update(
            {
                "final_value": float(values[-1]),
                "min_value": float(values.min()),
                "max_drawdown": float(drawdown.max()),
                "max_abs_inventory": float(np.abs(inventories).max()),
                "mean_abs_inventory": float(np.abs(inventories).mean()),
                "fills": int(np.count_nonzero(trade_volume)),
                "abs_volume": float(np.abs(trade_volume).sum()),
                "closing_abs_volume": float(abs(trade_volume[-1])),
                "touched_inventory_cap": bool(np.abs(inventories).max() >= PAPER.max_inventory),
            }
        )
        rows.append(row)
    episodes = pl.DataFrame(rows)
    summary = episodes.group_by("method", "variant").agg(
        pl.len().alias("episodes"),
        pl.col("final_value").mean().alias("pnl_mean"),
        pl.col("final_value").quantile(0.05).alias("pnl_p05"),
        pl.col("final_value").min().alias("pnl_min"),
        (pl.col("final_value") < 0.0).mean().alias("negative_episode_rate"),
        pl.col("max_drawdown").mean().alias("max_drawdown_mean"),
        pl.col("max_drawdown").quantile(0.95).alias("max_drawdown_p95"),
        pl.col("max_abs_inventory").mean().alias("max_abs_inventory_mean"),
        pl.col("max_abs_inventory").quantile(0.95).alias("max_abs_inventory_p95"),
        pl.col("closing_abs_volume").mean().alias("closing_abs_volume_mean"),
        pl.col("touched_inventory_cap").mean().alias("inventory_cap_touch_rate"),
    )
    return episodes.sort(group_cols), summary.sort("method")


def teacher_divergence_table(*, strengthening_root: Path) -> pl.DataFrame:
    paths = list((strengthening_root / "diagnostics").rglob("teacher_diagnostics.csv"))
    diagnostics = pl.concat([pl.read_csv(path) for path in paths], how="vertical_relaxed")
    diagnostics = diagnostics.with_columns(_stock_expr())
    return diagnostics.group_by("method", "variant", "latency_events").agg(
        pl.col("teacher_l2_diff_mean").mean().alias("teacher_l2_diff_mean"),
        pl.col("teacher_abs_bias_diff_mean").mean().alias("teacher_abs_bias_diff_mean"),
        pl.col("teacher_abs_spread_diff_mean").mean().alias("teacher_abs_spread_diff_mean"),
        pl.col("fills").mean().alias("fills_per_episode"),
        pl.col("max_abs_inventory").mean().alias("max_abs_inventory_mean"),
    ).sort("method", "latency_events")


def synthetic_validation(*, seed: int) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for stock, base_price in SYNTHETIC_STOCK_BASE_PRICES.items():
        panel = _build_panel(
            stocks=[(stock, base_price)],
            day_indices=range(21),
            events_per_day=10_000,
            seed=seed,
        )
        dataset = _merge_lob_datasets([entry[2] for entry in panel], day="all")
        ask1 = dataset.orderbook["ask1_price"].to_numpy().astype(float)
        bid1 = dataset.orderbook["bid1_price"].to_numpy().astype(float)
        ask_vol = dataset.orderbook["ask1_volume"].to_numpy().astype(float)
        bid_vol = dataset.orderbook["bid1_volume"].to_numpy().astype(float)
        mid = (ask1 + bid1) / 2.0
        changes = np.diff(mid)
        autocorr = float(np.corrcoef(changes[:-1], changes[1:])[0, 1]) if len(changes) > 2 else 0.0
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-7)
        trade_volume = dataset.trades["trade_volume_total"].to_numpy().astype(float)
        rows.append(
            {
                "stock": stock,
                "events": dataset.orderbook.height,
                "mean_spread": float((ask1 - bid1).mean()),
                "median_spread": float(np.median(ask1 - bid1)),
                "episode_sigma": estimate_episode_volatility(dataset, PAPER.episode_events),
                "event_mid_change_sd": float(np.std(changes)),
                "event_mid_change_autocorr_lag1": autocorr,
                "trade_event_rate": float(np.mean(trade_volume > 0.0)),
                "mean_trade_volume": float(trade_volume.mean()),
                "mean_abs_l1_imbalance": float(np.abs(imbalance).mean()),
                "mean_l1_depth": float((ask_vol + bid_vol).mean()),
            }
        )
    return pl.DataFrame(rows)


def attention_diagnostics(
    *,
    matched_root: Path,
    stock: str,
    seed_index: int,
    device: str,
    seed: int,
) -> pl.DataFrame:
    model_path = (
        matched_root
        / "soft_as_low_risk_lam_0p10"
        / stock
        / f"seed{seed_index}"
        / stock
        / f"soft_as_low_risk_lam_0p10_seed{seed_index}"
        / "ppo_model.zip"
    )
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = PPO.load(
        model_path,
        device=device,
        custom_objects={
            "policy_kwargs": {
                "features_extractor_class": AttnLOBFeatureExtractor,
                "features_extractor_kwargs": {
                    "encoder_checkpoint": None,
                    "freeze_encoder": False,
                },
                "log_std_init": PPO_LOG_STD_INIT,
            }
        },
    )
    encoder = model.policy.features_extractor.lob_encoder
    encoder.eval()
    torch_device = next(encoder.parameters()).device

    dataset = _attention_dataset(stock, seed=seed)
    candidates = _attention_candidate_windows(dataset)
    rows: list[dict[str, object]] = []
    for label, end in candidates:
        start = end - PAPER.window_length + 1
        lob_values = dataset.orderbook.select(lob_columns()).slice(start, PAPER.window_length)
        window = normalize_lob_window(lob_values.to_numpy())
        with torch.no_grad():
            _, weights = encoder(
                torch.from_numpy(window).float().unsqueeze(0).to(torch_device),
                return_attention_weights=True,
            )
        weights_np = weights.squeeze(0).detach().cpu().numpy()
        mass = weights_np.mean(axis=0)
        mass = mass / max(float(mass.sum()), 1e-12)
        rows.append(
            {
                "stock": stock,
                "seed_index": seed_index,
                "criterion": label,
                "window_end": end,
                **_window_market_stats(dataset, start=start, end=end),
                "attention_entropy": _normalized_entropy(mass),
                "attention_recent_5_mass": float(mass[-5:].sum()),
                "attention_recent_10_mass": float(mass[-10:].sum()),
                "attention_max_mass": float(mass.max()),
                "attention_top_lag": int((PAPER.window_length - 1) - int(mass.argmax())),
            }
        )
    return pl.DataFrame(rows)


def _attention_candidate_windows(dataset: LobDataset) -> list[tuple[str, int]]:
    ask1 = dataset.orderbook["ask1_price"].to_numpy().astype(float)
    bid1 = dataset.orderbook["bid1_price"].to_numpy().astype(float)
    ask_vol = dataset.orderbook["ask1_volume"].to_numpy().astype(float)
    bid_vol = dataset.orderbook["bid1_volume"].to_numpy().astype(float)
    trade_volume = dataset.trades["trade_volume_total"].to_numpy().astype(float)
    mid = (ask1 + bid1) / 2.0
    spread = ask1 - bid1
    imbalance = np.abs((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-7))
    ends = np.arange(PAPER.window_length - 1, len(mid))

    stats = {
        "low_mid_std": np.array([mid[end - PAPER.window_length + 1 : end + 1].std() for end in ends]),
        "high_mid_std": np.array([mid[end - PAPER.window_length + 1 : end + 1].std() for end in ends]),
        "high_spread": np.array([spread[end - PAPER.window_length + 1 : end + 1].mean() for end in ends]),
        "high_imbalance": np.array(
            [imbalance[end - PAPER.window_length + 1 : end + 1].mean() for end in ends]
        ),
        "high_trade_intensity": np.array(
            [trade_volume[end - PAPER.window_length + 1 : end + 1].sum() for end in ends]
        ),
    }
    selected: list[tuple[str, int]] = []
    taken: list[int] = []
    for label, values in stats.items():
        order = np.argsort(values) if label == "low_mid_std" else np.argsort(values)[::-1]
        for position in order:
            end = int(ends[position])
            if all(abs(end - other) >= PAPER.window_length for other in taken):
                selected.append((label, end))
                taken.append(end)
                break
    return selected


def _window_market_stats(dataset: LobDataset, *, start: int, end: int) -> dict[str, float]:
    ask1 = dataset.orderbook["ask1_price"].slice(start, PAPER.window_length).to_numpy().astype(float)
    bid1 = dataset.orderbook["bid1_price"].slice(start, PAPER.window_length).to_numpy().astype(float)
    ask_vol = dataset.orderbook["ask1_volume"].slice(start, PAPER.window_length).to_numpy().astype(float)
    bid_vol = dataset.orderbook["bid1_volume"].slice(start, PAPER.window_length).to_numpy().astype(float)
    trade_volume = (
        dataset.trades["trade_volume_total"].slice(start, PAPER.window_length).to_numpy().astype(float)
    )
    mid = (ask1 + bid1) / 2.0
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-7)
    return {
        "mid_std": float(mid.std()),
        "mid_range": float(mid.max() - mid.min()),
        "spread_mean": float((ask1 - bid1).mean()),
        "abs_imbalance_mean": float(np.abs(imbalance).mean()),
        "trade_volume_sum": float(trade_volume.sum()),
        "trade_event_count": int(np.count_nonzero(trade_volume)),
    }


def _attention_dataset(stock: str, *, seed: int) -> LobDataset:
    stock_index = STOCKS.index(stock)
    dataset = _build_panel(
        stocks=[(stock, SYNTHETIC_STOCK_BASE_PRICES[stock])],
        day_indices=[10],
        events_per_day=10_000,
        seed=seed + 1_000 * stock_index,
    )[0][2]
    return _filter_stable_windows(dataset)


def _training_dataset(stock: str, base_price: float, *, seed: int) -> LobDataset:
    train_panel = _build_panel(
        stocks=[(stock, base_price)],
        day_indices=range(10),
        events_per_day=10_000,
        seed=seed,
    )
    return _merge_lob_datasets([entry[2] for entry in train_panel], day="train")


def _trade_summary(trades: pl.DataFrame) -> pl.DataFrame:
    groups = ["method", "variant", "stock", "train_seed", "episode_id"]
    return trades.with_columns(_stock_expr()).group_by(groups).agg(
        pl.len().alias("log_rows"),
        (pl.col("trade_volume") != 0).sum().alias("fills"),
        pl.col("trade_volume").abs().sum().alias("abs_volume"),
        pl.col("inventory").abs().mean().alias("mean_abs_inventory_log"),
        pl.col("inventory").abs().max().alias("max_abs_inventory"),
        pl.col("value").last().alias("final_value"),
    )


def _normalized_entropy(mass: np.ndarray) -> float:
    mass = np.asarray(mass, dtype=float)
    mass = mass / max(float(mass.sum()), 1e-12)
    return float(-(mass * np.log(mass + 1e-12)).sum() / np.log(len(mass)))


def plot_calibration_comparison(frame: pl.DataFrame, output_path: Path) -> None:
    pivot = frame.pivot(
        values="as_gamma",
        index="stock",
        on="as_calibration",
        aggregate_function="first",
    ).sort("stock")
    labels = pivot["stock"].to_list()
    columns = [col for col in pivot.columns if col != "stock"]
    x = np.arange(len(labels))
    width = 0.18
    fig, axis = plt.subplots(figsize=(8, 3.8))
    for idx, column in enumerate(columns):
        axis.bar(x + (idx - (len(columns) - 1) / 2) * width, pivot[column].to_numpy(), width, label=column)
    axis.set_xticks(x, labels)
    axis.set_ylabel("AS gamma")
    axis.set_title("Training-data AS gamma calibrations")
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_teacher_quality(frame: pl.DataFrame, output_path: Path) -> None:
    labels = [
        f"{METHOD_LABELS.get(row['method'], row['method'])}\n{_format_stock(row['stock'])}"
        for row in frame.iter_rows(named=True)
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    axes[0].bar(x, frame["pnl_mean"].to_numpy(), color="#4C78A8")
    axes[0].set_ylabel("Direct-AS PnL")
    axes[1].bar(x, frame["fills_per_episode"].to_numpy(), color="#59A14F")
    axes[1].set_ylabel("Fills / episode")
    axes[1].set_xticks(x, labels, rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tail_risk(frame: pl.DataFrame, output_path: Path) -> None:
    labels = [METHOD_LABELS.get(method, method) for method in frame["method"].to_list()]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].bar(x, frame["pnl_p05"].to_numpy(), color="#E15759")
    axes[0].set_title("5th pct. PnL")
    axes[1].bar(x, frame["max_drawdown_p95"].to_numpy(), color="#F28E2B")
    axes[1].set_title("95th pct. drawdown")
    axes[2].bar(x, frame["max_abs_inventory_p95"].to_numpy(), color="#76B7B2")
    axes[2].set_title("95th pct. max |inventory|")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_synthetic_validation(frame: pl.DataFrame, output_path: Path) -> None:
    labels = frame["stock"].to_list()
    x = np.arange(len(labels))
    metrics = [
        ("mean_spread", "Mean spread"),
        ("episode_sigma", "Episode volatility"),
        ("trade_event_rate", "Trade event rate"),
        ("mean_abs_l1_imbalance", "Mean |L1 imbalance|"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.2))
    for axis, (column, title) in zip(axes.ravel(), metrics, strict=True):
        axis.bar(x, frame[column].to_numpy(), color="#4C78A8")
        axis.set_xticks(x, labels)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_attention_diagnostics(frame: pl.DataFrame, output_path: Path) -> None:
    labels = frame["criterion"].to_list()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6))
    plot_specs = [
        ("mid_std", "Mid-price std."),
        ("attention_entropy", "Attention entropy"),
        ("attention_recent_10_mass", "Mass on last 10 events"),
        ("attention_top_lag", "Top-attended lag"),
    ]
    for axis, (column, title) in zip(axes.ravel(), plot_specs, strict=True):
        axis.bar(x, frame[column].to_numpy(), color="#59A14F")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _read_csv(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(path)
    if "stock" in frame.columns:
        return frame.with_columns(_stock_expr())
    return frame


def _stock_expr() -> pl.Expr:
    return pl.col("stock").cast(pl.Utf8).map_elements(_format_stock, return_dtype=pl.Utf8).alias("stock")


def _format_stock(value: object) -> str:
    text = str(value)
    if text.isdigit():
        return f"{int(text):06d}"
    return text


def stage_path(path: Path) -> Path:
    return path.expanduser().resolve()


def main() -> None:
    home = Path.home()
    parser = argparse.ArgumentParser(description="Build extension appendix diagnostics.")
    parser.add_argument(
        "--matched-root",
        type=Path,
        default=home / "Downloads" / "mlfcs-as-matched-400k-4094973-20260621_030045",
    )
    parser.add_argument(
        "--strengthening-root",
        type=Path,
        default=home / "Downloads" / "as_paper_strengthening_4273399_4273401_20260622",
    )
    parser.add_argument("--paper-dir", type=Path, default=Path("papers/as-regularized-extension"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("runs/extensions/paper_appendix_diagnostics"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    run_appendix_diagnostics(
        matched_root=stage_path(args.matched_root),
        strengthening_root=stage_path(args.strengthening_root),
        paper_dir=stage_path(args.paper_dir),
        artifact_dir=stage_path(args.artifact_dir),
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
