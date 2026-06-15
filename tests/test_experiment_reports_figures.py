import numpy as np
import polars as pl
import pytest

from mlfcs_gapa.experiments.figures import (
    plot_attention_heatmap,
    plot_attention_market_grid,
    plot_decision_trace,
    plot_latency_figure,
)
from mlfcs_gapa.experiments.reports import (
    add_paper_table_columns,
    aggregate_period_table,
    summarize_paper_table,
)


def test_report_helpers_add_paper_scaled_columns() -> None:
    metrics = pl.DataFrame(
        {
            "method": ["A", "A", "B"],
            "stock": ["000001", "000001", "000001"],
            "nd_pnl": [100_000.0, 200_000.0, 50_000.0],
            "pnl_map": [1.0, 3.0, 2.0],
            "profit_ratio": [0.0001, 0.0002, 0.0003],
        }
    )

    scaled = add_paper_table_columns(metrics)
    summary = summarize_paper_table(metrics)

    assert scaled["nd_pnl_table"].to_list() == [1.0, 2.0, 0.5]
    assert np.allclose(scaled["profit_ratio_table"].to_numpy(), [1.0, 2.0, 3.0])
    assert summary.height == 2
    assert "pnl_map_mean" in summary.columns


def test_aggregate_period_table_matches_paper_convention() -> None:
    metrics = pl.DataFrame(
        {
            "method": ["AS", "AS", "C-PPO", "C-PPO", "C-PPO", "C-PPO"],
            "stock": ["000001"] * 6,
            "train_seed": [None, None, 0, 0, 1, 1],
            "pnl": [10.0, 20.0, 40.0, 60.0, 50.0, 70.0],
            "mean_quoted_spread": [0.01, 0.03, 0.02, 0.02, 0.02, 0.02],
            "mean_abs_inventory": [100.0, 300.0, 50.0, 50.0, 50.0, 50.0],
            "buy_notional": [1_000.0, 1_000.0, 500.0, 500.0, 500.0, 500.0],
        }
    )

    table = aggregate_period_table(metrics)

    as_row = table.filter(pl.col("method") == "AS")
    # Period totals: PnL 30 over mean spread 0.02 -> ND-PnL 1500 -> 0.015e5.
    assert as_row["nd_pnl_e5_mean"][0] == pytest.approx(30.0 / 0.02 / 1e5, rel=1e-4)
    assert as_row["pnl_map_mean"][0] == pytest.approx(30.0 / 200.0, rel=1e-4)
    assert as_row["profit_ratio_e4_mean"][0] == pytest.approx(30.0 / 2_000.0 * 1e4, rel=1e-4)
    assert as_row["nd_pnl_e5_std"][0] is None
    assert as_row["seeds"][0] == 1

    ppo_row = table.filter(pl.col("method") == "C-PPO")
    # Two seeds with period PnL 100 and 120 -> mean 110/0.02/1e5.
    assert ppo_row["nd_pnl_e5_mean"][0] == pytest.approx(110.0 / 0.02 / 1e5, rel=1e-4)
    assert ppo_row["nd_pnl_e5_std"][0] > 0
    assert ppo_row["seeds"][0] == 2


def test_latency_figure_is_written(tmp_path) -> None:
    metrics = pl.DataFrame(
        {
            "method": ["AS", "AS", "Random", "Random"],
            "latency_events": [1, 10, 1, 10],
            "nd_pnl": [100_000.0, 50_000.0, 20_000.0, -10_000.0],
            "pnl_map": [4.0, 2.0, 1.0, -1.0],
            "profit_ratio": [0.0004, 0.0002, 0.0001, -0.0001],
        }
    )
    output = tmp_path / "latency.png"

    plot_latency_figure(metrics, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_decision_trace_is_written(tmp_path) -> None:
    trades = pl.DataFrame(
        {
            "index": [1, 2, 3],
            "mid_price": [10.0, 10.1, 10.2],
            "ask_price": [10.1, 0.0, 10.3],
            "bid_price": [9.9, 10.0, 0.0],
            "inventory": [0, 100, 0],
            "value": [0.0, 1.0, 0.5],
        }
    )
    output = tmp_path / "decision.png"

    plot_decision_trace(trades, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_attention_heatmap_is_written(tmp_path) -> None:
    output = tmp_path / "attention.png"

    plot_attention_heatmap(np.ones((10, 50)) / 50.0, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_attention_heatmap_with_lob_state_is_written(tmp_path) -> None:
    output = tmp_path / "attention_lob_state.png"
    lob_window = np.arange(50 * 40, dtype=np.float32).reshape(50, 40)

    plot_attention_heatmap(np.ones((10, 50)) / 50.0, output, lob_window=lob_window)

    assert output.exists()
    assert output.stat().st_size > 0


def test_attention_market_grid_is_written(tmp_path) -> None:
    output = tmp_path / "attention_market_grid.png"
    weights = np.ones((10, 50)) / 50.0
    lob_window = np.arange(50 * 40, dtype=np.float32).reshape(50, 40)
    panel = (weights, lob_window)

    plot_attention_market_grid([panel, panel], [panel, panel], output)

    assert output.exists()
    assert output.stat().st_size > 0
