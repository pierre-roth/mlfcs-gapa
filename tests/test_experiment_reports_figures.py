import numpy as np
import polars as pl

from mlfcs_gapa.experiments.figures import (
    plot_attention_heatmap,
    plot_decision_trace,
    plot_latency_figure,
)
from mlfcs_gapa.experiments.reports import add_paper_table_columns, summarize_paper_table


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
