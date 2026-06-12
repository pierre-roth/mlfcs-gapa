import polars as pl

from mlfcs_gapa.cli import (
    _plot_paper_latency_figure,
    _select_attention_window_ends,
)
from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.paper.constants import PAPER


def test_select_attention_window_ends_are_disjoint() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=600, seed=3))

    stable, rapid = _select_attention_window_ends(dataset)

    assert len(stable) == 2
    assert len(rapid) == 2
    ends = [*stable, *rapid]
    assert all(end >= PAPER.window_length - 1 for end in ends)
    for i, first in enumerate(ends):
        for second in ends[i + 1 :]:
            assert abs(first - second) >= PAPER.window_length

    ask1 = dataset.orderbook["ask1_price"].to_numpy()
    bid1 = dataset.orderbook["bid1_price"].to_numpy()
    mid = (ask1 + bid1) / 2.0

    def window_std(end: int) -> float:
        return float(mid[end - PAPER.window_length + 1 : end + 1].std())

    assert max(window_std(end) for end in stable) <= min(window_std(end) for end in rapid)


def test_select_attention_window_ends_short_dataset_falls_back() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=120, seed=3))

    stable, rapid = _select_attention_window_ends(dataset)

    assert stable == []
    assert rapid == []


def test_paper_latency_figure_uses_five_methods(tmp_path) -> None:
    rows = []
    for method in ["C-PPO", "D-DQN", "AS", "Random", "Fixed_1", "Fixed_2", "Inv-RL"]:
        for latency in (1, 10):
            rows.append(
                {
                    "method": method,
                    "latency_events": latency,
                    "nd_pnl": 1_000.0 / latency,
                    "pnl_map": 2.0 / latency,
                    "profit_ratio": 0.0001 / latency,
                }
            )
    output = tmp_path / "figure_2_paper.png"

    _plot_paper_latency_figure(pl.DataFrame(rows), output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_runtime_helper_has_no_typer_defaults() -> None:
    """The full replication must call a plain helper, not the typer command."""

    import inspect

    from mlfcs_gapa.cli import _benchmark_runtime_rows

    signature = inspect.signature(_benchmark_runtime_rows)
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )
