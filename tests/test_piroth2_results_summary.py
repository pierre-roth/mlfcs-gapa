from __future__ import annotations

import pandas as pd

from piroth.results_summary import (
    _frame_to_markdown,
    _paired_seed_baseline_table,
    _ppo_seed_table,
    _ppo_tuned_table,
    _ppo_vs_as_table,
    _html,
)


def test_frame_to_markdown_without_optional_tabulate() -> None:
    frame = pd.DataFrame(
        [
            {"symbol": "000001", "pnl_mean": -4.916666, "episodes": 12.0},
            {"symbol": "000858", "pnl_mean": 74.0, "episodes": 12.0},
        ]
    )

    rendered = _frame_to_markdown(frame)

    assert "| symbol | pnl_mean | episodes |" in rendered
    assert "| 000001 | -4.9167  | 12       |" in rendered
    assert "| 000858 | 74       | 12       |" in rendered


def test_html_summary_contains_tables() -> None:
    html = _html({"example": pd.DataFrame([{"agent": "AS", "pnl_mean": 3.25}])})

    assert "<title>Paper Replication Result Summary</title>" in html
    assert "<h2>Example</h2>" in html
    assert "result-table" in html
    assert "3.25" in html


def test_ppo_seed_table_collects_finished_runs(tmp_path) -> None:
    finished = tmp_path / "piroth2_ppo_seed11_000858_20260424_232000"
    unfinished = tmp_path / "piroth2_ppo_seed17_000858_20260424_232000"
    finished.mkdir()
    unfinished.mkdir()
    pd.DataFrame(
        [
            {"pnl": 10.0, "reward": -2.0, "fill_rate": 0.10, "trades": 8, "avg_spread": 0.03},
            {"pnl": 14.0, "reward": -4.0, "fill_rate": 0.20, "trades": 10, "avg_spread": 0.02},
        ]
    ).to_csv(finished / "ppo_episodes.csv", index=False)

    table = _ppo_seed_table(tmp_path)

    assert table.to_dict("records") == [
        {
            "symbol": "000858",
            "seed": 11,
            "stamp": "20260424_232000",
            "episodes": 2.0,
            "pnl_mean": 12.0,
            "reward_mean": -3.0,
            "fill_rate_mean": 0.15000000000000002,
            "trades_mean": 9.0,
            "avg_spread_mean": 0.025,
        }
    ]


def test_paired_seed_baseline_table_collects_policy_metrics(tmp_path) -> None:
    run = tmp_path / "piroth2_baseline_seed17_000001_20260425_000500"
    run.mkdir()
    (run / "paper_baseline_summary.json").write_text(
        """
        {
          "paper_baselines": {
            "AS": {
              "episodes": 12,
              "pnl_mean": -7.5,
              "reward_mean": -101.0,
              "fill_rate_mean": 0.02,
              "trades_mean": 24.0,
              "avg_spread_mean": 0.03
            }
          }
        }
        """,
        encoding="utf-8",
    )

    table = _paired_seed_baseline_table(tmp_path)

    assert table.to_dict("records") == [
        {
            "symbol": "000001",
            "seed": 17,
            "stamp": "20260425_000500",
            "agent": "AS",
            "episodes": 12.0,
            "pnl_mean": -7.5,
            "reward_mean": -101.0,
            "fill_rate_mean": 0.02,
            "trades_mean": 24.0,
            "avg_spread_mean": 0.03,
        }
    ]


def test_ppo_tuned_table_collects_finished_runs(tmp_path) -> None:
    run = tmp_path / "piroth2_ppo_tuned_002415_seed11_20260424_235000"
    run.mkdir()
    pd.DataFrame(
        [
            {"pnl": -2.0, "reward": -10.0, "fill_rate": 0.04, "trades": 5, "avg_spread": 0.03},
            {"pnl": 6.0, "reward": -20.0, "fill_rate": 0.06, "trades": 7, "avg_spread": 0.03},
        ]
    ).to_csv(run / "ppo_episodes.csv", index=False)

    table = _ppo_tuned_table(tmp_path)

    assert table.to_dict("records") == [
        {
            "symbol": "002415",
            "seed": 11,
            "stamp": "20260424_235000",
            "episodes": 2.0,
            "pnl_mean": 2.0,
            "reward_mean": -15.0,
            "fill_rate_mean": 0.05,
            "trades_mean": 6.0,
            "avg_spread_mean": 0.03,
        }
    ]


def test_ppo_vs_as_table_reports_advantage() -> None:
    ppo = pd.DataFrame(
        [
            {
                "symbol": "000858",
                "seed": 11,
                "pnl_mean": 52.0,
                "reward_mean": -500.0,
                "fill_rate_mean": 0.14,
                "trades_mean": 135.0,
            }
        ]
    )
    baselines = pd.DataFrame(
        [
            {
                "symbol": "000858",
                "seed": 11,
                "agent": "AS",
                "pnl_mean": 81.0,
                "reward_mean": -300.0,
                "fill_rate_mean": 0.13,
                "trades_mean": 132.0,
            },
            {
                "symbol": "000858",
                "seed": 11,
                "agent": "Random",
                "pnl_mean": -10.0,
                "reward_mean": -600.0,
                "fill_rate_mean": 0.05,
                "trades_mean": 40.0,
            },
        ]
    )

    table = _ppo_vs_as_table(ppo, baselines)

    assert table[["symbol", "seed", "pnl_advantage", "reward_advantage"]].to_dict("records") == [
        {"symbol": "000858", "seed": 11, "pnl_advantage": -29.0, "reward_advantage": -200.0}
    ]
