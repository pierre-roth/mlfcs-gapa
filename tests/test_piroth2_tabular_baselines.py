from __future__ import annotations

import numpy as np

from piroth.config import DiagnosticsConfig
from piroth.tabular_baselines import evaluate_inventory_rl, evaluate_lob_rl, train_inventory_rl, train_lob_rl
from tests.test_piroth2_baselines import _calibration_day


def test_inventory_rl_trains_and_evaluates(tmp_path) -> None:
    config = _config()
    day = _calibration_day()

    qtable = train_inventory_rl([day], config, tmp_path / "models")
    summary = evaluate_inventory_rl([day], config, qtable, tmp_path)

    assert qtable.exists()
    assert (tmp_path / "inventory_rl_episodes.csv").exists()
    assert np.isfinite(summary["pnl_mean"])


def test_lob_rl_trains_and_evaluates(tmp_path) -> None:
    config = _config()
    day = _calibration_day()

    qtable = train_lob_rl([day], config, tmp_path / "models")
    summary = evaluate_lob_rl([day], config, qtable, tmp_path)

    assert qtable.exists()
    assert (tmp_path / "lob_rl_episodes.csv").exists()
    assert np.isfinite(summary["pnl_mean"])


def _config() -> DiagnosticsConfig:
    return DiagnosticsConfig(
        mode="smoke",
        symbol="000001",
        seed=5,
        lookback=2,
        latency=0,
        episode_length=20,
        stable_windows=["10:00:00-10:02:00"],
        max_train_episodes_per_day=1,
        max_eval_episodes_per_day=1,
        tabular_epochs=2,
        tabular_lob_lookback=4,
        reward_mode="hybrid",
        reward_spread_penalty_weight=0.0,
    )
