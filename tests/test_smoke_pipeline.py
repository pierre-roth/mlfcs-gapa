from __future__ import annotations

from pathlib import Path
import json

from lobmm.config import ExperimentConfig, PretrainConfig, RLTrainConfig
from lobmm.evaluate import run_evaluation
from lobmm.env import MarketMakingEnv
from lobmm.pipeline import load_symbol_splits, prepare_run, resolve_symbol_rl_config
from lobmm.pretrain import run_pretrain
from lobmm.report import run_report
from lobmm.train_rl import run_rl_training


def test_mode_defaults_keep_method_shape() -> None:
    smoke_cfg = RLTrainConfig(mode="smoke", symbols=["AAPL"]).apply_mode_defaults()
    assert smoke_cfg.pretrain_backbone == "attn"
    assert smoke_cfg.episode_length == 2000
    assert smoke_cfg.device in {"cpu", "cuda"}
    assert smoke_cfg.quote_scale_mode == "bps"
    assert smoke_cfg.target_episode_seconds == 120
    assert smoke_cfg.pretrain_balance_mode == "weighted_loss"
    assert smoke_cfg.pretrain_eval_samples_per_day == 1_024

    medium_cfg = RLTrainConfig(mode="medium", symbols=["AAPL"]).apply_mode_defaults()
    assert medium_cfg.train_days == 4
    assert medium_cfg.val_days == 1
    assert medium_cfg.test_days == 1
    assert medium_cfg.max_rows_per_day == 200_000
    assert medium_cfg.max_pretrain_samples_per_day == 50_000
    assert medium_cfg.max_train_episodes_per_day == 8
    assert medium_cfg.max_eval_episodes_per_day == 4
    assert medium_cfg.pretrain_epochs == 4
    assert medium_cfg.ppo_epochs == 4
    assert medium_cfg.pretrain_batch_size == 256
    assert medium_cfg.ppo_minibatch_size == 256
    assert medium_cfg.pretrain_eval_samples_per_day == 12_500

    full_cfg = RLTrainConfig(mode="full", symbols=["AAPL"]).apply_mode_defaults()
    assert full_cfg.train_days == 7
    assert full_cfg.val_days == 1
    assert full_cfg.test_days == 2
    assert full_cfg.max_rows_per_day is None
    assert full_cfg.max_pretrain_samples_per_day is None
    assert full_cfg.max_train_episodes_per_day is None
    assert full_cfg.max_eval_episodes_per_day is None


def test_load_symbol_splits_smoke() -> None:
    cfg = ExperimentConfig(mode="smoke", symbols=["AAPL", "GOOGL"]).apply_mode_defaults()
    for symbol in cfg.symbols:
        splits = load_symbol_splits(cfg, symbol)
        assert len(splits["train"]) == 2
        assert len(splits["val"]) == 1
        assert len(splits["test"]) == 1
        sample = splits["train"][0]
        assert sample.lob.shape[1] == 40
        assert sample.dynamic.shape[1] == 24
        assert sample.normalized_lob is not None
        assert sample.timestamps[0].strftime("%H:%M:%S") >= "10:00:00"
        assert sample.timestamps[-1].strftime("%H:%M:%S") < "15:30:00"


def test_env_one_step_smoke() -> None:
    cfg = RLTrainConfig(mode="smoke", symbols=["AAPL"]).apply_mode_defaults()
    splits = load_symbol_splits(cfg, "AAPL")
    env = MarketMakingEnv(splits["train"][0], cfg)
    span = env.available_episodes()[0]
    obs = env.reset(span)
    assert obs.flat.shape[0] > 0
    next_obs, reward, done, info = env.step([0.5, 0.5])
    assert next_obs.flat.shape == obs.flat.shape
    assert isinstance(reward, float)
    assert "inventory" in info
    assert done in {True, False}
    result = env.episode_result("test", 0)
    assert hasattr(result, "fill_rate")
    assert hasattr(result, "avg_spread_bps")


def test_selected_episodes_spread_across_day() -> None:
    cfg = RLTrainConfig(mode="smoke", symbols=["AAPL"]).apply_mode_defaults()
    splits = load_symbol_splits(cfg, "AAPL")
    env = MarketMakingEnv(splits["train"][0], cfg)
    episodes = env.available_episodes()
    selected = env.selected_episodes(2)
    assert len(episodes) >= 2
    assert selected[0] == episodes[0]
    assert selected[-1] == episodes[-1]


def test_symbol_episode_length_targets_about_two_minutes() -> None:
    cfg = RLTrainConfig(mode="smoke", symbols=["AAPL"]).apply_mode_defaults()
    splits = load_symbol_splits(cfg, "AAPL")
    resolved = resolve_symbol_rl_config(cfg, splits["train"])
    assert 25_000 <= resolved.episode_length <= 120_000


def test_bps_quotes_scale_with_midprice() -> None:
    cfg = RLTrainConfig(mode="smoke", symbols=["AAPL"]).apply_mode_defaults()
    splits = load_symbol_splits(cfg, "AAPL")
    resolved = resolve_symbol_rl_config(cfg, splits["train"])
    env = MarketMakingEnv(splits["train"][0], resolved)
    span = env.available_episodes()[0]
    env.reset(span)
    quote_idx = max(int(env.episode_decisions[env.step_cursor] - env.config.latency), env.config.lookback - 1)
    orders = env.action_to_orders([1.0, 1.0], quote_idx)
    expected = float(env.day.midprice[quote_idx]) * resolved.max_spread_bps * 1e-4
    assert abs(orders["spread"] - expected) <= 0.03


def test_inventory_penalty_is_normalized_to_limit() -> None:
    cfg = RLTrainConfig(mode="smoke", symbols=["AAPL"]).apply_mode_defaults()
    splits = load_symbol_splits(cfg, "AAPL")
    env = MarketMakingEnv(splits["train"][0], cfg)
    span = env.available_episodes()[0]
    env.reset(span)
    env.inventory = cfg.max_inventory * cfg.trade_unit
    reward = env._reward([], float(env.day.midprice[env.episode_decisions[0]]))
    assert abs(reward + cfg.zeta) < 1e-6


def test_pretrain_and_ppo_smoke(tmp_path: Path) -> None:
    pre_cfg = PretrainConfig(
        mode="smoke",
        symbols=["AAPL"],
        output_root=str(tmp_path),
        run_name="test_run",
        use_stable_hours=False,
        pretrain_epochs=1,
        max_rows_per_day=4_000,
        max_pretrain_samples_per_day=256,
    ).apply_mode_defaults()
    pre_result = run_pretrain(pre_cfg)
    assert "AAPL" in pre_result
    pre_summary = json.loads((tmp_path / "test_run" / "AAPL" / "pretrain" / "summary.json").read_text())
    assert pre_summary["pretrain_balance_mode"] == "weighted_loss"
    assert set(pre_summary["split_metrics"].keys()) == {"train", "val", "test"}
    assert len(pre_summary["class_weights"]) == 3
    ppo_cfg = RLTrainConfig(
        mode="smoke",
        symbols=["AAPL"],
        output_root=str(tmp_path),
        run_name="test_run",
        algorithm="ppo",
        state_mode="full",
        use_stable_hours=False,
        target_episode_seconds=None,
        episode_length=512,
        ppo_epochs=1,
        max_rows_per_day=4_000,
        max_train_episodes_per_day=1,
        max_eval_episodes_per_day=1,
    ).apply_mode_defaults()
    rl_result = run_rl_training(ppo_cfg)
    assert "AAPL" in rl_result
    assert (tmp_path / "test_run" / "AAPL" / "ppo" / "full" / "episodes.csv").exists()


def test_ppo_variants_use_distinct_output_dirs(tmp_path: Path) -> None:
    pre_cfg = PretrainConfig(
        mode="smoke",
        symbols=["AAPL"],
        output_root=str(tmp_path),
        run_name="variant_run",
        use_stable_hours=False,
        pretrain_epochs=1,
        max_rows_per_day=4_000,
        max_pretrain_samples_per_day=256,
    ).apply_mode_defaults()
    run_pretrain(pre_cfg)
    full_cfg = RLTrainConfig(
        mode="smoke",
        symbols=["AAPL"],
        output_root=str(tmp_path),
        run_name="variant_run",
        algorithm="ppo",
        state_mode="full",
        use_stable_hours=False,
        target_episode_seconds=None,
        episode_length=512,
        ppo_epochs=1,
        max_rows_per_day=4_000,
        max_train_episodes_per_day=1,
        max_eval_episodes_per_day=1,
    ).apply_mode_defaults()
    wo_lob_cfg = RLTrainConfig(
        mode="smoke",
        symbols=["AAPL"],
        output_root=str(tmp_path),
        run_name="variant_run",
        algorithm="ppo",
        state_mode="full",
        wo_lob_state=True,
        use_stable_hours=False,
        target_episode_seconds=None,
        episode_length=512,
        ppo_epochs=1,
        max_rows_per_day=4_000,
        max_train_episodes_per_day=1,
        max_eval_episodes_per_day=1,
    ).apply_mode_defaults()
    run_rl_training(full_cfg)
    run_rl_training(wo_lob_cfg)
    assert (tmp_path / "variant_run" / "AAPL" / "ppo" / "full" / "episodes.csv").exists()
    assert (tmp_path / "variant_run" / "AAPL" / "ppo" / "full_wo_lob" / "episodes.csv").exists()


def test_report_includes_baselines_and_outputs_tables(tmp_path: Path) -> None:
    pre_cfg = PretrainConfig(
        mode="smoke",
        symbols=["AAPL"],
        output_root=str(tmp_path),
        run_name="report_run",
        use_stable_hours=False,
        pretrain_epochs=1,
        max_rows_per_day=4_000,
        max_pretrain_samples_per_day=256,
    ).apply_mode_defaults()
    run_pretrain(pre_cfg)
    ppo_cfg = RLTrainConfig(
        mode="smoke",
        symbols=["AAPL"],
        output_root=str(tmp_path),
        run_name="report_run",
        algorithm="ppo",
        state_mode="full",
        use_stable_hours=False,
        target_episode_seconds=None,
        episode_length=512,
        ppo_epochs=1,
        max_rows_per_day=4_000,
        max_train_episodes_per_day=1,
        max_eval_episodes_per_day=1,
    ).apply_mode_defaults()
    run_rl_training(ppo_cfg)
    run_evaluation(ppo_cfg)
    report_dir = run_report(ExperimentConfig(mode="smoke", symbols=["AAPL"], output_root=str(tmp_path), run_name="report_run").apply_mode_defaults())
    method_summary = (report_dir / "method_summary.csv").read_text()
    assert "AS" in method_summary
    assert "C-PPO" in (report_dir / "continuous_paper_table.md").read_text()
    assert (report_dir / "continuous_overall_results.md").exists()
    assert (report_dir / "runtime_summary.md").exists()
    assert (report_dir / "policy_diagnostics.md").exists()


def test_prepare_run_keeps_root_config_and_stage_snapshots(tmp_path: Path) -> None:
    pre_cfg = PretrainConfig(
        mode="medium",
        symbols=["AAPL"],
        output_root=str(tmp_path),
        run_name="config_run",
        device="cuda",
    ).apply_mode_defaults()
    out_dir = prepare_run(pre_cfg, label="pretrain")

    report_cfg = ExperimentConfig(
        mode="medium",
        symbols=["AAPL"],
        output_root=str(tmp_path),
        run_name="config_run",
        device="cpu",
    ).apply_mode_defaults()
    prepare_run(report_cfg, label="report")

    root = json.loads((out_dir / "config.json").read_text())
    pre = json.loads((out_dir / "config_pretrain.json").read_text())
    report = json.loads((out_dir / "config_report.json").read_text())

    assert root["device"] == "cuda"
    assert pre["device"] == "cuda"
    assert report["device"] == "cpu"
