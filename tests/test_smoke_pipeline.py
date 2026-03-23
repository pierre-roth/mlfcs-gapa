from __future__ import annotations

from pathlib import Path

from lobmm.config import ExperimentConfig, PretrainConfig, RLTrainConfig
from lobmm.evaluate import run_evaluation
from lobmm.env import MarketMakingEnv
from lobmm.pipeline import load_symbol_splits
from lobmm.pretrain import run_pretrain
from lobmm.report import run_report
from lobmm.train_rl import run_rl_training


def test_mode_defaults_keep_method_shape() -> None:
    smoke_cfg = RLTrainConfig(mode="smoke", symbols=["AAPL"]).apply_mode_defaults()
    assert smoke_cfg.pretrain_backbone == "attn"
    assert smoke_cfg.episode_length == 2000
    assert smoke_cfg.device in {"cpu", "cuda", "mps"}

    full_cfg = RLTrainConfig(mode="full", symbols=["AAPL"]).apply_mode_defaults()
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


def test_pretrain_and_ppo_smoke(tmp_path: Path) -> None:
    pre_cfg = PretrainConfig(
        mode="smoke",
        symbols=["AAPL"],
        output_root=str(tmp_path),
        run_name="test_run",
        pretrain_epochs=1,
        max_rows_per_day=4_000,
        max_pretrain_samples_per_day=256,
    ).apply_mode_defaults()
    pre_result = run_pretrain(pre_cfg)
    assert "AAPL" in pre_result
    ppo_cfg = RLTrainConfig(
        mode="smoke",
        symbols=["AAPL"],
        output_root=str(tmp_path),
        run_name="test_run",
        algorithm="ppo",
        state_mode="full",
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
