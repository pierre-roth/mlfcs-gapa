from __future__ import annotations

from piroth.baselines import AvellanedaStoikovPolicy, calibrate_avellaneda_stoikov
from piroth.config import GenerateConfig, ReportConfig
from piroth.data import load_splits
from piroth.simulator import generate_dataset


def test_calibrated_as_uses_empirical_scale(tmp_path) -> None:
    data_dir = tmp_path / "sim"
    gen_cfg = GenerateConfig(mode="smoke", data_dir=str(data_dir), symbols=["000001"], seed=7).apply_mode_defaults()
    generate_dataset(gen_cfg)
    report_cfg = ReportConfig(mode="smoke", data_dir=str(data_dir), symbols=["000001"], seed=7).apply_mode_defaults()
    splits = load_splits(report_cfg, "000001")
    calibration = calibrate_avellaneda_stoikov(splits["train"], report_cfg)
    assert calibration.kappa > 0.0
    assert calibration.step_variance > 0.0
    policy = AvellanedaStoikovPolicy(report_cfg, calibration)
    day = splits["test"][0]
    idx = int(day.valid_label_indices(report_cfg.lookback, report_cfg.pretrain_horizon)[0])
    decision = policy.act(day, idx, inventory=0.0, step_cursor=0, total_steps=report_cfg.episode_length)
    assert report_cfg.tick_size <= decision.spread <= report_cfg.max_spread
    assert decision.spread < report_cfg.max_spread
