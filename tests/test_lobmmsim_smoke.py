from __future__ import annotations

from pathlib import Path

from lobmmsim.config import RLTrainConfig, SuiteConfig
from lobmmsim.run_pipeline import run_suite


def test_lobmmsim_end_to_end_smoke(tmp_path: Path) -> None:
    cfg = SuiteConfig(
        mode="smoke",
        data_dir=str(tmp_path / "data"),
        output_root=str(tmp_path / "artifacts"),
        run_name="synthetic_smoke",
        symbols=["000001"],
        seed=11,
    ).apply_mode_defaults()
    run_suite(cfg)
    root = tmp_path / "artifacts" / "synthetic_smoke" / "000001"
    assert (root / "pretrain" / "attn_lob.pt").exists()
    assert (root / "ppo" / "model.pt").exists()
    assert (root / "report" / "summary.json").exists()
