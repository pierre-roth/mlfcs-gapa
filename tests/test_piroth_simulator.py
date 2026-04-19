from __future__ import annotations

from pathlib import Path

import pandas as pd

from piroth.config import GenerateConfig
from piroth.simulator import generate_dataset


def test_simulator_is_deterministic(tmp_path: Path) -> None:
    cfg_a = GenerateConfig(mode="smoke", data_dir=str(tmp_path / "a"), symbols=["000001"], seed=5).apply_mode_defaults()
    cfg_b = GenerateConfig(mode="smoke", data_dir=str(tmp_path / "b"), symbols=["000001"], seed=5).apply_mode_defaults()
    generate_dataset(cfg_a)
    generate_dataset(cfg_b)
    price_a = pd.read_csv(tmp_path / "a" / "000001" / "20191101" / "price.csv")
    price_b = pd.read_csv(tmp_path / "b" / "000001" / "20191101" / "price.csv")
    assert price_a.equals(price_b)

