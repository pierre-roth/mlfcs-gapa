from __future__ import annotations

from piroth.baselines import calibrate_avellaneda_stoikov
from piroth.config import DiagnosticsConfig
from piroth.simulator import SyntheticMarketGenerator


def test_as_calibration_returns_positive_kappa() -> None:
    config = DiagnosticsConfig(mode="smoke", symbol="000001", seed=3)
    generator = SyntheticMarketGenerator(config)
    days = [generator.generate_day(day) for day in generator.train_days()]
    calibration = calibrate_avellaneda_stoikov(days, config)

    assert calibration.kappa > 0
    assert calibration.sigma2_event > 0
