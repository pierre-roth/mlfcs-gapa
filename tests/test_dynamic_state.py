import numpy as np

from mlfcs_gapa.data.dynamic import dynamic_market_state
from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day


def test_dynamic_market_state_has_24_finite_features() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=200, seed=51))
    state = dynamic_market_state(dataset, 120)

    assert state.shape == (24,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()
    assert np.all(state[6:] >= -1.000001)
    assert np.all(state[6:] <= 1.000001)
