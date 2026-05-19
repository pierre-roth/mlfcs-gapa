import numpy as np

from mlfcs_gapa.data.pretraining import build_pretrain_arrays
from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.paper.constants import PAPER


def test_pretrain_arrays_align_lob_windows_and_labels() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=140, seed=41))
    arrays = build_pretrain_arrays(dataset)

    assert arrays.x.ndim == 3
    assert arrays.x.shape[1:] == PAPER.lob_window_shape
    assert arrays.y.shape == (arrays.x.shape[0],)
    assert arrays.y.dtype == np.int64
    assert set(arrays.y.tolist()) <= {0, 1, 2}
    assert np.isfinite(arrays.x).all()
