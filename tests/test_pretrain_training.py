import torch
from torch import nn

from mlfcs_gapa.data.pretraining import build_pretrain_arrays
from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.paper.constants import PAPER
from mlfcs_gapa.training.pretrain import train_lob_classifier


class TinyLobClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(PAPER.window_length * PAPER.lob_width, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_train_lob_classifier_returns_table_metrics() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=81))
    arrays = build_pretrain_arrays(dataset)
    metrics = train_lob_classifier(
        TinyLobClassifier(),
        arrays,
        epochs=1,
        batch_size=32,
        seed=1,
    )

    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.n_train > 0
    assert metrics.n_val > 0


def test_train_lob_classifier_can_report_held_out_metrics() -> None:
    train_dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=82))
    eval_dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=83))
    train_arrays = build_pretrain_arrays(train_dataset)
    eval_arrays = build_pretrain_arrays(eval_dataset)

    metrics = train_lob_classifier(
        TinyLobClassifier(),
        train_arrays,
        evaluation_arrays=eval_arrays,
        epochs=1,
        batch_size=32,
        seed=1,
    )

    assert metrics.n_train > 0
    assert metrics.n_val == len(eval_arrays.y)
