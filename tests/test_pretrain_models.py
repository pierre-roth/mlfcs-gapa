import pytest
import torch

from mlfcs_gapa.models.pretrain_models import make_pretrain_model
from mlfcs_gapa.paper.constants import PAPER


@pytest.mark.parametrize("name", ["FC-LOB", "Conv-LOB", "DeepLOB", "Attn-LOB"])
def test_pretraining_models_emit_three_class_logits(name: str) -> None:
    model = make_pretrain_model(name)
    x = torch.randn(2, PAPER.window_length, PAPER.lob_width)
    logits = model(x)

    assert logits.shape == (2, 3)
