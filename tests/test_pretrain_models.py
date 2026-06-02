import pytest
import torch

from mlfcs_gapa.models.pretrain_models import (
    count_encoder_parameters,
    count_parameters,
    make_pretrain_model,
    paper_reported_parameter_count,
    pretrain_input_shape,
)


@pytest.mark.parametrize("name", ["FC-LOB", "Conv-LOB", "DeepLOB", "Attn-LOB"])
def test_pretraining_models_emit_three_class_logits(name: str) -> None:
    model = make_pretrain_model(name)
    x = torch.randn(2, *pretrain_input_shape(name))
    logits = model(x)

    assert logits.shape == (2, 3)


def test_encoder_parameter_counts_match_paper_table() -> None:
    for name in ["FC-LOB", "Conv-LOB", "DeepLOB", "Attn-LOB"]:
        model = make_pretrain_model(name)
        paper_count = paper_reported_parameter_count(name)
        assert count_encoder_parameters(model) == paper_count
        assert count_parameters(model) == paper_count + 195
