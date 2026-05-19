import torch

from mlfcs_gapa.models.attn_lob import AttnLOBClassifier, AttnLOBEncoder
from mlfcs_gapa.paper.constants import PAPER


def test_attn_lob_encoder_matches_figure_1_shapes() -> None:
    encoder = AttnLOBEncoder()
    x = torch.randn(4, PAPER.window_length, PAPER.lob_width)

    embedding, weights = encoder(x, return_attention_weights=True)

    assert embedding.shape == (4, 64)
    assert weights.shape == (4, 10, PAPER.window_length)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4, 10), atol=1e-5)


def test_attn_lob_classifier_outputs_three_pretrain_classes() -> None:
    model = AttnLOBClassifier()
    x = torch.randn(3, PAPER.window_length, PAPER.lob_width)
    logits = model(x)

    assert logits.shape == (3, 3)
