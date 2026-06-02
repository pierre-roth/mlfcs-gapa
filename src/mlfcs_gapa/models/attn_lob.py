"""PyTorch Attn-LOB implementation matching Figure 1 of the paper."""

from __future__ import annotations

import math

import torch
from torch import nn

from mlfcs_gapa.paper.constants import PAPER


class KerasStyleMultiHeadAttention(nn.Module):
    """Keras-compatible MHA used by the paper's reference Attn-LOB code."""

    def __init__(
        self,
        *,
        input_dim: int = 192,
        num_heads: int = 10,
        key_dim: int = 16,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.output_dim = output_dim
        projection_dim = num_heads * key_dim
        self.query_projection = nn.Linear(input_dim, projection_dim)
        self.key_projection = nn.Linear(input_dim, projection_dim)
        self.value_projection = nn.Linear(input_dim, projection_dim)
        self.output_projection = nn.Linear(projection_dim, output_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]
        query = self._project_heads(self.query_projection(query), batch_size)
        key = self._project_heads(self.key_projection(key), batch_size)
        value = self._project_heads(self.value_projection(value), batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.key_dim)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.key_dim)
        return self.output_projection(attended), weights

    def _project_heads(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        return tensor.reshape(batch_size, -1, self.num_heads, self.key_dim).transpose(1, 2)


class AttnLOBEncoder(nn.Module):
    """CNN-Inception-Attention LOB encoder.

    Input shape is `(batch, 50, 40)`. Internally this becomes
    `(batch, channels=1, time=50, width=40)` for Conv2D layers.
    """

    def __init__(self, attention_heads: int = 10, attention_key_dim: int = 16) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(1, 5), stride=(1, 5)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(1, 4)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
        )

        self.branch_3x1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(0.01),
        )
        self.branch_5x1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(0.01),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
        )
        self.attention = KerasStyleMultiHeadAttention(
            input_dim=192,
            num_heads=attention_heads,
            key_dim=attention_key_dim,
            output_dim=64,
        )
        self.attention_key_dim = attention_key_dim

    def forward(
        self,
        lob_window: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if lob_window.ndim != 3:
            raise ValueError("AttnLOBEncoder expects shape (batch, 50, 40)")
        if tuple(lob_window.shape[1:]) != PAPER.lob_window_shape:
            raise ValueError(
                f"expected input shape (*, {PAPER.lob_window_shape}), got {tuple(lob_window.shape)}"
            )

        x = lob_window.unsqueeze(1)
        x = self.conv_stack(x)
        if tuple(x.shape[1:]) != (32, PAPER.window_length, 1):
            raise RuntimeError(f"unexpected post-conv shape {tuple(x.shape)}")

        x = torch.cat([self.branch_3x1(x), self.branch_5x1(x), self.branch_pool(x)], dim=1)
        if tuple(x.shape[1:]) != (192, PAPER.window_length, 1):
            raise RuntimeError(f"unexpected inception shape {tuple(x.shape)}")

        sequence = x.squeeze(-1).transpose(1, 2)
        query = sequence[:, -1:, :]
        attended, weights = self.attention(query=query, key=sequence, value=sequence)
        embedding = attended.squeeze(1)
        if return_attention_weights:
            return embedding, weights.squeeze(2)
        return embedding


class AttnLOBClassifier(nn.Module):
    """Three-class pretraining head: down, stationary, up."""

    def __init__(self, encoder: AttnLOBEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder or AttnLOBEncoder()
        self.classifier = nn.Linear(64, 3)

    def forward(self, lob_window: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(lob_window))
