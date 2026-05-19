"""PyTorch Attn-LOB implementation matching Figure 1 of the paper."""

from __future__ import annotations

import torch
from torch import nn

from mlfcs_gapa.paper.constants import PAPER


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
        attention_dim = attention_heads * attention_key_dim
        self.attention_input_projection = nn.Linear(192, attention_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.attention_projection = nn.Linear(attention_dim, 64)
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

        sequence = self.attention_input_projection(x.squeeze(-1).transpose(1, 2))
        query = sequence[:, -1:, :]
        attended, weights = self.attention(
            query=query,
            key=sequence,
            value=sequence,
            need_weights=True,
            average_attn_weights=False,
        )
        embedding = self.attention_projection(attended.squeeze(1))
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
