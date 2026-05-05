from __future__ import annotations

import torch
from torch import nn


class FCLOBEncoder(nn.Module):
    """Fully connected LOB encoder baseline from the paper's Table I."""

    def __init__(self, output_dim: int = 64, lookback: int = 50, features: int = 40) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * features, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, output_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        if lob_state.ndim != 4:
            raise ValueError(f"Expected LOB tensor with 4 dims, got {tuple(lob_state.shape)}")
        return self.net(lob_state)


class ConvLOBEncoder(nn.Module):
    """Dilated convolutional LOB encoder baseline.

    The paper describes Conv-LOB as a fully convolutional network using dilated
    convolutions to accept longer temporal context. This implementation keeps
    the same 50x40 input used by the replication and emits the standard
    64-dimensional latent vector.
    """

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0), dilation=(1, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(2, 0), dilation=(2, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(4, 0), dilation=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(8, 0), dilation=(8, 1)),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, output_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        if lob_state.ndim != 4:
            raise ValueError(f"Expected LOB tensor with 4 dims, got {tuple(lob_state.shape)}")
        x = lob_state.permute(0, 3, 1, 2).contiguous()
        return self.net(x)


class DeepLOBEncoder(nn.Module):
    """DeepLOB-style CNN plus recurrent encoder baseline.

    DeepLOB uses convolutional spatial feature extraction followed by recurrent
    temporal aggregation. The spatial/inception front-end mirrors the LOB CNN
    used by Attn-LOB; the temporal aggregator is an LSTM rather than attention.
    """

    def __init__(self, output_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.spatial = nn.Sequential(
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
        )
        self.inception_3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), nn.LeakyReLU(0.01), nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"), nn.LeakyReLU(0.01))
        self.inception_5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), nn.LeakyReLU(0.01), nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"), nn.LeakyReLU(0.01))
        self.inception_pool = nn.Sequential(nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)), nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), nn.LeakyReLU(0.01))
        self.lstm = nn.LSTM(input_size=192, hidden_size=hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        if lob_state.ndim != 4:
            raise ValueError(f"Expected LOB tensor with 4 dims, got {tuple(lob_state.shape)}")
        x = lob_state.permute(0, 3, 1, 2).contiguous()
        x = self.spatial(x)
        x = torch.cat([self.inception_3(x), self.inception_5(x), self.inception_pool(x)], dim=1)
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        _, (hidden, _) = self.lstm(x)
        return self.projection(hidden[-1])


class AttnLOBEncoder(nn.Module):
    """Paper-faithful Attn-LOB encoder.

    Input shape is ``(batch, T, 40, 1)`` to match the paper/code. Internally the
    tensor is converted to PyTorch's ``NCHW`` format.
    """

    def __init__(self, output_dim: int = 64, attention_heads: int = 10, attention_key_dim: int = 16) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
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
        self.inception_3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), nn.LeakyReLU(0.01), nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"), nn.LeakyReLU(0.01))
        self.inception_5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), nn.LeakyReLU(0.01), nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"), nn.LeakyReLU(0.01))
        self.inception_pool = nn.Sequential(nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)), nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), nn.LeakyReLU(0.01))
        self.attention_heads = attention_heads
        self.attention_key_dim = attention_key_dim
        attention_dim = attention_heads * attention_key_dim
        self.query = nn.Linear(192, attention_dim)
        self.key = nn.Linear(192, attention_dim)
        self.value = nn.Linear(192, attention_dim)
        self.projection = nn.Linear(attention_dim, output_dim)

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        if lob_state.ndim != 4:
            raise ValueError(f"Expected LOB tensor with 4 dims, got {tuple(lob_state.shape)}")
        x = lob_state.permute(0, 3, 1, 2).contiguous()
        x = self.spatial(x)
        x = torch.cat([self.inception_3(x), self.inception_5(x), self.inception_pool(x)], dim=1)
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        query = x[:, -1:, :]
        q = self._split_heads(self.query(query))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attention_key_dim**0.5)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(x.shape[0], 1, -1)
        return self.projection(attended.squeeze(1))

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = tensor.shape
        return tensor.view(batch, steps, self.attention_heads, self.attention_key_dim).transpose(1, 2)


class PretrainClassifier(nn.Module):
    def __init__(self, encoder: nn.Module | None = None, output_dim: int = 64) -> None:
        super().__init__()
        self.encoder = encoder or AttnLOBEncoder()
        self.head = nn.Linear(output_dim, 3)

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(lob_state))


def build_lob_encoder(model_type: str, output_dim: int = 64, lookback: int = 50) -> nn.Module:
    normalized = model_type.strip().lower().replace("_", "").replace("-", "")
    if normalized in {"attnlob", "attn"}:
        return AttnLOBEncoder(output_dim=output_dim)
    if normalized in {"fclob", "fc"}:
        return FCLOBEncoder(output_dim=output_dim, lookback=lookback)
    if normalized in {"convlob", "conv"}:
        return ConvLOBEncoder(output_dim=output_dim)
    if normalized in {"deeplob", "deep"}:
        return DeepLOBEncoder(output_dim=output_dim)
    raise ValueError(f"Unknown pretrain_model_type: {model_type!r}")


def build_pretrain_classifier(model_type: str = "attnlob", output_dim: int = 64, lookback: int = 50) -> PretrainClassifier:
    return PretrainClassifier(build_lob_encoder(model_type, output_dim=output_dim, lookback=lookback), output_dim=output_dim)


class TradingBackbone(nn.Module):
    def __init__(
        self,
        encoder: AttnLOBEncoder | None = None,
        include_lob: bool = True,
        include_market: bool = True,
        include_agent: bool = True,
        alias_market_to_agent: bool = False,
    ) -> None:
        super().__init__()
        self.include_lob = include_lob
        self.include_market = include_market
        self.include_agent = include_agent
        self.alias_market_to_agent = alias_market_to_agent
        if alias_market_to_agent and include_market and not include_agent:
            raise ValueError("alias_market_to_agent requires include_agent=True")
        self.encoder = encoder or AttnLOBEncoder()
        in_dim = (64 if include_lob else 0) + (24 if include_market else 0) + (24 if include_agent else 0)
        self.fusion = nn.Sequential(nn.Linear(in_dim, 64), nn.LeakyReLU(0.01))

    def forward(self, lob_state: torch.Tensor | None, market_state: torch.Tensor | None, agent_state: torch.Tensor | None) -> torch.Tensor:
        pieces = []
        if self.include_lob:
            if lob_state is None:
                raise ValueError("lob_state is required")
            pieces.append(self.encoder(lob_state))
        if self.include_market:
            if self.alias_market_to_agent:
                if agent_state is None:
                    raise ValueError("agent_state is required when alias_market_to_agent=True")
                pieces.append(agent_state)
            else:
                if market_state is None:
                    raise ValueError("market_state is required")
                pieces.append(market_state)
        if self.include_agent:
            if agent_state is None:
                raise ValueError("agent_state is required")
            pieces.append(agent_state)
        return self.fusion(torch.cat(pieces, dim=1))


class PPOActorCritic(nn.Module):
    def __init__(self, backbone: TradingBackbone | None = None, initial_log_std: float = -1.5, initial_spread_bias: float = -0.70) -> None:
        super().__init__()
        self.backbone = backbone or TradingBackbone()
        self.actor_mean = nn.Linear(64, 2)
        self.actor_log_std = nn.Parameter(torch.full((2,), float(initial_log_std)))
        self.critic = nn.Linear(64, 1)
        self._initialize_actor(initial_spread_bias)

    def forward(self, lob_state: torch.Tensor, market_state: torch.Tensor, agent_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(lob_state, market_state, agent_state)
        mean = torch.tanh(self.actor_mean(features))
        value = self.critic(features).squeeze(-1)
        return mean, self.actor_log_std.expand_as(mean), value

    def _initialize_actor(self, initial_spread_bias: float) -> None:
        nn.init.zeros_(self.actor_mean.weight)
        with torch.no_grad():
            self.actor_mean.bias[0] = 0.0
            self.actor_mean.bias[1] = float(initial_spread_bias)


class DuelingDQN(nn.Module):
    def __init__(self, backbone: TradingBackbone | None = None, num_actions: int = 8) -> None:
        super().__init__()
        self.backbone = backbone or TradingBackbone()
        self.value = nn.Linear(64, 1)
        self.advantage = nn.Linear(64, num_actions)

    def forward(self, lob_state: torch.Tensor, market_state: torch.Tensor, agent_state: torch.Tensor) -> torch.Tensor:
        features = self.backbone(lob_state, market_state, agent_state)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
