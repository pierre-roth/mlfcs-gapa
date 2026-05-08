from __future__ import annotations

import math

import torch
from torch import nn


PAPER_PRETRAIN_LOOKBACKS = {
    "fclob": 100,
    "convlob": 1024,
    "deeplob": 100,
    "attnlob": 50,
}

PAPER_PRETRAIN_INPUTS = {
    "fclob": "4000 x 1",
    "convlob": "1024 x 40",
    "deeplob": "100 x 40",
    "attnlob": "50 x 40",
}


class FCLOBEncoder(nn.Module):
    """FC-LOB encoder matching the paper Table I parameter count.

    The authors' released ``get_fclob_model`` creates three Dense layers but
    wires each one directly to the flattened LOB input. The returned encoder is
    therefore effectively ``Flatten -> Dense(64)``, which is also the only FC
    interpretation that matches Table I: ``4000 * 64 + 64 = 256,064``.
    """

    def __init__(self, output_dim: int = 64, lookback: int = 100, features: int = 40) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * features, output_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        if lob_state.ndim != 4:
            raise ValueError(f"Expected LOB tensor with 4 dims, got {tuple(lob_state.shape)}")
        return self.net(lob_state)


class ConvLOBEncoder(nn.Module):
    """Dilated fully convolutional LOB encoder matching Table I dimensions.

    The paper only states that Conv-LOB is a fully convolutional model with
    dilated convolutions, similar to WaveNet, and reports a 1024x40 input with
    172,320 parameters. This implementation keeps that long temporal input and
    exact parameter count while using a small dilated temporal Conv1d stack.
    """

    def __init__(self, output_dim: int = 64) -> None:
        if output_dim != 64:
            raise ValueError("Paper Conv-LOB uses a fixed 64-dimensional latent output.")
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Conv1d(40, 56, kernel_size=4, padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv1d(56, 56, kernel_size=12, dilation=1, padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv1d(56, 56, kernel_size=12, dilation=2, padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv1d(56, 56, kernel_size=12, dilation=4, padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv1d(56, 64, kernel_size=14, padding="same"),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        if lob_state.ndim != 4:
            raise ValueError(f"Expected LOB tensor with 4 dims, got {tuple(lob_state.shape)}")
        if lob_state.shape[2] != 40:
            raise ValueError(f"Expected 40 LOB features, got {lob_state.shape[2]}")
        x = lob_state.squeeze(-1).permute(0, 2, 1).contiguous()
        return self.net(x)


class _KerasStyleLSTM(nn.Module):
    """Single-layer LSTM with one bias vector, matching Keras parameter counts."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch = sequence.shape[0]
        hidden = sequence.new_zeros(batch, self.hidden_size)
        cell = sequence.new_zeros(batch, self.hidden_size)
        for step in range(sequence.shape[1]):
            gates = (
                torch.matmul(sequence[:, step, :], self.weight_ih.t())
                + torch.matmul(hidden, self.weight_hh.t())
                + self.bias
            )
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)
            cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(cell_gate)
            hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden


class DeepLOBEncoder(nn.Module):
    """DeepLOB-style CNN plus LSTM encoder matching Table I.

    The 139,168 Table I count is reproduced by the authors' LOB
    convolution/inception front-end followed by a Keras-style LSTM with one bias
    vector and no extra projection layer.
    """

    def __init__(self, output_dim: int = 64, hidden_dim: int = 64) -> None:
        if output_dim != 64 or hidden_dim != 64:
            raise ValueError("Paper DeepLOB uses fixed output_dim=64 and hidden_dim=64.")
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
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same"),
            nn.LeakyReLU(0.01),
        )
        self.inception_3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), nn.LeakyReLU(0.01), nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"), nn.LeakyReLU(0.01))
        self.inception_5 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), nn.LeakyReLU(0.01), nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"), nn.LeakyReLU(0.01))
        self.inception_pool = nn.Sequential(nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)), nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"), nn.LeakyReLU(0.01))
        self.lstm = _KerasStyleLSTM(input_size=192, hidden_size=hidden_dim)

    def forward(self, lob_state: torch.Tensor) -> torch.Tensor:
        if lob_state.ndim != 4:
            raise ValueError(f"Expected LOB tensor with 4 dims, got {tuple(lob_state.shape)}")
        x = lob_state.permute(0, 3, 1, 2).contiguous()
        x = self.spatial(x)
        x = torch.cat([self.inception_3(x), self.inception_5(x), self.inception_pool(x)], dim=1)
        x = x.squeeze(-1).permute(0, 2, 1).contiguous()
        return self.lstm(x)


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


def paper_pretrain_model_slug(model_type: str) -> str:
    normalized = model_type.strip().lower().replace("_", "").replace("-", "")
    aliases = {"attn": "attnlob", "fc": "fclob", "conv": "convlob", "deep": "deeplob"}
    return aliases.get(normalized, normalized)


def paper_pretrain_lookback(model_type: str) -> int:
    model_slug = paper_pretrain_model_slug(model_type)
    return PAPER_PRETRAIN_LOOKBACKS[model_slug]


def paper_pretrain_input(model_type: str) -> str:
    model_slug = paper_pretrain_model_slug(model_type)
    return PAPER_PRETRAIN_INPUTS[model_slug]


def build_lob_encoder(model_type: str, output_dim: int = 64, lookback: int | None = None) -> nn.Module:
    normalized = paper_pretrain_model_slug(model_type)
    if normalized in {"attnlob", "attn"}:
        return AttnLOBEncoder(output_dim=output_dim)
    if normalized == "fclob":
        return FCLOBEncoder(output_dim=output_dim, lookback=lookback or paper_pretrain_lookback(normalized))
    if normalized == "convlob":
        return ConvLOBEncoder(output_dim=output_dim)
    if normalized == "deeplob":
        return DeepLOBEncoder(output_dim=output_dim)
    raise ValueError(f"Unknown pretrain_model_type: {model_type!r}")


def build_pretrain_classifier(model_type: str = "attnlob", output_dim: int = 64, lookback: int | None = None) -> PretrainClassifier:
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
