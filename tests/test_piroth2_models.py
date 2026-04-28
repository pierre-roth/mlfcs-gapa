from __future__ import annotations

import torch

from piroth.models import AttnLOBEncoder, DuelingDQN, PPOActorCritic, TradingBackbone
from piroth.config import DiagnosticsConfig
from piroth.training import _linear_schedule, _optimizer_for_model


def test_attnlob_encoder_matches_paper_output_shape() -> None:
    encoder = AttnLOBEncoder()
    encoder.eval()

    with torch.no_grad():
        encoded = encoder(torch.zeros(2, 50, 40, 1))

    assert encoded.shape == (2, 64)


def test_trading_heads_accept_lob_dynamic_and_agent_state() -> None:
    backbone = TradingBackbone(encoder=AttnLOBEncoder())
    ppo = PPOActorCritic(backbone)
    dqn = DuelingDQN(TradingBackbone(encoder=AttnLOBEncoder()))
    lob = torch.zeros(3, 50, 40, 1)
    market = torch.zeros(3, 24)
    agent = torch.zeros(3, 24)

    mean, log_std, value = ppo(lob, market, agent)
    q_values = dqn(lob, market, agent)

    assert mean.shape == (3, 2)
    assert log_std.shape == (3, 2)
    assert value.shape == (3,)
    assert q_values.shape == (3, 8)


def test_ppo_actor_initializes_to_neutral_sub_penalty_spread() -> None:
    ppo = PPOActorCritic(TradingBackbone(encoder=AttnLOBEncoder()))
    lob = torch.zeros(1, 50, 40, 1)
    market = torch.zeros(1, 24)
    agent = torch.zeros(1, 24)

    mean, _, _ = ppo(lob, market, agent)

    assert torch.allclose(mean[0, 0], torch.tensor(0.0), atol=1e-6)
    assert -0.65 < float(mean[0, 1]) < -0.55


def test_ppo_actor_initializes_with_bounded_exploration() -> None:
    ppo = PPOActorCritic(TradingBackbone(encoder=AttnLOBEncoder()))

    assert torch.allclose(ppo.actor_log_std, torch.full((2,), -1.5))


def test_ppo_actor_accepts_initial_policy_overrides() -> None:
    ppo = PPOActorCritic(
        TradingBackbone(encoder=AttnLOBEncoder()),
        initial_log_std=-2.2,
        initial_spread_bias=-1.1,
    )
    lob = torch.zeros(1, 50, 40, 1)
    market = torch.zeros(1, 24)
    agent = torch.zeros(1, 24)

    mean, log_std, _ = ppo(lob, market, agent)

    assert torch.allclose(log_std, torch.full((1, 2), -2.2))
    assert -0.81 < float(mean[0, 1]) < -0.79


def test_entropy_schedule_decays_linearly() -> None:
    assert _linear_schedule(0.01, 0.001, 0, 5) == 0.01
    assert torch.isclose(torch.tensor(_linear_schedule(0.01, 0.001, 4, 5)), torch.tensor(0.001))


def test_optimizer_can_scale_encoder_and_backbone_learning_rates() -> None:
    model = PPOActorCritic(TradingBackbone(encoder=AttnLOBEncoder()))
    config = DiagnosticsConfig(
        torch_learning_rate=1e-3,
        torch_encoder_learning_rate_scale=0.1,
        torch_backbone_learning_rate_scale=0.5,
    )

    optimizer = _optimizer_for_model(model, config)
    learning_rates = sorted(group["lr"] for group in optimizer.param_groups)

    assert learning_rates == [1e-4, 5e-4, 1e-3]


def test_backbone_can_reproduce_author_market_state_alias_bug() -> None:
    backbone = TradingBackbone(
        encoder=AttnLOBEncoder(),
        include_lob=False,
        alias_market_to_agent=True,
    )
    market = torch.ones(2, 24)
    agent = torch.full((2, 24), 3.0)

    aliased = backbone(None, market, agent)
    direct = backbone(None, agent, agent)

    assert torch.allclose(aliased, direct)
