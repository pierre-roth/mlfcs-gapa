import torch

from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.env.gym_env import PaperMarketMakingEnv
from mlfcs_gapa.models.attn_lob import AttnLOBClassifier
from mlfcs_gapa.training.ppo import AttnLOBFeatureExtractor


def test_attn_lob_feature_extractor_emits_policy_features() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=91))
    env = PaperMarketMakingEnv(dataset, episode_events=120, seed=1)
    observation, _ = env.reset()
    extractor = AttnLOBFeatureExtractor(env.observation_space, features_dim=32)

    batch = {key: torch.from_numpy(value).unsqueeze(0) for key, value in observation.items()}
    features = extractor(batch)

    assert tuple(features.shape) == (1, 32)
    assert torch.isfinite(features).all()


def test_attn_lob_feature_extractor_loads_classifier_encoder_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "attn_lob_classifier.pt"
    classifier = AttnLOBClassifier()
    torch.save({"model": "Attn-LOB", "state_dict": classifier.state_dict()}, checkpoint)

    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=92))
    env = PaperMarketMakingEnv(dataset, episode_events=120, seed=1)
    extractor = AttnLOBFeatureExtractor(
        env.observation_space,
        features_dim=32,
        encoder_checkpoint=str(checkpoint),
        freeze_encoder=True,
    )

    assert not any(parameter.requires_grad for parameter in extractor.lob_encoder.parameters())


def test_feature_extractor_supports_paper_ablation_modes() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=93))
    env = PaperMarketMakingEnv(dataset, episode_events=120, seed=1)
    observation, _ = env.reset()
    batch = {key: torch.from_numpy(value).unsqueeze(0) for key, value in observation.items()}

    mlp_extractor = AttnLOBFeatureExtractor(
        env.observation_space,
        features_dim=16,
        lob_mode="mlp",
        use_dynamic_state=False,
        use_agent_state=True,
    )
    no_lob_extractor = AttnLOBFeatureExtractor(
        env.observation_space,
        features_dim=16,
        lob_mode="none",
        use_dynamic_state=True,
        use_agent_state=True,
    )

    assert tuple(mlp_extractor(batch).shape) == (1, 16)
    assert tuple(no_lob_extractor(batch).shape) == (1, 16)
