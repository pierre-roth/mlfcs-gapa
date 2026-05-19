import numpy as np

from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.env.discrete_env import PaperDiscreteMarketMakingEnv, discrete_action_to_quote
from mlfcs_gapa.env.replay import HistoricalReplay


def test_discrete_action_mapping_matches_paper_offsets() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=120, seed=93))
    replay = HistoricalReplay(dataset)
    row = dataset.orderbook.row(60, named=True)

    action_0 = discrete_action_to_quote(replay, 60, 0)
    action_6 = discrete_action_to_quote(replay, 60, 6)

    assert action_0.ask_price == row["ask1_price"]
    assert action_0.bid_price == row["bid1_price"]
    assert np.isclose(action_6.ask_price, row["ask1_price"] + 0.02)
    assert np.isclose(action_6.bid_price, row["bid1_price"] - 0.02)


def test_discrete_environment_steps_and_reports_terminal_metrics() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=94))
    env = PaperDiscreteMarketMakingEnv(dataset, episode_events=100, seed=1)
    observation, _ = env.reset()

    assert set(observation) == {"lob_state", "dynamic_state", "agent_state"}
    done = False
    info: dict[str, object] = {}
    while not done:
        _, reward, terminated, truncated, info = env.step(0)
        assert np.isfinite(reward)
        done = terminated or truncated

    assert "metrics" in info
    assert "trade_log" in info


def test_discrete_environment_close_action_is_valid() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=95))
    env = PaperDiscreteMarketMakingEnv(dataset, episode_events=100, seed=1)
    env.reset()

    _, reward, terminated, truncated, info = env.step(7)

    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["quote"]["ask_price"] == 0.0  # type: ignore[index]
