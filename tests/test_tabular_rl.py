import numpy as np

from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.env.replay import Account, HistoricalReplay
from mlfcs_gapa.env.tabular_rl import (
    BestBidAskActionSpace,
    InventoryTimeEncoder,
    LobRlEncoder,
    OffsetActionSpace,
    QLearningConfig,
    train_and_evaluate_tabular_baseline,
    train_tabular_q_strategy,
)


def test_inventory_time_encoder_bins_inventory_and_remaining_time() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=120, seed=81))
    replay = HistoricalReplay(dataset)
    encoder = InventoryTimeEncoder(time_bins=12)

    flat = encoder.encode(dataset, replay, Account(inventory=0), 50, 0.0)
    long = encoder.encode(dataset, replay, Account(inventory=500), 50, 0.8)
    short = encoder.encode(dataset, replay, Account(inventory=-300), 50, 0.2)

    assert flat == (0, 12)
    assert long[0] == 3
    assert long[1] < flat[1]
    assert short[0] == -2


def test_lob_rl_encoder_matches_zhong_state_cardinality_components() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=150, seed=82))
    replay = HistoricalReplay(dataset)
    encoder = LobRlEncoder(mid_window=5)
    account = Account(inventory=500, value=-1.0)

    state = encoder.encode(dataset, replay, account, 80, 0.5)

    assert len(state) == 5
    assert state[0] in {0, 1}
    assert state[1] in {0, 1}
    assert state[2] in {-2, -1, 0, 1, 2}
    assert state[3] in {-2, -1, 0, 1, 2}
    assert state[4] in {0, 1}


def test_offset_action_space_quotes_around_best_prices() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=120, seed=83))
    replay = HistoricalReplay(dataset)
    action_space = OffsetActionSpace(tick_size=0.01)

    quote = action_space.quote(replay, 60, 8)
    bid1, ask1 = replay.best_bid_ask(60)

    assert len(action_space.actions) == 9
    assert quote.ask_price >= ask1
    assert quote.bid_price <= bid1


def test_best_bid_ask_action_space_restricts_large_imbalances() -> None:
    action_space = BestBidAskActionSpace()

    assert len(action_space.actions) == 4
    assert action_space.admissible_actions((0, 0, 0, -2, 0)) == (0, 2)
    assert action_space.admissible_actions((0, 0, 0, 2, 0)) == (0, 1)
    assert action_space.admissible_actions((0, 0, 0, 0, 0)) == (0, 1, 2, 3)


def test_train_tabular_q_strategy_returns_lookup_policy() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=84))
    strategy = train_tabular_q_strategy(
        dataset,
        name="Inv-RL",
        encoder=InventoryTimeEncoder(),
        action_space=OffsetActionSpace(),
        config=QLearningConfig(episodes=3, episode_events=100, seed=2),
    )

    assert strategy.name == "Inv-RL"
    assert len(strategy.q_table) > 0
    assert all(np.isfinite(values).all() for values in strategy.q_table.values())


def test_train_and_evaluate_tabular_baseline_returns_metrics_and_log() -> None:
    dataset = generate_synthetic_lob_day(SyntheticLobConfig(n_events=180, seed=85))
    metrics, log_rows, strategy = train_and_evaluate_tabular_baseline(
        dataset,
        name="LOB-RL",
        encoder=LobRlEncoder(mid_window=5),
        action_space=BestBidAskActionSpace(),
        config=QLearningConfig(episodes=3, episode_events=100, seed=3),
    )

    assert strategy.name == "LOB-RL"
    assert metrics["method"] == "LOB-RL"
    assert len(log_rows) > 0
    assert np.isfinite(metrics["pnl"])
