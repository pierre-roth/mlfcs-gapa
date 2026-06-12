"""Tabular RL baselines from Lim-Gorse and Zhong et al."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np

from mlfcs_gapa.data.schema import LobDataset
from mlfcs_gapa.env.actions import Quote, legalize_quote_prices
from mlfcs_gapa.env.baselines import evaluate_quote_strategy
from mlfcs_gapa.env.replay import Account, HistoricalReplay
from mlfcs_gapa.paper.constants import PAPER


StateKey = tuple[int, ...]


class StateEncoder(Protocol):
    def encode(
        self,
        dataset: LobDataset,
        replay: HistoricalReplay,
        account: Account,
        index: int,
        episode_progress: float,
    ) -> StateKey: ...


@dataclass(frozen=True)
class InventoryTimeEncoder:
    """Inventory/time state from Lim and Gorse."""

    time_bins: int = 12
    small_inventory_lots: int = 2
    medium_inventory_lots: int = 4

    def encode(
        self,
        dataset: LobDataset,
        replay: HistoricalReplay,
        account: Account,
        index: int,
        episode_progress: float,
    ) -> StateKey:
        del dataset, replay, index
        lots = account.inventory / PAPER.minimum_trade_unit
        if lots == 0:
            inventory_bin = 0
        else:
            sign = 1 if lots > 0 else -1
            abs_lots = abs(lots)
            if abs_lots <= self.small_inventory_lots:
                inventory_bin = sign * 1
            elif abs_lots <= self.medium_inventory_lots:
                inventory_bin = sign * 2
            else:
                inventory_bin = sign * 3
        remaining = 1.0 - episode_progress
        time_bin = int(np.clip(np.ceil(remaining * self.time_bins), 1, self.time_bins))
        return (inventory_bin, time_bin)


@dataclass(frozen=True)
class LobRlEncoder:
    """State aggregation from Zhong et al. used by the paper's LOB-RL baseline."""

    mid_window: int = 10
    mid_fraction_threshold: float = 0.35
    inventory_threshold_lots: int = 4
    pnl_threshold: float = 0.0

    def encode(
        self,
        dataset: LobDataset,
        replay: HistoricalReplay,
        account: Account,
        index: int,
        episode_progress: float,
    ) -> StateKey:
        del dataset, episode_progress
        bid_depth = float(replay.bid_volumes[index, 0] + replay.bid_volumes[index, 1])
        ask_depth = float(replay.ask_volumes[index, 0] + replay.ask_volumes[index, 1])
        bid_speed = int(replay.market_sell_volume[index] > bid_depth)
        ask_speed = int(replay.market_buy_volume[index] > ask_depth)
        mid_frac = self._mid_change_fraction(replay, index)
        mid_fraction = _signed_bucket(mid_frac, self.mid_fraction_threshold)
        inv_sign = _signed_bucket(
            account.inventory / PAPER.minimum_trade_unit,
            self.inventory_threshold_lots,
        )
        cum_pnl = int(account.value <= self.pnl_threshold)
        return (bid_speed, ask_speed, mid_fraction, inv_sign, cum_pnl)

    def _mid_change_fraction(self, replay: HistoricalReplay, index: int) -> float:
        if index <= 0:
            return 0.0
        start_prev = max(0, index - 2 * self.mid_window)
        end_prev = max(0, index - self.mid_window)
        start_curr = end_prev
        end_curr = index
        if end_prev <= start_prev or end_curr <= start_curr:
            return 0.0
        prev = replay.mid_prices[start_prev:end_prev]
        curr = replay.mid_prices[start_curr:end_curr]
        if len(prev) == 0 or len(curr) == 0:
            return 0.0
        combined = np.concatenate([prev, curr])
        price_range = combined.max() - combined.min()
        if price_range == 0:
            return 0.0
        return float((curr.mean() - prev.mean()) / price_range)


@dataclass(frozen=True)
class OffsetActionSpace:
    """Lim-Gorse 9-action offset space: bid/ask offsets in {0, 1, 2} ticks."""

    tick_size: float = 0.01

    @property
    def actions(self) -> tuple[tuple[int, int], ...]:
        return tuple(product((0, 1, 2), (0, 1, 2)))

    def quote(self, replay: HistoricalReplay, index: int, action_id: int) -> Quote:
        bid_offset, ask_offset = self.actions[action_id]
        bid1, ask1 = replay.best_bid_ask(index)
        ask, bid = legalize_quote_prices(
            ask1 + ask_offset * self.tick_size,
            bid1 - bid_offset * self.tick_size,
            tick_size=self.tick_size,
        )
        return Quote(
            ask_price=ask,
            ask_volume=-PAPER.minimum_trade_unit,
            bid_price=bid,
            bid_volume=PAPER.minimum_trade_unit,
            reservation_price=(ask + bid) / 2.0,
            spread=ask - bid,
        )

    def admissible_actions(self, state: StateKey) -> tuple[int, ...]:
        del state
        return tuple(range(len(self.actions)))


@dataclass(frozen=True)
class BestBidAskActionSpace:
    """Zhong et al. action pairs: rest at best bid and/or best ask."""

    @property
    def actions(self) -> tuple[tuple[int, int], ...]:
        return ((0, 0), (0, 1), (1, 0), (1, 1))

    def quote(self, replay: HistoricalReplay, index: int, action_id: int) -> Quote:
        bid_active, ask_active = self.actions[action_id]
        bid1, ask1 = replay.best_bid_ask(index)
        ask = ask1 if ask_active else 0.0
        bid = bid1 if bid_active else 0.0
        return Quote(
            ask_price=ask,
            ask_volume=-PAPER.minimum_trade_unit if ask_active else 0,
            bid_price=bid,
            bid_volume=PAPER.minimum_trade_unit if bid_active else 0,
            reservation_price=(ask + bid) / 2.0 if ask_active and bid_active else 0.0,
            spread=ask - bid if ask_active and bid_active else 0.0,
        )

    def admissible_actions(self, state: StateKey) -> tuple[int, ...]:
        """Zhong et al. inventory-side action restriction.

        The fourth state component is invSign. When the inventory imbalance is
        large, the lookup policy may only quote on the side that reduces it, or
        stay out of the market.
        """

        inv_sign = state[3]
        if inv_sign == -2:
            return (0, 2)  # no quote or bid only
        if inv_sign == 2:
            return (0, 1)  # no quote or ask only
        return tuple(range(len(self.actions)))


@dataclass
class TabularPolicyStrategy:
    name: str
    encoder: StateEncoder
    action_space: OffsetActionSpace | BestBidAskActionSpace
    q_table: dict[StateKey, np.ndarray]

    def quote(
        self,
        replay: HistoricalReplay,
        account: Account,
        decision_index: int,
        episode_progress: float,
    ) -> Quote:
        state = self.encoder.encode(
            replay.dataset, replay, account, decision_index, episode_progress
        )
        values = self.q_table.get(state)
        admissible = self.action_space.admissible_actions(state)
        if values is None:
            action_id = admissible[-1]
        else:
            action_id = _argmax_admissible(values, admissible)
        return self.action_space.quote(replay, decision_index, action_id)


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 20
    episode_events: int = 500
    learning_rate: float = 0.1
    discount: float = 0.99
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    seed: int = 1


def train_tabular_q_strategy(
    dataset: LobDataset,
    *,
    name: str,
    encoder: StateEncoder,
    action_space: OffsetActionSpace | BestBidAskActionSpace,
    config: QLearningConfig = QLearningConfig(),
) -> TabularPolicyStrategy:
    """Train a small tabular Q-learning policy on the replay environment."""

    rng = np.random.default_rng(config.seed)
    q_table: dict[StateKey, np.ndarray] = {}
    max_start = max(0, dataset.orderbook.height - config.episode_events - 1)

    for episode in range(config.episodes):
        episode_start = int(rng.integers(0, max_start + 1)) if max_start else 0
        first_index = episode_start + PAPER.window_length
        episode_end = min(episode_start + config.episode_events, dataset.orderbook.height - 1)
        replay = HistoricalReplay(dataset, rng=rng)
        account = Account()
        epsilon = _linear_decay(config.epsilon_start, config.epsilon_end, episode, config.episodes)

        previous_value = account.value
        for index in range(first_index, episode_end):
            progress = (index - episode_start) / max(1, episode_end - episode_start)
            state = encoder.encode(dataset, replay, account, index, progress)
            values = q_table.setdefault(
                state, np.zeros(len(action_space.actions), dtype=np.float64)
            )
            admissible = action_space.admissible_actions(state)
            if rng.random() < epsilon:
                action_id = int(rng.choice(admissible))
            else:
                action_id = _argmax_admissible(values, admissible)

            quote = action_space.quote(replay, index, action_id)
            fill = replay.match(index, quote)
            mid = replay.mid_price(index)
            account.apply_fill(fill, mid)
            reward = account.value - previous_value
            previous_value = account.value

            next_index = min(index + 1, episode_end - 1)
            next_progress = (next_index - episode_start) / max(1, episode_end - episode_start)
            next_state = encoder.encode(dataset, replay, account, next_index, next_progress)
            next_values = q_table.setdefault(
                next_state, np.zeros(len(action_space.actions), dtype=np.float64)
            )
            next_admissible = action_space.admissible_actions(next_state)
            td_target = reward + config.discount * float(
                next_values[_argmax_admissible(next_values, next_admissible)]
            )
            values[action_id] = values[action_id] + config.learning_rate * (
                td_target - values[action_id]
            )

        close_fill = replay.close_position(episode_end - 1, account)
        mid = replay.mid_price(episode_end - 1)
        account.apply_fill(close_fill, mid)

    return TabularPolicyStrategy(
        name=name,
        encoder=encoder,
        action_space=action_space,
        q_table=q_table,
    )


def train_and_evaluate_tabular_baseline(
    dataset: LobDataset,
    *,
    name: str,
    encoder: StateEncoder,
    action_space: OffsetActionSpace | BestBidAskActionSpace,
    config: QLearningConfig = QLearningConfig(),
    latency_events: int = 1,
) -> tuple[dict[str, float], list[dict[str, float | int | str]], TabularPolicyStrategy]:
    strategy = train_tabular_q_strategy(
        dataset,
        name=name,
        encoder=encoder,
        action_space=action_space,
        config=config,
    )
    metrics, log_rows = evaluate_quote_strategy(
        dataset,
        strategy,
        episode_events=config.episode_events,
        latency_events=latency_events,
        seed=config.seed,
    )
    return metrics, log_rows, strategy


def _signed_bucket(value: float, threshold: float) -> int:
    if value == 0:
        return 0
    sign = 1 if value > 0 else -1
    abs_value = abs(value)
    if abs_value > threshold:
        return sign * 2
    return sign * 1


def _linear_decay(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    fraction = step / (total_steps - 1)
    return float(start + fraction * (end - start))


def _argmax_admissible(values: np.ndarray, admissible: tuple[int, ...]) -> int:
    if len(admissible) == 0:
        raise ValueError("admissible action set cannot be empty")
    local_id = int(np.argmax(values[list(admissible)]))
    return int(admissible[local_id])
