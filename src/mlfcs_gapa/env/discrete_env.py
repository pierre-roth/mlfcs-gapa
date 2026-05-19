"""Discrete-action environment for the paper's D-DQN agent."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from gymnasium import spaces

from mlfcs_gapa.data.schema import LobDataset
from mlfcs_gapa.env.actions import Quote, legalize_quote_prices
from mlfcs_gapa.env.gym_env import PaperMarketMakingEnv
from mlfcs_gapa.env.replay import HistoricalReplay, compute_episode_metrics
from mlfcs_gapa.env.rewards import hybrid_reward
from mlfcs_gapa.paper.constants import PAPER


class PaperDiscreteMarketMakingEnv(PaperMarketMakingEnv):
    """Paper discrete-action replay environment.

    Actions 0-6 quote one buy and one sell order using best-price tick offsets.
    Action 7 immediately liquidates the current inventory with a market order.
    """

    def __init__(
        self,
        dataset: LobDataset,
        *,
        episode_start: int = 0,
        episode_events: int = PAPER.episode_events,
        latency_events: int = 1,
        seed: int = 1,
        tick_size: float = 0.01,
    ) -> None:
        super().__init__(
            dataset,
            episode_start=episode_start,
            episode_events=episode_events,
            latency_events=latency_events,
            seed=seed,
        )
        self.tick_size = tick_size
        self.action_space = spaces.Discrete(8)

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        action_id = int(action)
        if not self.action_space.contains(action_id):
            raise ValueError(f"discrete action must be in [0, 7], got {action_id}")

        decision_index = self._decision_index()
        current_mid = self.replay.mid_price(self.current_index)
        previous_value = self.account.value
        if action_id == 7:
            quote = _empty_quote()
            fill = self.replay.close_position(self.current_index, self.account)
        else:
            quote = discrete_action_to_quote(self.replay, decision_index, action_id, self.tick_size)
            quote = self._apply_inventory_cap(quote)
            fill = self.replay.match(self.current_index, quote)

        self.account.apply_fill(fill, current_mid)
        reward_breakdown = hybrid_reward(
            delta_pnl=self.account.value - previous_value,
            mid_price=current_mid,
            trade_price=fill.trade_price,
            trade_volume=fill.trade_volume,
            inventory=self.account.inventory,
        )
        reward = reward_breakdown.reward

        if quote.ask_price > 0 and quote.bid_price > 0:
            self.quoted_spreads.append(quote.ask_price - quote.bid_price)
        self.values.append(self.account.value)
        self.inventories.append(self.account.inventory)
        self.trade_log.append(
            {
                "index": self.current_index,
                "mid_price": current_mid,
                "action": action_id,
                "ask_price": quote.ask_price,
                "bid_price": quote.bid_price,
                "trade_price": fill.trade_price,
                "trade_volume": fill.trade_volume,
                "cash": self.account.cash,
                "inventory": self.account.inventory,
                "value": self.account.value,
                "reservation_price": quote.reservation_price,
                "spread": quote.spread,
            }
        )

        terminated = self.current_index + 1 >= self.episode_end
        info: dict[str, object] = {
            "fill": asdict(fill),
            "quote": asdict(quote),
            "reward": asdict(reward_breakdown),
        }
        if terminated:
            close_reward = self._close_episode(current_mid)
            reward += close_reward
            info["metrics"] = asdict(self._episode_metrics())
            info["trade_log"] = self.trade_log
        else:
            self.current_index += 1

        return self._observation(), float(reward), terminated, False, info

    def _episode_metrics(self):
        return compute_episode_metrics(
            self.values,
            self.inventories,
            self.quoted_spreads,
            self.account.buy_notional,
        )


def discrete_action_to_quote(
    replay: HistoricalReplay,
    decision_index: int,
    action_id: int,
    tick_size: float = 0.01,
) -> Quote:
    """Convert one of the paper's seven quote actions to a bid/ask quote."""

    offsets = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
        4: (0, 2),
        5: (2, 0),
        6: (2, 2),
    }
    if action_id not in offsets:
        raise ValueError("quote action must be in [0, 6]")
    ask_offset, bid_offset = offsets[action_id]
    bid1, ask1 = replay.best_bid_ask(decision_index)
    ask, bid = legalize_quote_prices(
        ask1 + ask_offset * tick_size,
        bid1 - bid_offset * tick_size,
        tick_size=tick_size,
    )
    return Quote(
        ask_price=ask,
        ask_volume=-PAPER.minimum_trade_unit,
        bid_price=bid,
        bid_volume=PAPER.minimum_trade_unit,
        reservation_price=(ask + bid) / 2.0,
        spread=ask - bid,
    )


def _empty_quote() -> Quote:
    return Quote(
        ask_price=0.0,
        ask_volume=0,
        bid_price=0.0,
        bid_volume=0,
        reservation_price=0.0,
        spread=0.0,
    )
