"""Gymnasium environment for paper-faithful C-PPO experiments."""

from __future__ import annotations

from dataclasses import asdict, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mlfcs_gapa.data.dynamic import DynamicStateCache
from mlfcs_gapa.data.features import normalize_lob_window
from mlfcs_gapa.data.schema import LobDataset, lob_columns
from mlfcs_gapa.env.actions import Quote, continuous_action_to_quote
from mlfcs_gapa.env.replay import Account, HistoricalReplay, compute_episode_metrics
from mlfcs_gapa.env.rewards import hybrid_reward
from mlfcs_gapa.paper.constants import PAPER


class PaperMarketMakingEnv(gym.Env):
    """Historical event-replay environment matching the paper's C-PPO setup."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: LobDataset,
        *,
        episode_start: int = 0,
        episode_events: int = PAPER.episode_events,
        latency_events: int = 1,
        normalize_actions: bool = False,
        random_episode_starts: bool = False,
        eta: float = PAPER.eta_dampened_pnl,
        zeta: float = PAPER.zeta_inventory_penalty,
        seed: int = 1,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.episode_start = episode_start
        self.episode_events = episode_events
        self.latency_events = latency_events
        self.normalize_actions = normalize_actions
        self.random_episode_starts = random_episode_starts
        self.eta = eta
        self.zeta = zeta
        self.rng = np.random.default_rng(seed)
        self.replay = HistoricalReplay(dataset, rng=self.rng)
        self.dynamic_state_cache = DynamicStateCache.from_dataset(dataset)
        self.lob_values = dataset.orderbook.select(lob_columns()).to_numpy()

        action_low = -1.0 if normalize_actions else 0.0
        self.action_space = spaces.Box(
            low=action_low, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "lob_state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=PAPER.lob_window_shape,
                    dtype=np.float32,
                ),
                "dynamic_state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
                ),
                "agent_state": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            }
        )
        self.account = Account()
        self.current_index = 0
        self.episode_end = 0
        self.values: list[float] = []
        self.inventories: list[int] = []
        self.quoted_spreads: list[float] = []
        self.trade_log: list[dict[str, float | int]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.replay.rng = self.rng

        options = options or {}
        episode_events = int(options.get("episode_events", self.episode_events))
        if "episode_start" in options:
            self.episode_start = int(options["episode_start"])
        elif self.random_episode_starts:
            max_start = max(0, self.dataset.orderbook.height - episode_events - 1)
            self.episode_start = int(self.rng.integers(0, max_start + 1)) if max_start else 0
        self.episode_end = min(
            self.episode_start + episode_events,
            self.dataset.orderbook.height - 1,
        )
        self.current_index = self.episode_start + PAPER.window_length + self.latency_events - 1
        if self.current_index >= self.episode_end:
            raise ValueError("episode is too short for the paper LOB window and latency")

        self.account = Account()
        self.values = [0.0]
        self.inventories = [0]
        self.quoted_spreads = []
        self.trade_log = []

        return self._observation(), {"current_index": self.current_index}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        decision_index = self._decision_index()
        decision_mid = self.replay.mid_price(decision_index)
        paper_action = self._paper_action(action)
        quote = continuous_action_to_quote(paper_action, decision_mid, self.account.inventory)
        quote = self._apply_inventory_cap(quote)

        current_mid = self.replay.mid_price(self.current_index)
        previous_value = self.account.value
        fill = self.replay.match(self.current_index, quote)
        self.account.apply_fill(fill, current_mid)
        reward_breakdown = hybrid_reward(
            delta_pnl=self.account.value - previous_value,
            mid_price=current_mid,
            trade_price=fill.trade_price,
            trade_volume=fill.trade_volume,
            inventory=self.account.inventory,
            eta=self.eta,
            zeta=self.zeta,
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
                "ask_price": quote.ask_price,
                "bid_price": quote.bid_price,
                "trade_price": fill.trade_price,
                "trade_volume": fill.trade_volume,
                "cash": self.account.cash,
                "inventory": self.account.inventory,
                "value": self.account.value,
                "action_bias": float(paper_action[0]),
                "action_spread": float(paper_action[1]),
                "reservation_price": quote.reservation_price,
                "spread": quote.spread,
            }
        )

        terminated = self.current_index + 1 >= self.episode_end
        info: dict[str, object] = {
            "fill": asdict(fill),
            "quote": asdict(quote),
            "paper_action": paper_action.tolist(),
            "reward": asdict(reward_breakdown),
        }
        if terminated:
            close_reward = self._close_episode(current_mid)
            reward += close_reward
            info["metrics"] = asdict(
                compute_episode_metrics(
                    self.values,
                    self.inventories,
                    self.quoted_spreads,
                    self.account.buy_notional,
                )
            )
            info["trade_log"] = self.trade_log
        else:
            self.current_index += 1

        return self._observation(), float(reward), terminated, False, info

    def _observation(self) -> dict[str, np.ndarray]:
        index = self._decision_index()
        start = index - PAPER.window_length + 1
        lob_state = normalize_lob_window(self.lob_values[start : index + 1]).astype(np.float32)
        progress = (self.current_index - self.episode_start) / max(
            1, self.episode_end - self.episode_start
        )
        agent_state = np.asarray(
            [self.account.inventory / PAPER.max_inventory, progress],
            dtype=np.float32,
        )
        return {
            "lob_state": lob_state,
            "dynamic_state": self.dynamic_state_cache.state(index),
            "agent_state": agent_state,
        }

    def _decision_index(self) -> int:
        return max(PAPER.window_length - 1, self.current_index - self.latency_events)

    def _paper_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if self.normalize_actions:
            action = np.clip(action, -1.0, 1.0)
            return (action + 1.0) / 2.0
        return action

    def _apply_inventory_cap(self, quote: Quote) -> Quote:
        if self.account.inventory <= -PAPER.max_inventory:
            return replace(quote, ask_price=0.0, ask_volume=0)
        if self.account.inventory >= PAPER.max_inventory:
            return replace(quote, bid_price=0.0, bid_volume=0)
        return quote

    def _close_episode(self, current_mid: float) -> float:
        previous_value = self.account.value
        fill = self.replay.close_position(self.current_index, self.account)
        self.account.apply_fill(fill, current_mid)
        reward = hybrid_reward(
            delta_pnl=self.account.value - previous_value,
            mid_price=current_mid,
            trade_price=fill.trade_price,
            trade_volume=fill.trade_volume,
            inventory=self.account.inventory,
            eta=self.eta,
            zeta=self.zeta,
        ).reward
        self.values.append(self.account.value)
        self.inventories.append(self.account.inventory)
        self.trade_log.append(
            {
                "index": self.current_index,
                "mid_price": current_mid,
                "ask_price": 0.0,
                "bid_price": 0.0,
                "trade_price": fill.trade_price,
                "trade_volume": fill.trade_volume,
                "cash": self.account.cash,
                "inventory": self.account.inventory,
                "value": self.account.value,
                "reservation_price": 0.0,
                "spread": 0.0,
            }
        )
        return reward
