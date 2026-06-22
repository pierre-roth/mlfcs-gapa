"""Paper environment wrapper with AS behavioral constraints.

The base replication environment remains unchanged. This extension subclass
adds optional soft or hard AS guidance in paper action coordinates.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from mlfcs_gapa.env.actions import continuous_action_to_quote
from mlfcs_gapa.env.baselines import AvellanedaStoikovStrategy
from mlfcs_gapa.env.gym_env import PaperMarketMakingEnv
from mlfcs_gapa.env.replay import compute_episode_metrics
from mlfcs_gapa.env.rewards import hybrid_reward
from mlfcs_gapa.extensions.as_guidance import (
    ASGuidanceConfig,
    apply_hard_as_window,
    as_divergence_penalty,
    as_teacher_action,
    env_action_to_paper_action,
    make_as_strategy,
    quote_divergence_penalty,
)
from mlfcs_gapa.paper.constants import PAPER


class ASGuidedMarketMakingEnv(PaperMarketMakingEnv):
    """Market-making env with optional AS soft or hard action guidance."""

    def __init__(
        self,
        *args,
        as_strategy: AvellanedaStoikovStrategy | None = None,
        guidance: ASGuidanceConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.as_strategy = as_strategy or make_as_strategy(
            self.dataset, episode_events=self.episode_events
        )
        self.guidance = guidance or ASGuidanceConfig()

    def step(self, action: np.ndarray):
        decision_index = self._decision_index()
        decision_mid = self.replay.mid_price(decision_index)
        progress = (self.current_index - self.episode_start) / max(
            1, self.episode_end - self.episode_start
        )
        raw_paper_action = env_action_to_paper_action(
            action, normalize_actions=self.normalize_actions
        )
        teacher_action = as_teacher_action(
            self.as_strategy,
            self.replay,
            self.account,
            decision_index,
            progress,
            max_bias=self.guidance.max_bias,
            max_spread=self.guidance.max_spread,
        )

        paper_action = raw_paper_action
        if self.guidance.mode == "hard":
            paper_action = apply_hard_as_window(
                raw_paper_action,
                teacher_action,
                hard_window_bias=self.guidance.hard_window_bias,
                hard_window_spread=self.guidance.hard_window_spread,
            )
        elif self.guidance.mode not in {"none", "soft"}:
            raise ValueError("guidance mode must be one of: none, soft, hard")

        quote = continuous_action_to_quote(
            paper_action,
            decision_mid,
            self.account.inventory,
            max_bias=self.guidance.max_bias,
            max_spread=self.guidance.max_spread,
        )
        quote = self._apply_inventory_cap(quote)

        current_mid = self.replay.mid_price(self.current_index)
        previous_value = self.account.value
        fill = self.replay.match(self.current_index, quote)
        self.account.apply_fill(fill, current_mid)
        delta_pnl = self.account.value - previous_value
        reward_breakdown = hybrid_reward(
            delta_pnl=delta_pnl,
            mid_price=current_mid,
            trade_price=fill.trade_price,
            trade_volume=fill.trade_volume,
            inventory=self.account.inventory,
            eta=self.eta,
            zeta=self.zeta,
        )
        reward = self._base_reward(delta_pnl=delta_pnl, hybrid=reward_breakdown.reward)
        guidance_penalty = 0.0
        guidance_penalty_scale = self._soft_penalty_scale(progress)
        if self.guidance.mode == "soft":
            if self.guidance.penalty_space == "quote":
                teacher_quote = continuous_action_to_quote(
                    teacher_action,
                    decision_mid,
                    self.account.inventory,
                    max_bias=self.guidance.max_bias,
                    max_spread=self.guidance.max_spread,
                )
                guidance_penalty = quote_divergence_penalty(
                    np.asarray([quote.ask_price, quote.bid_price], dtype=np.float64),
                    np.asarray(
                        [teacher_quote.ask_price, teacher_quote.bid_price],
                        dtype=np.float64,
                    ),
                    soft_penalty=guidance_penalty_scale,
                    scale=self.guidance.max_spread,
                    penalty_norm=self.guidance.penalty_norm,
                    huber_delta=self.guidance.huber_delta,
                    adaptive_target=self.guidance.adaptive_target,
                )
            else:
                guidance_penalty = as_divergence_penalty(
                    raw_paper_action,
                    teacher_action,
                    soft_penalty=guidance_penalty_scale,
                    bias_weight=self.guidance.bias_weight,
                    spread_weight=self.guidance.spread_weight,
                    penalty_norm=self.guidance.penalty_norm,
                    huber_delta=self.guidance.huber_delta,
                    adaptive_target=self.guidance.adaptive_target,
                )
            reward -= guidance_penalty

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
                "raw_action_bias": float(raw_paper_action[0]),
                "raw_action_spread": float(raw_paper_action[1]),
                "action_bias": float(paper_action[0]),
                "action_spread": float(paper_action[1]),
                "teacher_action_bias": float(teacher_action[0]),
                "teacher_action_spread": float(teacher_action[1]),
                "as_guidance_penalty": guidance_penalty,
                "as_guidance_penalty_scale": guidance_penalty_scale,
                "base_reward": self.guidance.base_reward,
                "profit_reward": delta_pnl,
                "hybrid_reward": reward_breakdown.reward,
                "reservation_price": quote.reservation_price,
                "spread": quote.spread,
            }
        )

        terminated = self.current_index + 1 >= self.episode_end
        info: dict[str, object] = {
            "fill": asdict(fill),
            "quote": asdict(quote),
            "paper_action": paper_action.tolist(),
            "raw_paper_action": raw_paper_action.tolist(),
            "teacher_action": teacher_action.tolist(),
            "as_guidance_penalty": guidance_penalty,
            "as_guidance_penalty_scale": guidance_penalty_scale,
            "base_reward": self.guidance.base_reward,
            "profit_reward": delta_pnl,
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

    def _base_reward(self, *, delta_pnl: float, hybrid: float) -> float:
        if self.guidance.base_reward == "profit":
            return float(delta_pnl)
        if self.guidance.base_reward == "paper_hybrid":
            return float(hybrid)
        raise ValueError("base_reward must be one of: paper_hybrid, profit")

    def _soft_penalty_scale(self, progress: float) -> float:
        start = self.guidance.soft_penalty
        end = self.guidance.soft_penalty_end
        if end is None or self.guidance.penalty_schedule == "constant":
            return float(start)
        if self.guidance.penalty_schedule == "episode_decay":
            return float(start + (end - start) * np.clip(progress, 0.0, 1.0))
        if self.guidance.penalty_schedule == "episode_warmup":
            return float(start + (end - start) * np.clip(progress, 0.0, 1.0))
        raise ValueError("unknown AS penalty schedule")

    def _apply_inventory_cap(self, quote):
        if self.account.inventory <= -PAPER.max_inventory:
            return replace(quote, ask_price=0.0, ask_volume=0)
        if self.account.inventory >= PAPER.max_inventory:
            return replace(quote, bid_price=0.0, bid_volume=0)
        return quote

    def _close_episode(self, current_mid: float) -> float:
        previous_value = self.account.value
        fill = self.replay.close_position(self.current_index, self.account)
        self.account.apply_fill(fill, current_mid)
        delta_pnl = self.account.value - previous_value
        reward_breakdown = hybrid_reward(
            delta_pnl=delta_pnl,
            mid_price=current_mid,
            trade_price=fill.trade_price,
            trade_volume=fill.trade_volume,
            inventory=self.account.inventory,
            eta=self.eta,
            zeta=self.zeta,
        )
        reward = self._base_reward(delta_pnl=delta_pnl, hybrid=reward_breakdown.reward)
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
                "as_guidance_penalty": 0.0,
                "base_reward": self.guidance.base_reward,
                "profit_reward": delta_pnl,
                "hybrid_reward": reward_breakdown.reward,
                "reservation_price": 0.0,
                "spread": 0.0,
            }
        )
        return reward
