from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from .config import RLTrainConfig
from .data import DayData
from .metrics import EpisodeResult
from .utils import price_legal_check


@dataclass
class Observation:
    lob: np.ndarray | None
    flat: np.ndarray
    
@dataclass
class Fill:
    price: float
    volume: float
    taker: bool = False
 


class MarketMakingEnv:
    def __init__(
        self,
        day: DayData,
        config: RLTrainConfig,
        state_mode: str = "full",
        wo_lob_state: bool = False,
        wo_dynamic_state: bool = False,
        reward_mode: str = "hybrid",
    ) -> None:
        self.day = day
        self.config = config
        self.state_mode = state_mode
        self.wo_lob_state = wo_lob_state
        self.wo_dynamic_state = wo_dynamic_state
        self.reward_mode = reward_mode
        self.decision_indices = np.arange(config.lookback - 1 + config.latency, len(day.midprice), dtype=np.int64)
        if len(self.decision_indices) == 0:
            raise RuntimeError(f"No tradable indices for {day.symbol} {day.day}")
        self.num_discrete_actions = 8
        self.eval_episode_index: int | None = None
        self.eval_context_key: str | None = None

    def spawn(self) -> "MarketMakingEnv":
        return MarketMakingEnv(
            self.day,
            self.config,
            state_mode=self.state_mode,
            wo_lob_state=self.wo_lob_state,
            wo_dynamic_state=self.wo_dynamic_state,
            reward_mode=self.reward_mode,
        )

    def _episode_ranges(self) -> list[tuple[int, int]]:
        length = self.config.episode_length
        segments = []
        for start in range(0, len(self.decision_indices), length):
            end = min(start + length, len(self.decision_indices))
            if end - start > 4:
                segments.append((start, end))
        return segments

    def available_episodes(self) -> list[tuple[int, int]]:
        return self._episode_ranges()

    def selected_episodes(self, limit: int | None) -> list[tuple[int, int]]:
        episodes = self.available_episodes()
        if limit is None or limit >= len(episodes):
            return episodes
        indices = np.linspace(0, len(episodes) - 1, num=limit, dtype=np.int64)
        return [episodes[int(idx)] for idx in indices]

    def set_eval_context(self, episode_index: int | None) -> None:
        self.eval_episode_index = episode_index


    def reset(self, episode_span: tuple[int, int]) -> Observation:
        self.episode_span = episode_span
        self.episode_decisions = self.decision_indices[episode_span[0] : episode_span[1]]
        self.eval_context_key = None
        if self.config.deterministic_evaluation and self.eval_episode_index is not None:
            self.eval_context_key = "|".join(
                [
                    str(self.config.eval_seed_base),
                    self.day.symbol,
                    self.day.day,
                    str(self.eval_episode_index),
                    str(episode_span[0]),
                    str(episode_span[1]),
                    str(self.config.latency),
                ]
            )
        self.step_cursor = 0
        if self.config.random_initial_inventory:
            start_units = self._initial_inventory_units()
        else:
            start_units = 0
        self.inventory = float(start_units * self.config.trade_unit)
        self.initial_inventory = float(self.inventory)
        self.cash = 0.0
        self.value = 0.0
        self.value_prev = 0.0
        self.turnover = 0.0
        self.trades = 0
        self.rewards = 0.0
        self.trading_pnl = 0.0
        self.trading_pnl_units = 0.0
        self.fill_steps = 0
        self.quote_spreads: list[float] = []
        self.quote_spreads_bps: list[float] = []
        self.quote_biases_bps: list[float] = []
        self.ask_distance_bps: list[float] = []
        self.bid_distance_bps: list[float] = []
        self.inventory_history: list[float] = []
        self.step_logs: list[dict[str, float]] = []
        return self._build_observation(self.episode_decisions[self.step_cursor] - self.config.latency)
    
    def _build_flat_features(self, data_idx: int) -> np.ndarray:
        time_ratio = self.step_cursor / max(len(self.episode_decisions), 1)
        agent = np.array([self.inventory / max(self.config.max_inventory * self.config.trade_unit, 1), time_ratio], dtype=np.float32)
        if self.state_mode == "inventory_only":
            return agent
        if self.state_mode == "handcrafted":
            return np.concatenate([self.day.handcrafted[data_idx], agent]).astype(np.float32)
        dynamic = np.zeros(0, dtype=np.float32) if self.wo_dynamic_state else self.day.dynamic[data_idx]
        return np.concatenate([dynamic, agent]).astype(np.float32)

    def _build_observation(self, data_idx: int) -> Observation:
        lob = None
        if not self.wo_lob_state and self.state_mode == "full":
            start = data_idx - self.config.lookback + 1
            lob = self.day.normalized_lob[start : data_idx + 1]
        return Observation(lob=lob, flat=self._build_flat_features(data_idx))

    def _level_volume(self, event_idx: int, side: str, price: float) -> float:
        row = self.day.lob[event_idx]
        for level in range(10):
            base = level * 4
            level_price = row[base] if side == "ask" else row[base + 2]
            level_vol = row[base + 1] if side == "ask" else row[base + 3]
            if abs(level_price - price) < self.config.tick_size / 2:
                return float(level_vol)
        return 0.0

    def _continuous_orders(self, action: np.ndarray, quote_idx: int) -> dict[str, float]:
        mid = float(self.day.midprice[quote_idx])
        centered = 2.0 * np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0) - 1.0
        if self.config.quote_scale_mode == "bps":
            directional_bias = float(centered[0]) * self.config.max_bias_bps * 1e-4 * mid
            inventory_bias = np.sign(self.inventory) * float(abs(centered[1])) * self.config.max_inventory_skew_bps * 1e-4 * mid
            spread = max(
                self.config.tick_size,
                float(np.clip(action[2], 0.0, 1.0)) * self.config.max_spread_bps * 1e-4 * mid,
            )
        else:
            directional_bias = float(centered[0]) * self.config.max_bias
            inventory_bias = np.sign(self.inventory) * float(abs(centered[1])) * self.config.max_bias
            spread = max(self.config.tick_size, float(np.clip(action[2], 0.0, 1.0)) * self.config.max_spread)
        reservation = mid + directional_bias - inventory_bias
        ask_price, bid_price = price_legal_check(reservation + spread / 2, reservation - spread / 2, self.config.tick_size)
        return {
            "ask_price": ask_price,
            "ask_volume": -self.config.trade_unit,
            "bid_price": bid_price,
            "bid_volume": self.config.trade_unit,
            "spread": ask_price - bid_price,
            "reservation": reservation,
            "directional_bias": directional_bias,
            "inventory_bias": inventory_bias,
        }

    def _discrete_orders(self, action: int, quote_idx: int) -> dict[str, float]:
        ask = float(self.day.ask1[quote_idx])
        bid = float(self.day.bid1[quote_idx])
        tick = self.config.tick_size
        if action == 0:
            ask_price, bid_price = ask, bid
        elif action == 1:
            ask_price, bid_price = ask, bid - tick
        elif action == 2:
            ask_price, bid_price = ask + tick, bid
        elif action == 3:
            ask_price, bid_price = ask + tick, bid - tick
        elif action == 4:
            ask_price, bid_price = ask, bid - 2 * tick
        elif action == 5:
            ask_price, bid_price = ask + 2 * tick, bid
        elif action == 6:
            ask_price, bid_price = ask + 2 * tick, bid - 2 * tick
        elif action == 7:
            if self.inventory > 0:
                return {"ask_price": float(self.day.bid1[quote_idx]), "ask_volume": -abs(self.inventory), "bid_price": 0.0, "bid_volume": 0.0, "spread": 0.0}
            if self.inventory < 0:
                return {"ask_price": 0.0, "ask_volume": 0.0, "bid_price": float(self.day.ask1[quote_idx]), "bid_volume": abs(self.inventory), "spread": 0.0}
            return {"ask_price": 0.0, "ask_volume": 0.0, "bid_price": 0.0, "bid_volume": 0.0, "spread": 0.0}
        else:
            raise ValueError(f"Unknown discrete action {action}")
        ask_price, bid_price = price_legal_check(ask_price, bid_price, tick)
        return {
            "ask_price": ask_price,
            "ask_volume": -self.config.trade_unit,
            "bid_price": bid_price,
            "bid_volume": self.config.trade_unit,
            "spread": ask_price - bid_price,
        }

    def action_to_orders(self, action: np.ndarray | int | dict[str, float], quote_idx: int) -> dict[str, float]:
        if isinstance(action, dict):
            orders = dict(action)
        elif self.state_mode == "discrete":
            orders = self._discrete_orders(int(action), quote_idx)
        elif np.isscalar(action):
            orders = self._discrete_orders(int(action), quote_idx)
        else:
            orders = self._continuous_orders(np.asarray(action, dtype=np.float32), quote_idx)
        inv_limit = self.config.max_inventory * self.config.trade_unit
        if self.inventory >= inv_limit:
            orders["bid_volume"] = 0.0
        if self.inventory <= -inv_limit:
            orders["ask_volume"] = 0.0
        return orders

    def _fill_draw(self, event_idx: int, side: str, price: float) -> float:
        if self.eval_context_key is None:
            return float(np.random.random())
        tick_index = int(round(price / max(self.config.tick_size, 1e-8)))
        key = f"{self.eval_context_key}|{event_idx}|{side}|{tick_index}"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)
 
    def _initial_inventory_units(self) -> int:
        max_units = max(0, min(self.config.initial_inventory_max, self.config.max_inventory))
        if max_units <= 0:
            return 0
        if self.eval_context_key is None:
            return int(np.random.randint(-max_units, max_units + 1))
        key = f"{self.eval_context_key}|initial_inventory"
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        draw = int.from_bytes(digest, byteorder="big", signed=False)
        return int(draw % (2 * max_units + 1)) - max_units
    
    def _match_one_side(self, event_idx: int, side: str, price: float, volume: float) -> list[Fill]:
        if volume == 0 or price == 0:
            return []
        if self.config.fill_model == "legacy":
            return self._match_legacy(event_idx, side, price, volume)
        return self._match_queue(event_idx, side, price, volume)
    
    # ------------------------------------------------------------------
    # Legacy fill model (original paper logic and the model we used before)
    # ------------------------------------------------------------------
    def _match_legacy(self, event_idx: int, side: str, price: float, volume: float) -> list[Fill]:
        trades = self.day.trades_by_index.get(event_idx)
        if trades is None or trades.price.size == 0:
            return []
        traded_prices = trades.price
        traded_sizes = trades.size
        if trades.aggressor_side is not None:
            desired_side = "B" if side == "ask" else "A"
            side_mask = trades.aggressor_side == desired_side
            if not np.any(side_mask):
                return []
            traded_prices = traded_prices[side_mask]
            traded_sizes = traded_sizes[side_mask]
        fills: list[Fill] = []
        if side == "ask":
            book_cross_price = float(self.day.bid1[event_idx])
            signed_volume = -abs(volume)
            if price <= book_cross_price:
                fills.append(Fill(book_cross_price, signed_volume, taker=True))
                return fills
            better = traded_prices > price
            exact = np.isclose(traded_prices, price)
            if np.any(better):
                fills.append(Fill(price, signed_volume, taker=False))
            elif np.any(exact):
                exact_volume = float(traded_sizes[exact].sum())
                depth = self._level_volume(event_idx, "ask", price)
                probability = exact_volume / max(exact_volume + depth, 1e-8)
                if self._fill_draw(event_idx, side, price) < probability:
                    fills.append(Fill(price, signed_volume, taker=False))
        else:
            book_cross_price = float(self.day.ask1[event_idx])
            signed_volume = abs(volume)
            if price >= book_cross_price:
                fills.append(Fill(book_cross_price, signed_volume, taker=True))
                return fills
            better = traded_prices < price
            exact = np.isclose(traded_prices, price)
            if np.any(better):
                fills.append(Fill(price, signed_volume, taker=False))
            elif np.any(exact):
                exact_volume = float(traded_sizes[exact].sum())
                depth = self._level_volume(event_idx, "bid", price)
                probability = exact_volume / max(exact_volume + depth, 1e-8)
                if self._fill_draw(event_idx, side, price) < probability:
                    fills.append(Fill(price, signed_volume, taker=False))
        return fills
    
    # ------------------------------------------------------------------
    # Queue-position-aware fill model
    # ------------------------------------------------------------------
    # MSG_COLUMNS indices for direct numpy access:
    #   0: market_buy_volume   4: limit_buy_volume   8: withdraw_buy_volume
    #   2: market_sell_volume   6: limit_sell_volume  10: withdraw_sell_volume
    _MSG_WITHDRAW_BUY_VOL = 8
    _MSG_WITHDRAW_SELL_VOL = 10
 
    def _queue_ahead(self, quote_idx: int, event_idx: int, side: str, price: float) -> float:
        """Estimate how many shares are ahead of the agent in the queue.
 
        1. Start with the displayed depth at the agent's price level when
           the order was placed (quote_idx).
        2. Subtract cancellations (withdrawals) that occurred between
           quote_idx and event_idx — these are orders ahead of us that
           left the queue, improving our position.
 
        With ``queue_position="back"`` we assume the agent is behind all
        existing depth.  With ``queue_position="uniform"`` we assume a
        random position in the queue (more optimistic).
        """
        depth_at_placement = self._level_volume(quote_idx, side, price)
 
        if self.config.queue_position == "uniform":
            # Random position: on average half the queue is ahead
            draw = self._fill_draw(event_idx, side, price)
            depth_at_placement = depth_at_placement * draw
 
        # Estimate queue attrition from cancellations between placement
        # and the current event.  msg has per-event withdraw volumes but
        # they are aggregated across all price levels, so we scale by the
        # fraction of total depth at our level.  This is approximate but
        # directionally correct.
        if event_idx > quote_idx and hasattr(self.day, 'msg') and self.day.msg is not None:
            col = self._MSG_WITHDRAW_SELL_VOL if side == "ask" else self._MSG_WITHDRAW_BUY_VOL
            # Sum all withdrawals between quote placement and fill check
            start = quote_idx + 1
            end = event_idx + 1  # inclusive of event_idx
            if start < end and end <= len(self.day.msg):
                total_withdrawals = float(self.day.msg[start:end, col].sum())
                # Fraction of total side depth at our price level
                total_depth = self._total_side_depth(quote_idx, side)
                if total_depth > 0:
                    level_fraction = depth_at_placement / total_depth
                    attrition = total_withdrawals * level_fraction
                    depth_at_placement = max(0.0, depth_at_placement - attrition)
 
        return depth_at_placement
 
    def _total_side_depth(self, event_idx: int, side: str) -> float:
        """Sum of displayed volume across all 10 levels on one side."""
        row = self.day.lob[event_idx]
        total = 0.0
        for level in range(10):
            base = level * 4
            vol = row[base + 1] if side == "ask" else row[base + 3]
            total += float(vol)
        return total
 
    def _match_queue(self, event_idx: int, side: str, price: float, volume: float) -> list[Fill]:
        agent_size = abs(volume)
        quote_idx = max(int(event_idx - self.config.latency), self.config.lookback - 1)
 
        #agent crosses the spread 
        if side == "ask":
            book_cross_price = float(self.day.bid1[event_idx])
            if price <= book_cross_price:
                return [Fill(book_cross_price, -agent_size, taker=True)]
        else:
            book_cross_price = float(self.day.ask1[event_idx])
            if price >= book_cross_price:
                return [Fill(book_cross_price, agent_size, taker=True)]
 
        #check if market trades reach our queue position
        trades = self.day.trades_by_index.get(event_idx)
        if trades is None or trades.price.size == 0:
            return []
 
        traded_prices = trades.price
        traded_sizes = trades.size
        if trades.aggressor_side is not None:
            desired_side = "B" if side == "ask" else "A"
            side_mask = trades.aggressor_side == desired_side
            if not np.any(side_mask):
                return []
            traded_prices = traded_prices[side_mask]
            traded_sizes = traded_sizes[side_mask]
 
        # Trades that swept through our price level (traded at a worse
        # price for the aggressor than our quote) guarantee a fill.
        if side == "ask":
            through = traded_prices > price + self.config.tick_size / 2
        else:
            through = traded_prices < price - self.config.tick_size / 2
 
        if np.any(through):
            signed = -agent_size if side == "ask" else agent_size
            return [Fill(price, signed, taker=False)]
 
        # Trades at our exact price level: check queue position
        exact = np.isclose(traded_prices, price, atol=self.config.tick_size / 2)
        if not np.any(exact):
            return []
 
        exact_volume = float(traded_sizes[exact].sum())
        queue_ahead = self._queue_ahead(quote_idx, event_idx, side, price)
 
        # Volume that reaches past the orders ahead of us
        volume_reaching_us = exact_volume - queue_ahead
        if volume_reaching_us <= 0:
            return []
 
        filled = min(agent_size, volume_reaching_us)
        signed = -filled if side == "ask" else filled
        return [Fill(price, signed, taker=False)]
    
    def _reward_unit(self, midprice: float, spread: float) -> float:
        if self.config.reward_scale_mode == "ticks":
            return max(self.config.tick_size, 1e-8)
        return max(spread, self.config.tick_size, 1e-8)
 
    def _inventory_penalty(self) -> float:
        inv_limit = max(self.config.max_inventory * self.config.trade_unit, 1)
        inv_norm = float(self.inventory / inv_limit)
        progress = float(self.step_cursor / max(len(self.episode_decisions) - 1, 1))
        if self.reward_mode == "trade_inventory":
            return 0.0
        if self.reward_mode == "trade_inventory_ramp":
            start = self.config.zeta_start if self.config.zeta_start is not None else self.config.zeta
            end = self.config.zeta_end if self.config.zeta_end is not None else start
            return float((start + (end - start) * progress**2) * inv_norm**2)
        if self.reward_mode == "trade_inventory_l1l2":
            l2_coeff = self.config.zeta_l2 if self.config.zeta_l2 is not None else self.config.zeta
            return float(l2_coeff * inv_norm**2 + self.config.zeta_l1 * abs(inv_norm))
        return float(self.config.zeta * inv_norm**2)
 
    def _terminal_inventory_penalty(self, midprice: float, spread: float) -> float:
        net_inventory = self.inventory - self.initial_inventory
        if net_inventory == 0:
            return 0.0
        reward_unit = self._reward_unit(midprice, spread)
        liquidation_cost = abs(net_inventory) * (0.5 * max(spread, self.config.tick_size) + self.config.taker_fee_per_share)
        return float(self.config.terminal_inventory_cost_scale * liquidation_cost / reward_unit)
 

    def step(self, action: np.ndarray | int | dict[str, float]) -> tuple[Observation, float, bool, dict[str, float]]:
        event_idx = int(self.episode_decisions[self.step_cursor])
        quote_idx = max(int(event_idx - self.config.latency), self.config.lookback - 1)
        orders = self.action_to_orders(action, quote_idx)
        fills = []
        fills.extend(self._match_one_side(event_idx, "ask", float(orders["ask_price"]), float(abs(orders["ask_volume"]))))
        fills.extend(self._match_one_side(event_idx, "bid", float(orders["bid_price"]), float(abs(orders["bid_volume"]))))
        self.quote_spreads.append(float(orders.get("spread", 0.0)))
        midprice = float(self.day.midprice[event_idx])
        ask_distance_bps = 1e4 * (float(orders["ask_price"]) - midprice) / max(midprice, 1e-8) if orders["ask_price"] > 0 else 0.0
        bid_distance_bps = 1e4 * (midprice - float(orders["bid_price"])) / max(midprice, 1e-8) if orders["bid_price"] > 0 else 0.0
        spread_bps = 1e4 * float(orders.get("spread", 0.0)) / max(midprice, 1e-8)
        if orders["ask_price"] > 0 and orders["bid_price"] > 0:
            reservation = 0.5 * (float(orders["ask_price"]) + float(orders["bid_price"]))
            bias_bps = 1e4 * (reservation - midprice) / max(midprice, 1e-8)
        else:
            bias_bps = 0.0
        self.quote_spreads_bps.append(float(spread_bps))
        self.quote_biases_bps.append(float(bias_bps))
        self.ask_distance_bps.append(float(ask_distance_bps))
        self.bid_distance_bps.append(float(bid_distance_bps))
 
        reward_unit = self._reward_unit(midprice, float(orders.get("spread", self.day.spread[event_idx])))
        trade_edge_step = 0.0
        fee_step = 0.0
        for fill in fills:
            self.inventory += fill.volume
            self.cash -= fill.volume * fill.price
            self.turnover += abs(fill.volume * fill.price)
            self.trades += 1
            edge = float(fill.volume * (midprice - fill.price))
            fee = (-self.config.taker_fee_per_share if fill.taker else self.config.maker_rebate_per_share) * abs(fill.volume)
            trade_edge_step += edge + fee
            fee_step += fee
        if fills:
            self.fill_steps += 1
 
        self.value = self.cash + self.inventory * midprice
        self.trading_pnl += trade_edge_step
        trade_units = trade_edge_step / reward_unit
        self.trading_pnl_units += trade_units
        reward = float(trade_units - self._inventory_penalty())
        self.inventory_history.append(float(self.inventory))
 
        self.step_cursor += 1
        done = self.step_cursor >= len(self.episode_decisions)
        terminal_penalty = 0.0
        if done and not self.config.allow_terminal_inventory and self.inventory != 0:
            flatten_price = float(self.day.bid1[event_idx] if self.inventory > 0 else self.day.ask1[event_idx])
            flatten_volume = -self.inventory
            taker_fill = Fill(flatten_price, flatten_volume, taker=True)
            self.inventory += flatten_volume
            self.cash -= flatten_volume * flatten_price
            self.turnover += abs(flatten_volume * flatten_price)
            self.trades += 1
            edge = float(taker_fill.volume * (midprice - taker_fill.price))
            fee = -self.config.taker_fee_per_share * abs(taker_fill.volume)
            trade_edge_step = edge + fee
            self.trading_pnl += trade_edge_step
            trade_units = trade_edge_step / reward_unit
            self.trading_pnl_units += trade_units
            flatten_reward = float(trade_units)
            reward += flatten_reward
            self.value = self.cash
        elif done and self.config.allow_terminal_inventory and self.inventory != 0:
            terminal_penalty = self._terminal_inventory_penalty(midprice, float(orders.get("spread", self.day.spread[event_idx])))
            reward -= terminal_penalty
 
        self.rewards += reward
        self.step_logs.append(
            {
                "timestamp": self.day.timestamps[event_idx],
                "midprice": midprice,
                "inventory": float(self.inventory),
                "initial_inventory": float(self.initial_inventory),
                "ask_quote": float(orders["ask_price"]),
                "bid_quote": float(orders["bid_price"]),
                "spread_bps": float(spread_bps),
                "bias_bps": float(bias_bps),
                "ask_distance_bps": float(ask_distance_bps),
                "bid_distance_bps": float(bid_distance_bps),
                "fills": float(len(fills)),
                "reward": reward,
                "trade_edge_dollars": float(trade_edge_step),
                "trade_edge_units": float(trade_units),
                "terminal_inventory_penalty": float(terminal_penalty),
                "fees": float(fee_step),
                "cash": float(self.cash),
                "value": float(self.value),
                "turnover": float(self.turnover),
            }
        )
        next_obs = self._build_observation(max(quote_idx, self.config.lookback - 1)) if done else self._build_observation(max(int(self.episode_decisions[self.step_cursor] - self.config.latency), self.config.lookback - 1))
        return next_obs, reward, done, {"fills": len(fills), "inventory": float(self.inventory)}
    
    def episode_result(self, method: str, episode_index: int, latency: int | None = None) -> EpisodeResult:
        avg_spread = float(np.mean([spread for spread in self.quote_spreads if spread > 0])) if self.quote_spreads else 0.0
        avg_position = float(np.mean(self.inventory_history)) if self.inventory_history else 0.0
        avg_abs_position = float(np.mean(np.abs(self.inventory_history))) if self.inventory_history else 0.0
        pnl = float(self.value)
        nd_pnl = pnl / avg_spread if avg_spread > 0 else 0.0
        pnl_map = pnl / avg_abs_position if avg_abs_position > 0 else 0.0
        profit_ratio = pnl / self.turnover if self.turnover > 0 else 0.0
        return EpisodeResult(
            symbol=self.day.symbol,
            day=self.day.day,
            method=method,
            episode_index=episode_index,
            pnl=pnl,
            nd_pnl=nd_pnl,
            pnl_map=pnl_map,
            profit_ratio=profit_ratio,
            avg_position=avg_position,
            avg_abs_position=avg_abs_position,
            avg_spread=avg_spread,
            turnover=float(self.turnover),
            reward=float(self.rewards),
            trades=int(self.trades),
            latency=latency if latency is not None else self.config.latency,
            fill_rate=float(self.fill_steps / max(len(self.episode_decisions), 1)),
            avg_bias_bps=float(np.mean(self.quote_biases_bps)) if self.quote_biases_bps else 0.0,
            avg_ask_distance_bps=float(np.mean(self.ask_distance_bps)) if self.ask_distance_bps else 0.0,
            avg_bid_distance_bps=float(np.mean(self.bid_distance_bps)) if self.bid_distance_bps else 0.0,
            avg_spread_bps=float(np.mean(self.quote_spreads_bps)) if self.quote_spreads_bps else 0.0,
        )

    def episode_trace(self) -> pd.DataFrame:
        if not self.step_logs:
            return pd.DataFrame()
        return pd.DataFrame(self.step_logs)