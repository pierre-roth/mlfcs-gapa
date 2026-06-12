"""Paper baseline strategies and deterministic evaluation loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import log
from typing import Protocol

import numpy as np

from mlfcs_gapa.data.schema import LobDataset
from mlfcs_gapa.env.actions import Quote, legalize_quote_prices
from mlfcs_gapa.env.replay import Account, HistoricalReplay, compute_episode_metrics
from mlfcs_gapa.paper.constants import PAPER


class QuoteStrategy(Protocol):
    name: str

    def quote(
        self,
        replay: HistoricalReplay,
        account: Account,
        decision_index: int,
        episode_progress: float,
    ) -> Quote: ...


@dataclass(frozen=True)
class FixedLevelStrategy:
    level: int

    @property
    def name(self) -> str:
        return f"Fixed_{self.level}"

    def quote(
        self,
        replay: HistoricalReplay,
        account: Account,
        decision_index: int,
        episode_progress: float,
    ) -> Quote:
        del account, episode_progress
        if not 1 <= self.level <= PAPER.lob_levels:
            raise ValueError(f"level must be in [1, {PAPER.lob_levels}]")
        ask = float(replay.ask_prices[decision_index, self.level - 1])
        bid = float(replay.bid_prices[decision_index, self.level - 1])
        return Quote(
            ask_price=ask,
            ask_volume=-PAPER.minimum_trade_unit,
            bid_price=bid,
            bid_volume=PAPER.minimum_trade_unit,
            reservation_price=(ask + bid) / 2.0,
            spread=ask - bid,
        )


@dataclass
class RandomLevelStrategy:
    max_level: int = 5
    seed: int = 1

    @property
    def name(self) -> str:
        return "Random"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def quote(
        self,
        replay: HistoricalReplay,
        account: Account,
        decision_index: int,
        episode_progress: float,
    ) -> Quote:
        del account, episode_progress
        ask_level = int(self.rng.integers(1, self.max_level + 1))
        bid_level = int(self.rng.integers(1, self.max_level + 1))
        ask = float(replay.ask_prices[decision_index, ask_level - 1])
        bid = float(replay.bid_prices[decision_index, bid_level - 1])
        return Quote(
            ask_price=ask,
            ask_volume=-PAPER.minimum_trade_unit,
            bid_price=bid,
            bid_volume=PAPER.minimum_trade_unit,
            reservation_price=(ask + bid) / 2.0,
            spread=ask - bid,
        )


@dataclass(frozen=True)
class AvellanedaStoikovStrategy:
    sigma: float
    gamma: float = 0.1
    kappa: float = 100.0
    tick_size: float = 0.01

    @property
    def name(self) -> str:
        return "AS"

    def quote(
        self,
        replay: HistoricalReplay,
        account: Account,
        decision_index: int,
        episode_progress: float,
    ) -> Quote:
        mid = replay.mid_price(decision_index)
        q_lots = account.inventory / PAPER.minimum_trade_unit
        tau = max(0.0, 1.0 - episode_progress)
        reservation = mid - q_lots * self.gamma * self.sigma * self.sigma * tau
        spread = self.gamma * self.sigma * self.sigma * tau + (2.0 / self.gamma) * log(
            1.0 + self.gamma / self.kappa
        )
        ask, bid = legalize_quote_prices(
            reservation + spread / 2.0,
            reservation - spread / 2.0,
            tick_size=self.tick_size,
        )
        return Quote(
            ask_price=ask,
            ask_volume=-PAPER.minimum_trade_unit,
            bid_price=bid,
            bid_volume=PAPER.minimum_trade_unit,
            reservation_price=reservation,
            spread=spread,
        )


def estimate_event_volatility(dataset: LobDataset) -> float:
    ask1 = dataset.orderbook["ask1_price"].to_numpy().astype(np.float64)
    bid1 = dataset.orderbook["bid1_price"].to_numpy().astype(np.float64)
    mid = (ask1 + bid1) / 2.0
    returns = np.diff(np.log(mid))
    if len(returns) == 0:
        return 0.0
    return float(np.std(returns))


def evaluate_quote_strategy(
    dataset: LobDataset,
    strategy: QuoteStrategy,
    *,
    episode_start: int = 0,
    episode_events: int = PAPER.episode_events,
    latency_events: int = 1,
    seed: int = 1,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    replay = HistoricalReplay(dataset, rng=np.random.default_rng(seed))
    account = Account()
    first_index = episode_start + PAPER.window_length + latency_events - 1
    episode_end = min(episode_start + episode_events, dataset.orderbook.height - 1)
    if first_index >= episode_end:
        raise ValueError("episode is too short for evaluation")

    values = [account.value]
    inventories = [account.inventory]
    quoted_spreads: list[float] = []
    log_rows: list[dict[str, float | int | str]] = []

    for current_index in range(first_index, episode_end):
        progress = (current_index - episode_start) / max(1, episode_end - episode_start)
        decision_index = max(PAPER.window_length - 1, current_index - latency_events)
        quote = strategy.quote(replay, account, decision_index, progress)
        quote = _apply_inventory_cap(quote, account)
        mid = replay.mid_price(current_index)
        fill = replay.match(current_index, quote)
        account.apply_fill(fill, mid)
        if quote.ask_price > 0 and quote.bid_price > 0:
            quoted_spreads.append(quote.ask_price - quote.bid_price)
        values.append(account.value)
        inventories.append(account.inventory)
        log_rows.append(
            {
                "method": strategy.name,
                "index": current_index,
                "mid_price": mid,
                "ask_price": quote.ask_price,
                "bid_price": quote.bid_price,
                "trade_price": fill.trade_price,
                "trade_volume": fill.trade_volume,
                "inventory": account.inventory,
                "cash": account.cash,
                "value": account.value,
            }
        )

    mid = replay.mid_price(episode_end - 1)
    close_fill = replay.close_position(episode_end - 1, account)
    account.apply_fill(close_fill, mid)
    values.append(account.value)
    inventories.append(account.inventory)
    log_rows.append(
        {
            "method": strategy.name,
            "index": episode_end - 1,
            "mid_price": mid,
            "ask_price": 0.0,
            "bid_price": 0.0,
            "trade_price": close_fill.trade_price,
            "trade_volume": close_fill.trade_volume,
            "inventory": account.inventory,
            "cash": account.cash,
            "value": account.value,
        }
    )

    metrics = compute_episode_metrics(values, inventories, quoted_spreads, account.buy_notional)
    result = asdict(metrics)
    result["method"] = strategy.name
    return result, log_rows


def _apply_inventory_cap(quote: Quote, account: Account) -> Quote:
    if account.inventory <= -PAPER.max_inventory:
        return replace(quote, ask_price=0.0, ask_volume=0)
    if account.inventory >= PAPER.max_inventory:
        return replace(quote, bid_price=0.0, bid_volume=0)
    return quote
