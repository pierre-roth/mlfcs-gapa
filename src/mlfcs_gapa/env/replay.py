"""Historical event-replay market-making simulator primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs_gapa.data.schema import LobDataset
from mlfcs_gapa.env.actions import Quote
from mlfcs_gapa.paper.constants import PAPER


@dataclass(frozen=True)
class Fill:
    trade_price: float
    trade_volume: int

    @property
    def occurred(self) -> bool:
        return self.trade_volume != 0


@dataclass
class Account:
    cash: float = 0.0
    inventory: int = 0
    value: float = 0.0
    buy_notional: float = 0.0

    def mark_to_mid(self, mid_price: float) -> float:
        self.value = self.cash + self.inventory * mid_price
        return self.value

    def apply_fill(self, fill: Fill, mid_price: float) -> None:
        if fill.trade_volume == 0:
            self.mark_to_mid(mid_price)
            return
        self.inventory += fill.trade_volume
        self.cash -= fill.trade_volume * fill.trade_price
        if fill.trade_volume > 0:
            self.buy_notional += fill.trade_volume * fill.trade_price
        self.mark_to_mid(mid_price)


@dataclass(frozen=True)
class EpisodeMetrics:
    pnl: float
    nd_pnl: float
    pnl_map: float
    profit_ratio: float
    mean_inventory: float
    mean_abs_inventory: float
    mean_quoted_spread: float


class HistoricalReplay:
    """Paper-style event replay over one canonical LOB dataset."""

    def __init__(
        self,
        dataset: LobDataset,
        *,
        tick_size: float = 0.01,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.dataset = dataset
        self.tick_size = tick_size
        self.rng = rng or np.random.default_rng(1)
        self.orderbook = dataset.orderbook
        self.trades = dataset.trades

    def mid_price(self, index: int) -> float:
        row = self.orderbook.row(index, named=True)
        return (float(row["ask1_price"]) + float(row["bid1_price"])) / 2.0

    def best_bid_ask(self, index: int) -> tuple[float, float]:
        row = self.orderbook.row(index, named=True)
        return float(row["bid1_price"]), float(row["ask1_price"])

    def match(self, index: int, quote: Quote) -> Fill:
        """Match one bid/ask quote pair against historical event `index`.

        The rules follow the paper and the queue-position interpretation:

        - crossing ask executes as market sell at best bid.
        - crossing bid executes as market buy at best ask.
        - passive ask fills if historical max trade price exceeds the ask.
        - passive bid fills if historical min trade price is below the bid.
        - equal-price passive fills are probabilistic because the agent is at
          the back of the queue.
        """

        bid1, ask1 = self.best_bid_ask(max(0, index - 1))
        trade = self.trades.row(index, named=True)

        sell_fill = self._match_sell(index, quote, bid1, trade)
        buy_fill = self._match_buy(index, quote, ask1, trade)

        if sell_fill.occurred and buy_fill.occurred:
            # One historical event may touch both sides in synthetic data. The
            # paper simulator stores one trade result per event, so choose the
            # side with larger notional and keep behavior deterministic.
            if abs(sell_fill.trade_price * sell_fill.trade_volume) >= abs(
                buy_fill.trade_price * buy_fill.trade_volume
            ):
                return sell_fill
            return buy_fill
        if sell_fill.occurred:
            return sell_fill
        return buy_fill

    def close_position(self, index: int, account: Account) -> Fill:
        bid1, ask1 = self.best_bid_ask(max(0, index - 1))
        if account.inventory < 0:
            return Fill(trade_price=ask1, trade_volume=-account.inventory)
        if account.inventory > 0:
            return Fill(trade_price=bid1, trade_volume=-account.inventory)
        return Fill(trade_price=0.0, trade_volume=0)

    def _match_sell(
        self, index: int, quote: Quote, best_bid: float, trade: dict[str, object]
    ) -> Fill:
        if quote.ask_price <= 0 or quote.ask_volume >= 0:
            return Fill(0.0, 0)
        if quote.ask_price <= best_bid:
            return Fill(best_bid, quote.ask_volume)

        trade_max = float(trade["trade_price_max"])
        trade_max_volume = int(trade["trade_price_max_volume"])
        if trade_max_volume <= 0:
            return Fill(0.0, 0)
        if trade_max > quote.ask_price:
            return Fill(quote.ask_price, quote.ask_volume)
        if np.isclose(trade_max, quote.ask_price):
            depth = self._displayed_depth(index, side="ask", price=quote.ask_price)
            if self._queue_fill(trade_max_volume, depth):
                return Fill(quote.ask_price, quote.ask_volume)
        return Fill(0.0, 0)

    def _match_buy(
        self, index: int, quote: Quote, best_ask: float, trade: dict[str, object]
    ) -> Fill:
        if quote.bid_price <= 0 or quote.bid_volume <= 0:
            return Fill(0.0, 0)
        if quote.bid_price >= best_ask:
            return Fill(best_ask, quote.bid_volume)

        trade_min = float(trade["trade_price_min"])
        trade_min_volume = int(trade["trade_price_min_volume"])
        if trade_min_volume <= 0:
            return Fill(0.0, 0)
        if trade_min < quote.bid_price:
            return Fill(quote.bid_price, quote.bid_volume)
        if np.isclose(trade_min, quote.bid_price):
            depth = self._displayed_depth(index, side="bid", price=quote.bid_price)
            if self._queue_fill(trade_min_volume, depth):
                return Fill(quote.bid_price, quote.bid_volume)
        return Fill(0.0, 0)

    def _displayed_depth(self, index: int, *, side: str, price: float) -> int:
        row = self.orderbook.row(index, named=True)
        for level in range(1, PAPER.lob_levels + 1):
            if np.isclose(float(row[f"{side}{level}_price"]), price):
                return int(row[f"{side}{level}_volume"])
        return int(row[f"{side}1_volume"])

    def _queue_fill(self, traded_volume: int, displayed_depth: int) -> bool:
        probability = traded_volume / (traded_volume + displayed_depth + 1e-7)
        return bool(self.rng.random() < probability)


def compute_episode_metrics(
    values: list[float],
    inventories: list[int],
    quoted_spreads: list[float],
    buy_notional: float,
    *,
    initial_value: float = 0.0,
) -> EpisodeMetrics:
    final_value = values[-1] if values else initial_value
    pnl = final_value - initial_value
    mean_spread = float(np.mean(quoted_spreads)) if quoted_spreads else 0.0
    mean_inventory = float(np.mean(inventories)) if inventories else 0.0
    mean_abs_inventory = float(np.mean(np.abs(inventories))) if inventories else 0.0
    return EpisodeMetrics(
        pnl=pnl,
        nd_pnl=pnl / (mean_spread + 1e-7),
        pnl_map=pnl / (mean_abs_inventory + 1e-7),
        profit_ratio=pnl / (buy_notional + 1e-7),
        mean_inventory=mean_inventory,
        mean_abs_inventory=mean_abs_inventory,
        mean_quoted_spread=mean_spread,
    )
