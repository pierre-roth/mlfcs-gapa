from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DiagnosticsConfig, SimulatorConfig, SymbolSpec
from .orderbook import FIFOOrderBook, TradeFill


@dataclass
class SyntheticDay:
    symbol: str
    day: str
    ask: pd.DataFrame
    bid: pd.DataFrame
    price: pd.DataFrame
    trades: pd.DataFrame
    msg: pd.DataFrame
    event_log: pd.DataFrame
    latent: pd.DataFrame
    depth_cube: np.ndarray

    def export(self, root: Path) -> None:
        day_root = root / self.symbol / self.day
        day_root.mkdir(parents=True, exist_ok=True)
        self.ask.to_csv(day_root / "ask.csv", index=False)
        self.bid.to_csv(day_root / "bid.csv", index=False)
        self.price.to_csv(day_root / "price.csv", index=False)
        self.trades.to_csv(day_root / "trades.csv", index=False)
        self.msg.to_csv(day_root / "msg.csv", index=False)
        self.event_log.to_csv(day_root / "event_log.csv", index=False)
        self.latent.to_csv(day_root / "latent.csv", index=False)


class SyntheticMarketGenerator:
    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.config.apply_mode_defaults()
        self.spec = config.symbol_spec
        if config.events_per_day_override is not None:
            self.spec = replace(self.spec, events_per_day=config.events_per_day_override)

    def business_days(self) -> list[str]:
        return [
            (pd.Timestamp("2019-11-01") + pd.tseries.offsets.BDay(i)).strftime("%Y%m%d")
            for i in range(self.config.num_days)
        ]

    def train_days(self) -> list[str]:
        return self.business_days()[: self.config.train_days]

    def test_days(self) -> list[str]:
        start = self.config.train_days
        stop = start + self.config.test_days
        return self.business_days()[start:stop]

    def generate_day(self, day: str) -> SyntheticDay:
        seed = _stable_seed(self.config.seed, self.config.symbol, day)
        rng = np.random.default_rng(seed)
        timestamps = _generate_timestamps(day, self.spec, self.config, rng)

        book = FIFOOrderBook(self.spec.tick_size, self.config.levels)
        mm_inventory = {f"mm_{idx}": 0 for idx in range(self.config.competing_mm_count)}
        fair_tick = self._initial_fair_tick(rng)
        regime = 0.0
        metaorder_direction = 0
        metaorder_strength = 0.0
        volatility_multiplier = 1.0
        last_market_direction = 0
        prev_mid = fair_tick * self.spec.tick_size

        ask_rows: list[dict[str, float | int | pd.Timestamp]] = []
        bid_rows: list[dict[str, float | int | pd.Timestamp]] = []
        price_rows: list[dict[str, float | int | pd.Timestamp]] = []
        trade_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
        msg_rows: list[dict[str, float | int | pd.Timestamp]] = []
        event_log_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
        latent_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
        depth_frames: list[np.ndarray] = []

        self._seed_book(book, fair_tick, timestamps[0], rng)
        for mm_id in mm_inventory:
            self._refresh_market_maker(book, mm_id, mm_inventory[mm_id], fair_tick, timestamps[0], rng)

        for event_idx, ts in enumerate(timestamps):
            top_imbalance = _top_imbalance(book)
            spread_ticks = _spread_ticks(book)
            fair_tick, regime, metaorder_direction, metaorder_strength, regime_shift, volatility_multiplier = self._evolve_fair_value(
                fair_tick=fair_tick,
                regime=regime,
                metaorder_direction=metaorder_direction,
                metaorder_strength=metaorder_strength,
                volatility_multiplier=volatility_multiplier,
                book=book,
                rng=rng,
            )
            event_kind = self._sample_event_kind(book, fair_tick, metaorder_direction, top_imbalance, spread_ticks, rng)
            event_records, fills = self._apply_event(
                book=book,
                event_kind=event_kind,
                fair_tick=fair_tick,
                timestamp=ts,
                mm_inventory=mm_inventory,
                last_market_direction=last_market_direction,
                rng=rng,
            )
            for record in event_records:
                if record.get("event_type") == "market_order":
                    side = str(record.get("side", ""))
                    if side == "buy":
                        last_market_direction = 1
                    elif side == "sell":
                        last_market_direction = -1
                    break
            event_records.extend(self._touch_replenish(book, fair_tick, ts, rng))
            if self._needs_replenishment(book):
                self._seed_book(book, fair_tick, ts, rng, replenish_only=True)
            event_log_rows.extend(event_records)
            msg_rows.append(_paper_msg_row(ts, event_records))
            for fill in fills:
                signed_size = fill.size if fill.aggressor_side == "A" else -fill.size
                trade_rows.append(
                    {
                        "timestamp": fill.timestamp,
                        "price": fill.price,
                        "size": fill.size,
                        "signed_size": signed_size,
                        "aggressor_side": fill.aggressor_side,
                        "taker_agent": fill.taker_agent,
                        "maker_agent_id": fill.maker_agent_id,
                        "maker_agent": fill.maker_agent,
                        "maker_order_id": fill.maker_order_id,
                        "queue_ahead": fill.queue_ahead,
                    }
                )
                if fill.maker_agent_id in mm_inventory:
                    mm_inventory[fill.maker_agent_id] += -fill.size if fill.aggressor_side == "B" else fill.size

            top_levels = book.top_levels()
            ask_row, bid_row = _top_level_rows(ts, top_levels)
            ask_rows.append(ask_row)
            bid_rows.append(bid_row)
            best_bid = top_levels["bid"][0][0]
            best_ask = top_levels["ask"][0][0]
            mid = 0.5 * (best_bid + best_ask)
            price_rows.append(
                {
                    "timestamp": ts,
                    "midprice": mid,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "ask1_price": best_ask,
                    "bid1_price": best_bid,
                    "spread": best_ask - best_bid,
                    "spread_ticks": int(round((best_ask - best_bid) / self.spec.tick_size)),
                    "microprice": _microprice(top_levels),
                    "return_bp": 10_000.0 * (mid / prev_mid - 1.0) if prev_mid > 0 else 0.0,
                }
            )
            prev_mid = mid
            latent_rows.append(
                {
                    "timestamp": ts,
                    "fair_value": fair_tick * self.spec.tick_size,
                    "fair_value_tick": fair_tick,
                    "regime_drift": regime,
                    "metaorder_direction": metaorder_direction,
                    "metaorder_strength": metaorder_strength,
                    "volatility_multiplier": volatility_multiplier,
                    "regime_shift": regime_shift,
                    "event_kind": event_kind,
                    "top_imbalance": top_imbalance,
                    "queue_pressure": _queue_pressure(book, fair_tick),
                }
            )
            center_tick = int(round(mid / self.spec.tick_size))
            depth_frames.append(_depth_frame(book, center_tick, self.config.export_depth_radius_ticks))

        trades = pd.DataFrame(trade_rows)
        if trades.empty:
            trades = pd.DataFrame(
                columns=[
                    "timestamp",
                    "price",
                    "size",
                    "signed_size",
                    "aggressor_side",
                    "taker_agent",
                    "maker_agent_id",
                    "maker_agent",
                    "maker_order_id",
                    "queue_ahead",
                ]
            )
        msg = pd.DataFrame(msg_rows)
        if msg.empty:
            msg = pd.DataFrame(columns=_paper_msg_columns())
        event_log = pd.DataFrame(event_log_rows)
        if event_log.empty:
            event_log = pd.DataFrame(columns=["timestamp", "event_type", "agent_type", "agent_id", "side", "price", "size", "fair_value", "maker_order_id"])
        latent = pd.DataFrame(latent_rows)
        return SyntheticDay(
            symbol=self.config.symbol,
            day=day,
            ask=pd.DataFrame(ask_rows),
            bid=pd.DataFrame(bid_rows),
            price=pd.DataFrame(price_rows),
            trades=trades,
            msg=msg,
            event_log=event_log,
            latent=latent,
            depth_cube=np.asarray(depth_frames, dtype=np.float32),
        )

    def _initial_fair_tick(self, rng: np.random.Generator) -> float:
        base_tick = self.spec.base_price / self.spec.tick_size
        return float(round(base_tick + rng.normal(0.0, 2.0)))

    def _evolve_fair_value(
        self,
        fair_tick: float,
        regime: float,
        metaorder_direction: int,
        metaorder_strength: float,
        volatility_multiplier: float,
        book: FIFOOrderBook,
        rng: np.random.Generator,
    ) -> tuple[float, float, int, float, int, float]:
        regime_shift = 0
        if rng.random() < self.config.regime_switch_prob:
            regime = rng.choice([-1.0, 0.0, 1.0]) * self.config.regime_drift_ticks * self.spec.volatility_scale
            regime_shift = 1
        else:
            regime *= self.config.regime_persistence

        if metaorder_direction == 0 and rng.random() < self.config.metaorder_start_prob:
            book_mid = book.midprice()
            sign = 1 if fair_tick * self.spec.tick_size >= book_mid else -1
            metaorder_direction = sign if rng.random() < 0.7 else -sign
            metaorder_strength = self.config.metaorder_drift_ticks * self.spec.volatility_scale
        elif metaorder_direction != 0 and rng.random() < self.config.metaorder_persistence:
            metaorder_strength = min(metaorder_strength * 1.002, self.config.metaorder_drift_ticks * 1.8)
        else:
            metaorder_direction = 0
            metaorder_strength = 0.0

        shock = 0.0
        if rng.random() < self.config.shock_prob:
            shock = rng.choice([-1.0, 1.0]) * self.config.shock_size_ticks * self.spec.volatility_scale
        if self.config.volatility_cluster_strength > 0.0:
            volatility_multiplier = 1.0 + self.config.volatility_cluster_persistence * (volatility_multiplier - 1.0)
            volatility_multiplier += self.config.volatility_cluster_strength * min(abs(shock) / max(self.config.shock_size_ticks, 1e-8), 1.0)
            volatility_multiplier = float(np.clip(volatility_multiplier, 0.5, 4.0))
        else:
            volatility_multiplier = 1.0
        noise = rng.normal(0.0, self.config.fair_value_vol_ticks * self.spec.volatility_scale * volatility_multiplier)
        base_tick = self.spec.base_price / self.spec.tick_size
        mean_reversion_target = book.midprice() / self.spec.tick_size if book.midprice() > 0 else fair_tick
        fair_tick = fair_tick + noise + regime + metaorder_direction * metaorder_strength + shock
        fair_tick += self.config.fair_value_reversion * (mean_reversion_target - fair_tick)
        fair_tick += self.config.anchor_reversion * (base_tick - fair_tick)
        fair_tick = float(np.clip(fair_tick, base_tick - self.config.daily_price_band_ticks, base_tick + self.config.daily_price_band_ticks))
        return fair_tick, regime, metaorder_direction, metaorder_strength, regime_shift, volatility_multiplier

    def _sample_event_kind(
        self,
        book: FIFOOrderBook,
        fair_tick: float,
        metaorder_direction: int,
        top_imbalance: float,
        spread_ticks: int,
        rng: np.random.Generator,
    ) -> str:
        mid_tick = book.midprice() / self.spec.tick_size if book.midprice() > 0 else fair_tick
        dislocation = abs(fair_tick - mid_tick)
        activity = min(
            2.0,
            0.35 * dislocation
            + 0.55 * abs(metaorder_direction)
            + 0.45 * abs(top_imbalance)
            + 0.30 * max(spread_ticks - self.spec.default_spread_ticks, 0),
        )
        noise_scale = max(self.config.noise_taker_count, 1) / 64.0
        informed_scale = max(self.config.informed_taker_count, 1) / 10.0
        liquidity_scale = max(self.config.liquidity_provider_count, 1) / 10.0
        mm_scale = max(self.config.competing_mm_count, 1) / 6.0
        weights = np.asarray(
            [
                self.config.noise_market_order_prob * noise_scale * (1.0 + 0.55 * activity),
                self.config.informed_market_order_prob * informed_scale * (1.0 + 0.95 * activity),
                self.config.liquidity_add_prob * liquidity_scale * (1.0 + 0.35 * activity + 0.25 * max(spread_ticks - 1, 0)),
                self.config.cancel_prob * (1.0 + 0.70 * activity + 0.30 * abs(top_imbalance)),
                self.config.mm_refresh_prob * mm_scale * (1.0 + 0.90 * activity + 0.50 * max(spread_ticks - 1, 0)),
            ],
            dtype=np.float64,
        )
        weights /= weights.sum()
        return str(rng.choice(["noise_market", "informed_market", "liquidity_add", "cancel", "mm_refresh"], p=weights))

    def _apply_event(
        self,
        book: FIFOOrderBook,
        event_kind: str,
        fair_tick: float,
        timestamp: pd.Timestamp,
        mm_inventory: dict[str, int],
        last_market_direction: int,
        rng: np.random.Generator,
    ) -> tuple[list[dict[str, float | int | str | pd.Timestamp]], list[TradeFill]]:
        if event_kind == "noise_market":
            return self._market_order_event(book, timestamp, fair_tick, rng, informed=False, last_market_direction=last_market_direction)
        if event_kind == "informed_market":
            return self._market_order_event(book, timestamp, fair_tick, rng, informed=True, last_market_direction=last_market_direction)
        if event_kind == "liquidity_add":
            return self._liquidity_add_event(book, timestamp, fair_tick, rng), []
        if event_kind == "cancel":
            return self._cancel_event(book, timestamp, fair_tick, rng), []
        if event_kind == "mm_refresh":
            mm_id = str(rng.choice(list(mm_inventory)))
            inventory = mm_inventory[mm_id]
            return self._refresh_market_maker(book, mm_id, inventory, fair_tick, timestamp, rng), []
        raise ValueError(f"Unknown event kind {event_kind!r}")

    def _market_order_event(
        self,
        book: FIFOOrderBook,
        timestamp: pd.Timestamp,
        fair_tick: float,
        rng: np.random.Generator,
        informed: bool,
        last_market_direction: int,
    ) -> tuple[list[dict[str, float | int | str | pd.Timestamp]], list[TradeFill]]:
        mid_tick = book.midprice() / self.spec.tick_size if book.midprice() > 0 else fair_tick
        if informed:
            direction = 1 if fair_tick >= mid_tick else -1
            agent_type = "informed_taker"
            lots = max(1, int(round(rng.gamma(shape=2.0, scale=self.config.market_order_mean_lots * self.config.informed_order_scale))))
        else:
            bias = np.clip((fair_tick - mid_tick) / 4.0, -0.25, 0.25)
            if last_market_direction and rng.random() < self.config.order_flow_memory:
                direction = last_market_direction
            else:
                direction = 1 if rng.random() < 0.5 + bias else -1
            agent_type = "noise_taker"
            lots = max(1, int(round(rng.gamma(shape=1.8, scale=self.config.market_order_mean_lots))))
        quantity = lots * self.spec.lot_size
        side = "buy" if direction > 0 else "sell"
        fills = book.market_order(side, quantity, timestamp, agent_type)
        records = [
            {
                "timestamp": timestamp,
                "event_type": "market_order",
                "agent_type": agent_type,
                "agent_id": f"{agent_type}_{int(rng.integers(1_000_000))}",
                "side": side,
                "price": np.nan,
                "size": quantity,
                "fair_value": fair_tick * self.spec.tick_size,
                "maker_order_id": -1,
            }
        ]
        for fill in fills:
            records.append(
                {
                    "timestamp": timestamp,
                    "event_type": "fill_hint",
                    "agent_type": fill.maker_agent,
                    "agent_id": fill.maker_agent,
                    "side": "sell" if fill.aggressor_side == "B" else "buy",
                    "price": fill.price,
                    "size": fill.size,
                    "fair_value": fair_tick * self.spec.tick_size,
                    "maker_order_id": fill.maker_order_id,
                }
            )
        return records, fills

    def _liquidity_add_event(
        self,
        book: FIFOOrderBook,
        timestamp: pd.Timestamp,
        fair_tick: float,
        rng: np.random.Generator,
    ) -> list[dict[str, float | int | str | pd.Timestamp]]:
        best_bid = book.best_bid_tick() or int(round(fair_tick - 1))
        best_ask = book.best_ask_tick() or int(round(fair_tick + 1))
        stale_side = "ask" if fair_tick > 0.5 * (best_bid + best_ask) else "bid"
        side = stale_side if rng.random() < 0.55 else ("bid" if stale_side == "ask" else "ask")
        join_touch = rng.random() < self.config.touch_join_probability
        offset = 0 if join_touch else int(rng.geometric(1.0 - self.spec.depth_decay))
        if best_ask - best_bid > max(3, self.spec.default_spread_ticks * 2):
            base_tick = int(round(fair_tick)) + (1 if side == "ask" else -1)
        else:
            base_tick = best_ask if side == "ask" else best_bid
        tick = base_tick + offset if side == "ask" else base_tick - offset
        tick = _passive_tick(book, side, tick)
        lots = max(1, int(round(rng.gamma(shape=2.2, scale=self.config.limit_order_mean_lots))))
        quantity = lots * self.spec.lot_size
        book.add_limit_order(side, tick, quantity, f"lp_{int(rng.integers(1_000_000))}", "liquidity_provider", timestamp)
        return [
            {
                "timestamp": timestamp,
                "event_type": "limit_add",
                "agent_type": "liquidity_provider",
                "agent_id": "liquidity_provider",
                "side": side,
                "price": tick * self.spec.tick_size,
                "size": quantity,
                "fair_value": fair_tick * self.spec.tick_size,
                "maker_order_id": -1,
            }
        ]

    def _cancel_event(
        self,
        book: FIFOOrderBook,
        timestamp: pd.Timestamp,
        fair_tick: float,
        rng: np.random.Generator,
    ) -> list[dict[str, float | int | str | pd.Timestamp]]:
        book_mid = book.midprice() / self.spec.tick_size if book.midprice() > 0 else fair_tick
        stale_side = "ask" if fair_tick > book_mid else "bid"
        side = stale_side if rng.random() < self.config.stale_cancel_bias else ("bid" if stale_side == "ask" else "ask")
        removed = book.cancel_random_fraction(
            side,
            rng,
            mean_fraction=self.config.cancel_mean_fraction,
            near_touch_bias=min(0.92, 0.55 + 0.40 * self.config.queue_deplete_scale),
            lot_size=self.spec.lot_size,
        )
        if removed is None:
            return []
        return [
            {
                "timestamp": timestamp,
                "event_type": "cancel",
                "agent_type": removed.agent_type,
                "agent_id": removed.agent_id,
                "side": side,
                "price": removed.tick * self.spec.tick_size,
                "size": removed.quantity,
                "fair_value": fair_tick * self.spec.tick_size,
                "maker_order_id": removed.order_id,
                }
            ]

    def _touch_replenish(
        self,
        book: FIFOOrderBook,
        fair_tick: float,
        timestamp: pd.Timestamp,
        rng: np.random.Generator,
    ) -> list[dict[str, float | int | str | pd.Timestamp]]:
        records: list[dict[str, float | int | str | pd.Timestamp]] = []
        best_bid = book.best_bid_tick()
        best_ask = book.best_ask_tick()
        if best_bid is None or best_ask is None:
            return records
        mid_tick = 0.5 * (best_bid + best_ask)
        stale_side = "ask" if fair_tick > mid_tick else "bid"
        top_target = int(self.spec.base_depth * self.config.queue_deplete_scale)
        for side in (stale_side, "bid" if stale_side == "ask" else "ask"):
            if rng.random() > self.config.touch_replenish_probability:
                continue
            touch_tick = best_ask if side == "ask" else best_bid
            touch_tick = _passive_tick(book, side, touch_tick)
            current_depth = book.aggregated_depth(side, touch_tick)
            if current_depth >= top_target:
                continue
            deficit = max(top_target - current_depth, self.spec.lot_size)
            base_lots = max(1, int(round(rng.gamma(shape=2.0, scale=self.config.limit_order_mean_lots * 0.8))))
            quantity = max(deficit, base_lots * self.spec.lot_size)
            book.add_limit_order(side, touch_tick, quantity, f"touch_{side}_{int(rng.integers(1_000_000))}", "touch_replenisher", timestamp)
            records.append(
                {
                    "timestamp": timestamp,
                    "event_type": "touch_replenish",
                    "agent_type": "touch_replenisher",
                    "agent_id": "touch_replenisher",
                    "side": side,
                    "price": touch_tick * self.spec.tick_size,
                    "size": quantity,
                    "fair_value": fair_tick * self.spec.tick_size,
                    "maker_order_id": -1,
                }
            )
        return records

    def _refresh_market_maker(
        self,
        book: FIFOOrderBook,
        mm_id: str,
        inventory: int,
        fair_tick: float,
        timestamp: pd.Timestamp,
        rng: np.random.Generator,
    ) -> list[dict[str, float | int | str | pd.Timestamp]]:
        records: list[dict[str, float | int | str | pd.Timestamp]] = []
        book.cancel_agent_orders(mm_id)
        top_bid = book.best_bid_tick() or int(round(fair_tick - self.spec.default_spread_ticks))
        top_ask = book.best_ask_tick() or int(round(fair_tick + self.spec.default_spread_ticks))
        touch_spread = max(top_ask - top_bid, 1)
        local_spread = min(touch_spread, max(2, self.spec.default_spread_ticks * 2))
        desired_half = max(1.0, 0.5 * local_spread * self.config.mm_refresh_sensitivity + self.config.mm_half_spread_ticks)
        reservation = fair_tick - np.sign(inventory) * self.config.mm_inventory_skew_ticks * min(abs(inventory) / self.spec.lot_size, 8.0)
        bid_tick = int(np.floor(reservation - desired_half))
        ask_tick = int(np.ceil(reservation + desired_half))
        bid_tick = _passive_tick(book, "bid", bid_tick)
        ask_tick = _passive_tick(book, "ask", ask_tick)
        if ask_tick <= bid_tick:
            ask_tick = bid_tick + 1
        for level in range(self.config.mm_depth_levels):
            level_size = max(1, int(round(rng.gamma(shape=1.8, scale=1.2)))) * self.spec.lot_size
            bid_level = bid_tick - level
            ask_level = ask_tick + level
            book.add_limit_order("bid", bid_level, level_size, mm_id, "competing_mm", timestamp)
            book.add_limit_order("ask", ask_level, level_size, mm_id, "competing_mm", timestamp)
            records.append(
                {
                    "timestamp": timestamp,
                    "event_type": "mm_refresh",
                    "agent_type": "competing_mm",
                    "agent_id": mm_id,
                    "side": "bid",
                    "price": bid_level * self.spec.tick_size,
                    "size": level_size,
                    "fair_value": fair_tick * self.spec.tick_size,
                    "maker_order_id": -1,
                }
            )
            records.append(
                {
                    "timestamp": timestamp,
                    "event_type": "mm_refresh",
                    "agent_type": "competing_mm",
                    "agent_id": mm_id,
                    "side": "ask",
                    "price": ask_level * self.spec.tick_size,
                    "size": level_size,
                    "fair_value": fair_tick * self.spec.tick_size,
                    "maker_order_id": -1,
                }
            )
        return records

    def _seed_book(
        self,
        book: FIFOOrderBook,
        fair_tick: float,
        timestamp: pd.Timestamp,
        rng: np.random.Generator,
        replenish_only: bool = False,
    ) -> None:
        center_tick = int(round(fair_tick))
        best_bid = book.best_bid_tick()
        best_ask = book.best_ask_tick()
        if best_bid is None:
            best_bid = center_tick - max(self.spec.default_spread_ticks, 1)
        if best_ask is None:
            best_ask = center_tick + max(self.spec.default_spread_ticks, 1)
        if best_ask <= best_bid:
            best_ask = best_bid + self.spec.default_spread_ticks
        if best_ask - best_bid > max(4, self.spec.default_spread_ticks * 2):
            half = max(self.spec.default_spread_ticks, 1)
            best_bid = _passive_tick(book, "bid", center_tick - half)
            best_ask = _passive_tick(book, "ask", center_tick + half)
            if best_ask <= best_bid:
                best_ask = best_bid + 1
        for level in range(self.config.levels):
            bid_tick = best_bid - level
            ask_tick = best_ask + level
            target_depth = int(self.spec.base_depth * (self.spec.depth_decay ** level))
            if not replenish_only or book.aggregated_depth("bid", bid_tick) < int(target_depth * self.config.queue_replenish_scale):
                extra = max(target_depth - book.aggregated_depth("bid", bid_tick), 0)
                if extra > 0:
                    quantity = int(max(extra, self.spec.lot_size))
                    book.add_limit_order("bid", bid_tick, quantity, f"seed_bid_{level}", "seed_provider", timestamp)
            if not replenish_only or book.aggregated_depth("ask", ask_tick) < int(target_depth * self.config.queue_replenish_scale):
                extra = max(target_depth - book.aggregated_depth("ask", ask_tick), 0)
                if extra > 0:
                    quantity = int(max(extra, self.spec.lot_size))
                    book.add_limit_order("ask", ask_tick, quantity, f"seed_ask_{level}", "seed_provider", timestamp)

    def _needs_replenishment(self, book: FIFOOrderBook) -> bool:
        best_bid = book.best_bid_tick()
        best_ask = book.best_ask_tick()
        if best_bid is None or best_ask is None:
            return True
        if best_ask - best_bid > 4:
            return True
        top = book.top_levels()
        top_target = int(self.spec.base_depth * self.config.queue_deplete_scale)
        if top["bid"][0][1] < top_target or top["ask"][0][1] < top_target:
            return True
        return top["bid"][self.config.levels - 1][1] == 0 or top["ask"][self.config.levels - 1][1] == 0


def _stable_seed(base_seed: int, symbol: str, day: str) -> int:
    payload = f"{base_seed}|{symbol}|{day}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big", signed=False) % (2**32)


def _paper_msg_columns() -> list[str]:
    return [
        "timestamp",
        "market_buy_volume",
        "market_buy_n",
        "market_sell_volume",
        "market_sell_n",
        "limit_buy_volume",
        "limit_buy_n",
        "limit_sell_volume",
        "limit_sell_n",
        "withdraw_buy_volume",
        "withdraw_buy_n",
        "withdraw_sell_volume",
        "withdraw_sell_n",
    ]


def _paper_msg_row(
    timestamp: pd.Timestamp,
    records: list[dict[str, float | int | str | pd.Timestamp]],
) -> dict[str, float | int | pd.Timestamp]:
    row: dict[str, float | int | pd.Timestamp] = {column: 0 for column in _paper_msg_columns()}
    row["timestamp"] = timestamp
    for record in records:
        event_type = str(record.get("event_type", ""))
        side = str(record.get("side", ""))
        size = int(record.get("size", 0) or 0)
        if size <= 0:
            continue
        if event_type == "market_order":
            if side == "buy":
                row["market_buy_volume"] = int(row["market_buy_volume"]) + size
                row["market_buy_n"] = int(row["market_buy_n"]) + 1
            elif side == "sell":
                row["market_sell_volume"] = int(row["market_sell_volume"]) + size
                row["market_sell_n"] = int(row["market_sell_n"]) + 1
        elif event_type in {"limit_add", "touch_replenish", "mm_refresh"}:
            if side == "bid":
                row["limit_buy_volume"] = int(row["limit_buy_volume"]) + size
                row["limit_buy_n"] = int(row["limit_buy_n"]) + 1
            elif side == "ask":
                row["limit_sell_volume"] = int(row["limit_sell_volume"]) + size
                row["limit_sell_n"] = int(row["limit_sell_n"]) + 1
        elif event_type == "cancel":
            if side == "bid":
                row["withdraw_buy_volume"] = int(row["withdraw_buy_volume"]) + size
                row["withdraw_buy_n"] = int(row["withdraw_buy_n"]) + 1
            elif side == "ask":
                row["withdraw_sell_volume"] = int(row["withdraw_sell_volume"]) + size
                row["withdraw_sell_n"] = int(row["withdraw_sell_n"]) + 1
    return row


def _generate_timestamps(
    day: str,
    spec: SymbolSpec,
    config: SimulatorConfig,
    rng: np.random.Generator,
) -> pd.DatetimeIndex:
    day_ts = pd.Timestamp(day)
    minute_stamps: list[pd.Timestamp] = []
    for raw in config.session_windows:
        start_s, end_s = raw.split("-", maxsplit=1)
        start = day_ts + pd.to_timedelta(start_s)
        end = day_ts + pd.to_timedelta(end_s)
        current = start
        while current < end:
            minute_stamps.append(current)
            current += pd.Timedelta(minutes=1)

    weights = []
    for idx, minute in enumerate(minute_stamps):
        pos = idx / max(len(minute_stamps) - 1, 1)
        u_shape = 1.2 + 0.9 * abs(pos - 0.5) * 2.0
        stable_bonus = 0.3 if _is_in_windows(minute, config.stable_windows) else 0.0
        weights.append(u_shape + stable_bonus)
    probs = np.asarray(weights, dtype=np.float64)
    probs /= probs.sum()
    counts = rng.multinomial(spec.events_per_day, probs)

    stamps: list[pd.Timestamp] = []
    for minute, count in zip(minute_stamps, counts, strict=True):
        if count == 0:
            continue
        offsets = np.sort(rng.uniform(0.0, 60.0, size=count))
        jitter = rng.normal(0.0, config.timestamp_jitter_fraction, size=count)
        for raw in np.clip(offsets + jitter, 0.0, 59.999):
            stamps.append(minute + pd.to_timedelta(float(raw), unit="s"))
    return pd.DatetimeIndex(stamps)


def _is_in_windows(ts: pd.Timestamp, windows: list[str]) -> bool:
    clock = ts.strftime("%H:%M:%S")
    for raw in windows:
        start, end = raw.split("-", maxsplit=1)
        if start <= clock <= end:
            return True
    return False


def _top_level_rows(
    timestamp: pd.Timestamp,
    top_levels: dict[str, list[tuple[float, int]]],
) -> tuple[dict[str, float | int | pd.Timestamp], dict[str, float | int | pd.Timestamp]]:
    ask_row: dict[str, float | int | pd.Timestamp] = {"timestamp": timestamp}
    bid_row: dict[str, float | int | pd.Timestamp] = {"timestamp": timestamp}
    for level, (price, size) in enumerate(top_levels["ask"], start=1):
        ask_row[f"ask{level}_price"] = price
        ask_row[f"ask{level}_volume"] = size
    for level, (price, size) in enumerate(top_levels["bid"], start=1):
        bid_row[f"bid{level}_price"] = price
        bid_row[f"bid{level}_volume"] = size
    return ask_row, bid_row


def _top_imbalance(book: FIFOOrderBook) -> float:
    bid = book.best_bid_tick()
    ask = book.best_ask_tick()
    if bid is None or ask is None:
        return 0.0
    bid_depth = book.aggregated_depth("bid", bid)
    ask_depth = book.aggregated_depth("ask", ask)
    total = bid_depth + ask_depth
    if total == 0:
        return 0.0
    return float((bid_depth - ask_depth) / total)


def _spread_ticks(book: FIFOOrderBook) -> int:
    bid = book.best_bid_tick()
    ask = book.best_ask_tick()
    if bid is None or ask is None:
        return 0
    return int(max(ask - bid, 0))


def _passive_tick(book: FIFOOrderBook, side: str, proposed_tick: int) -> int:
    if side == "bid":
        best_ask = book.best_ask_tick()
        return proposed_tick if best_ask is None else min(proposed_tick, best_ask - 1)
    best_bid = book.best_bid_tick()
    return proposed_tick if best_bid is None else max(proposed_tick, best_bid + 1)


def _queue_pressure(book: FIFOOrderBook, fair_tick: float) -> float:
    bid = book.best_bid_tick()
    ask = book.best_ask_tick()
    if bid is None or ask is None:
        return 0.0
    fair_side = 1 if fair_tick >= 0.5 * (bid + ask) else -1
    if fair_side > 0:
        bid_depth = book.aggregated_depth("bid", bid)
        ask_depth = book.aggregated_depth("ask", ask)
        return float((ask_depth - bid_depth) / max(ask_depth + bid_depth, 1))
    bid_depth = book.aggregated_depth("bid", bid)
    ask_depth = book.aggregated_depth("ask", ask)
    return float((bid_depth - ask_depth) / max(ask_depth + bid_depth, 1))


def _microprice(top_levels: dict[str, list[tuple[float, int]]]) -> float:
    bid_p, bid_v = top_levels["bid"][0]
    ask_p, ask_v = top_levels["ask"][0]
    total = bid_v + ask_v
    if total == 0:
        return 0.5 * (bid_p + ask_p)
    return (ask_p * bid_v + bid_p * ask_v) / total


def _depth_frame(book: FIFOOrderBook, center_tick: int, radius: int) -> np.ndarray:
    frame = np.zeros(2 * radius + 1, dtype=np.float32)
    for rel in range(-radius, radius + 1):
        tick = center_tick + rel
        idx = rel + radius
        if rel < 0:
            frame[idx] = book.aggregated_depth("bid", tick)
        elif rel > 0:
            frame[idx] = -book.aggregated_depth("ask", tick)
    return frame
