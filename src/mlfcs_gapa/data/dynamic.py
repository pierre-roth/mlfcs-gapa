"""Dynamic market-state features from the paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlfcs_gapa.data.schema import LobDataset


RV_RSI_WINDOWS_SECONDS: tuple[int, int, int] = (300, 600, 1800)
OSI_WINDOWS_SECONDS: tuple[int, int, int] = (10, 60, 300)
_DYNAMIC_CACHE_BY_DATASET_ID: dict[int, "DynamicStateCache"] = {}


def dynamic_market_state(dataset: LobDataset, index: int) -> np.ndarray:
    """Return the 24-dimensional dynamic state described in the paper."""

    cache = _DYNAMIC_CACHE_BY_DATASET_ID.get(id(dataset))
    if cache is None:
        cache = DynamicStateCache.from_dataset(dataset)
        _DYNAMIC_CACHE_BY_DATASET_ID[id(dataset)] = cache
    return cache.state(index)


@dataclass(frozen=True)
class DynamicStateCache:
    """Cached arrays for repeated dynamic-state evaluation on one dataset."""

    timestamps: np.ndarray
    squared_log_return_prefix: np.ndarray
    gain_prefix: np.ndarray
    loss_prefix: np.ndarray
    message_prefixes: dict[str, np.ndarray]

    @classmethod
    def from_dataset(cls, dataset: LobDataset) -> "DynamicStateCache":
        timestamps = _timestamps(dataset)
        ask1 = dataset.orderbook["ask1_price"].to_numpy().astype(np.float64)
        bid1 = dataset.orderbook["bid1_price"].to_numpy().astype(np.float64)
        midprices = (ask1 + bid1) / 2.0

        log_returns = np.zeros_like(midprices)
        log_returns[1:] = np.diff(np.log(midprices))
        price_changes = np.zeros_like(midprices)
        price_changes[1:] = np.diff(midprices)

        message_prefixes = {
            column: _prefix_sum(dataset.messages[column].to_numpy().astype(np.float64))
            for column in (
                "market_buy_volume",
                "market_sell_volume",
                "market_buy_n",
                "market_sell_n",
                "limit_buy_volume",
                "limit_sell_volume",
                "limit_buy_n",
                "limit_sell_n",
                "withdraw_buy_volume",
                "withdraw_sell_volume",
                "withdraw_buy_n",
                "withdraw_sell_n",
            )
        }
        return cls(
            timestamps=timestamps,
            squared_log_return_prefix=_prefix_sum(np.square(log_returns)),
            gain_prefix=_prefix_sum(np.maximum(price_changes, 0.0)),
            loss_prefix=_prefix_sum(np.maximum(-price_changes, 0.0)),
            message_prefixes=message_prefixes,
        )

    def state(self, index: int) -> np.ndarray:
        features: list[float] = []
        for seconds in RV_RSI_WINDOWS_SECONDS:
            start, end = self._window_bounds(index, seconds)
            features.append(self._realized_volatility(start, end) * 1e4)

        for seconds in RV_RSI_WINDOWS_SECONDS:
            start, end = self._window_bounds(index, seconds)
            features.append(self._relative_strength_index(start, end))

        for seconds in OSI_WINDOWS_SECONDS:
            start, end = self._window_bounds(index, seconds)
            features.extend(
                [
                    self._order_strength_index(
                        "market_buy_volume", "market_sell_volume", start, end
                    ),
                    self._order_strength_index("market_buy_n", "market_sell_n", start, end),
                    self._order_strength_index("limit_buy_volume", "limit_sell_volume", start, end),
                    self._order_strength_index("limit_buy_n", "limit_sell_n", start, end),
                    self._order_strength_index(
                        "withdraw_buy_volume", "withdraw_sell_volume", start, end
                    ),
                    self._order_strength_index("withdraw_buy_n", "withdraw_sell_n", start, end),
                ]
            )

        state = np.asarray(features, dtype=np.float32)
        if state.shape != (24,):
            raise RuntimeError(f"dynamic state must have shape (24,), got {state.shape}")
        return state

    def _window_bounds(self, index: int, seconds: int) -> tuple[int, int]:
        current = self.timestamps[index]
        start_time = current - np.timedelta64(seconds, "s")
        start = int(np.searchsorted(self.timestamps, start_time, side="left"))
        return start, index + 1

    def _realized_volatility(self, start: int, end: int) -> float:
        if end - start < 2:
            return 0.0
        return float(np.sqrt(_range_sum(self.squared_log_return_prefix, start + 1, end)))

    def _relative_strength_index(self, start: int, end: int) -> float:
        if end - start < 2:
            return 0.5
        gains = _range_sum(self.gain_prefix, start + 1, end)
        losses = _range_sum(self.loss_prefix, start + 1, end)
        if gains == 0 and losses == 0:
            return 0.5
        return float(gains / (gains + losses + 1e-7))

    def _order_strength_index(
        self, buy_column: str, sell_column: str, start: int, end: int
    ) -> float:
        buy_sum = _range_sum(self.message_prefixes[buy_column], start, end)
        sell_sum = _range_sum(self.message_prefixes[sell_column], start, end)
        return (buy_sum - sell_sum) / (buy_sum + sell_sum + 1e-7)


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])


def _range_sum(prefix: np.ndarray, start: int, end: int) -> float:
    return float(prefix[end] - prefix[start])


def _timestamps(dataset: LobDataset) -> np.ndarray:
    return dataset.orderbook["timestamp"].to_numpy()
