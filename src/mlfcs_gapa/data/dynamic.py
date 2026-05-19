"""Dynamic market-state features from the paper."""

from __future__ import annotations

import numpy as np

from mlfcs_gapa.data.schema import LobDataset


RV_RSI_WINDOWS_SECONDS: tuple[int, int, int] = (300, 600, 1800)
OSI_WINDOWS_SECONDS: tuple[int, int, int] = (10, 60, 300)


def dynamic_market_state(dataset: LobDataset, index: int) -> np.ndarray:
    """Return the 24-dimensional dynamic state described in the paper."""

    timestamps = _timestamps(dataset)
    ask1 = dataset.orderbook["ask1_price"].to_numpy().astype(np.float64)
    bid1 = dataset.orderbook["bid1_price"].to_numpy().astype(np.float64)
    midprices = (ask1 + bid1) / 2.0

    features: list[float] = []
    for seconds in RV_RSI_WINDOWS_SECONDS:
        mask = _time_window_mask(timestamps, index, seconds)
        window_prices = midprices[mask]
        features.append(_realized_volatility(window_prices) * 1e4)

    for seconds in RV_RSI_WINDOWS_SECONDS:
        mask = _time_window_mask(timestamps, index, seconds)
        window_prices = midprices[mask]
        features.append(_relative_strength_index(window_prices))

    messages = dataset.messages
    for seconds in OSI_WINDOWS_SECONDS:
        mask = _time_window_mask(timestamps, index, seconds)
        features.extend(
            [
                _order_strength_index(
                    messages["market_buy_volume"].to_numpy()[mask],
                    messages["market_sell_volume"].to_numpy()[mask],
                ),
                _order_strength_index(
                    messages["market_buy_n"].to_numpy()[mask],
                    messages["market_sell_n"].to_numpy()[mask],
                ),
                _order_strength_index(
                    messages["limit_buy_volume"].to_numpy()[mask],
                    messages["limit_sell_volume"].to_numpy()[mask],
                ),
                _order_strength_index(
                    messages["limit_buy_n"].to_numpy()[mask],
                    messages["limit_sell_n"].to_numpy()[mask],
                ),
                _order_strength_index(
                    messages["withdraw_buy_volume"].to_numpy()[mask],
                    messages["withdraw_sell_volume"].to_numpy()[mask],
                ),
                _order_strength_index(
                    messages["withdraw_buy_n"].to_numpy()[mask],
                    messages["withdraw_sell_n"].to_numpy()[mask],
                ),
            ]
        )

    state = np.asarray(features, dtype=np.float32)
    if state.shape != (24,):
        raise RuntimeError(f"dynamic state must have shape (24,), got {state.shape}")
    return state


def _timestamps(dataset: LobDataset) -> np.ndarray:
    return dataset.orderbook["timestamp"].to_numpy()


def _time_window_mask(timestamps: np.ndarray, index: int, seconds: int) -> np.ndarray:
    current = timestamps[index]
    start = current - np.timedelta64(seconds, "s")
    return (timestamps <= current) & (timestamps >= start)


def _realized_volatility(prices: np.ndarray) -> float:
    if len(prices) < 2:
        return 0.0
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    return float(np.sqrt(np.square(returns).sum()))


def _relative_strength_index(prices: np.ndarray) -> float:
    if len(prices) < 2:
        return 0.5
    changes = np.diff(prices)
    gains = np.maximum(changes, 0.0).sum()
    losses = np.maximum(-changes, 0.0).sum()
    if gains == 0 and losses == 0:
        return 0.5
    return float(gains / (gains + losses + 1e-7))


def _order_strength_index(buy: np.ndarray, sell: np.ndarray) -> float:
    buy_sum = float(np.asarray(buy, dtype=np.float64).sum())
    sell_sum = float(np.asarray(sell, dtype=np.float64).sum())
    return (buy_sum - sell_sum) / (buy_sum + sell_sum + 1e-7)
