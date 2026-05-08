from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DiagnosticsConfig
from .paper_env import EpisodeMetrics, PaperAction, PaperTradingEnv
from .paper_evaluation import summarize_metrics
from .simulator import SyntheticDay
from .utils import save_json


INVENTORY_ACTIONS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)


class TabularQPolicy:
    uses_state = False

    def __init__(self, name: str, q_values: dict[tuple[int, ...], np.ndarray], config: DiagnosticsConfig) -> None:
        self.name = name
        self.q_values = q_values
        self.config = config

    def act(self, state, env: PaperTradingEnv) -> PaperAction:
        state_key = _state_for_policy(self.name, env, self.config)
        action = _greedy_action(self.q_values, state_key, _action_count(self.name), _admissible_actions(self.name, state_key, env, self.config))
        return _action_to_quote(self.name, action, env, self.config)


def train_inventory_rl(days: list[SyntheticDay], config: DiagnosticsConfig, output_dir: Path) -> Path:
    return _train_tabular_baseline("Inventory-RL", days, config, output_dir)


def train_lob_rl(days: list[SyntheticDay], config: DiagnosticsConfig, output_dir: Path) -> Path:
    return _train_tabular_baseline("LOB-RL", days, config, output_dir)


def evaluate_inventory_rl(days: list[SyntheticDay], config: DiagnosticsConfig, qtable_path: Path, output_dir: Path) -> dict[str, float]:
    return _evaluate_tabular_baseline("Inventory-RL", days, config, qtable_path, output_dir)


def evaluate_lob_rl(days: list[SyntheticDay], config: DiagnosticsConfig, qtable_path: Path, output_dir: Path) -> dict[str, float]:
    return _evaluate_tabular_baseline("LOB-RL", days, config, qtable_path, output_dir)


def _train_tabular_baseline(name: str, days: list[SyntheticDay], config: DiagnosticsConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    q_values: dict[tuple[int, ...], np.ndarray] = {}
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float | int | str]] = []
    episode_specs = [(day, span_idx, span) for day in days for span_idx, span in enumerate(_episode_spans(day, config, config.max_train_episodes_per_day))]
    if not episode_specs:
        raise ValueError(f"No train episodes available for {name}")
    epochs = max(int(config.tabular_epochs), 1)
    for epoch in range(epochs):
        alpha = _linear_schedule(config.tabular_alpha_start, config.tabular_alpha_end, epoch, epochs)
        epsilon = _linear_schedule(config.tabular_epsilon_start, config.tabular_epsilon_end, epoch, epochs)
        order = rng.permutation(len(episode_specs))
        epoch_metrics: list[EpisodeMetrics] = []
        for position in order:
            day, span_idx, (start, stop) = episode_specs[int(position)]
            metrics = _train_episode(name, q_values, day, span_idx, start, stop, config, alpha, epsilon, rng)
            epoch_metrics.append(metrics)
        frame = pd.DataFrame([asdict(metric) for metric in epoch_metrics])
        history.append(
            {
                "epoch": epoch + 1,
                "policy": name,
                "episodes": int(len(frame)),
                "epsilon": float(epsilon),
                "alpha": float(alpha),
                "pnl_mean": float(frame["pnl"].mean()),
                "reward_mean": float(frame["reward"].mean()),
                "avg_abs_position_mean": float(frame["avg_abs_position"].mean()),
                "fill_rate_mean": float(frame["fill_rate"].mean()),
                "states": int(len(q_values)),
            }
        )
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / f"{_slug(name)}_train_history.csv", index=False)
    path = output_dir / f"{_slug(name)}_qtable.npz"
    _save_qtable(path, q_values)
    save_json(
        output_dir / f"{_slug(name)}_summary.json",
        {
            "policy": name,
            "qtable_path": str(path),
            "states": len(q_values),
            "action_count": _action_count(name),
            "final_epoch": history[-1],
            "assumptions": _assumptions(name, config),
        },
    )
    return path


def _evaluate_tabular_baseline(name: str, days: list[SyntheticDay], config: DiagnosticsConfig, qtable_path: Path, output_dir: Path) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    q_values = _load_qtable(qtable_path, _action_count(name))
    metrics: list[EpisodeMetrics] = []
    for day in days:
        for episode_index, (start, stop) in enumerate(_episode_spans(day, config, config.max_eval_episodes_per_day)):
            env = PaperTradingEnv(day, config, start, stop, episode_index=episode_index, policy_name=name, rng_seed=config.seed)
            env.reset()
            terminal = False
            while not terminal:
                state_key = _state_for_policy(name, env, config)
                action = _greedy_action(q_values, state_key, _action_count(name), _admissible_actions(name, state_key, env, config))
                result = env.step(_action_to_quote(name, action, env, config), compute_next_state=False)
                terminal = result.terminal
            metrics.append(env.metrics())
    frame = pd.DataFrame([asdict(metric) for metric in metrics])
    frame.to_csv(output_dir / f"{_slug(name)}_episodes.csv", index=False)
    daily = _daily_results(frame)
    daily.to_csv(output_dir / f"{_slug(name)}_daily.csv", index=False)
    summary = summarize_metrics(frame)
    save_json(output_dir / f"{_slug(name)}_eval_summary.json", {"policy": name, **summary, "assumptions": _assumptions(name, config)})
    return summary


def _train_episode(
    name: str,
    q_values: dict[tuple[int, ...], np.ndarray],
    day: SyntheticDay,
    episode_index: int,
    start: int,
    stop: int,
    config: DiagnosticsConfig,
    alpha: float,
    epsilon: float,
    rng: np.random.Generator,
) -> EpisodeMetrics:
    env = PaperTradingEnv(day, config, start, stop, episode_index=episode_index, policy_name=name, rng_seed=config.seed)
    env.reset()
    terminal = False
    while not terminal:
        state_key = _state_for_policy(name, env, config)
        admissible = _admissible_actions(name, state_key, env, config)
        action = _epsilon_greedy(q_values, state_key, _action_count(name), admissible, epsilon, rng)
        result = env.step(_action_to_quote(name, action, env, config), compute_next_state=False)
        next_key = _state_for_policy(name, env, config) if not result.terminal else None
        _q_update(q_values, state_key, action, result.reward, next_key, _action_count(name), alpha, config.discount, name, env, config)
        terminal = result.terminal
    return env.metrics()


def _state_for_policy(name: str, env: PaperTradingEnv, config: DiagnosticsConfig) -> tuple[int, ...]:
    if name == "Inventory-RL":
        return (_inventory_bucket(env.inventory, config), _time_bucket(env, config))
    if name == "LOB-RL":
        return (
            _side_speed(env, "A", config),
            _side_speed(env, "B", config),
            _mid_change_bucket(env, config),
            _lob_inventory_bucket(env.inventory, config),
            1 if env.value > config.tabular_pnl_threshold else 0,
        )
    raise ValueError(f"Unknown tabular policy {name}")


def _action_to_quote(name: str, action: int, env: PaperTradingEnv, config: DiagnosticsConfig) -> PaperAction:
    quote_idx = max(env.event_idx - config.latency, 0)
    tick = config.symbol_spec.tick_size
    lot = config.trade_unit
    price = env.day.price.iloc[quote_idx]
    ask1 = float(price["ask1_price"])
    bid1 = float(price["bid1_price"])
    if name == "Inventory-RL":
        bid_offset, ask_offset = INVENTORY_ACTIONS[action]
        return PaperAction(ask_price=ask1 + ask_offset * tick, ask_volume=-lot, bid_price=bid1 - bid_offset * tick, bid_volume=lot)
    if name == "LOB-RL":
        ask_volume = -lot if action in {1, 3} else 0
        bid_volume = lot if action in {2, 3} else 0
        return PaperAction(ask_price=ask1 if ask_volume else 0.0, ask_volume=ask_volume, bid_price=bid1 if bid_volume else 0.0, bid_volume=bid_volume)
    raise ValueError(f"Unknown tabular policy {name}")


def _admissible_actions(name: str, state_key: tuple[int, ...], env: PaperTradingEnv, config: DiagnosticsConfig) -> list[int]:
    if name == "Inventory-RL":
        return list(range(len(INVENTORY_ACTIONS)))
    if name == "LOB-RL":
        inv_bucket = state_key[3]
        if inv_bucket == 0:
            return [0, 2]
        if inv_bucket == 4:
            return [0, 1]
        return [0, 1, 2, 3]
    raise ValueError(f"Unknown tabular policy {name}")


def _q_update(
    q_values: dict[tuple[int, ...], np.ndarray],
    state_key: tuple[int, ...],
    action: int,
    reward: float,
    next_key: tuple[int, ...] | None,
    action_count: int,
    alpha: float,
    discount: float,
    name: str,
    env: PaperTradingEnv,
    config: DiagnosticsConfig,
) -> None:
    row = q_values.setdefault(state_key, np.zeros(action_count, dtype=np.float64))
    if next_key is None:
        target = reward
    else:
        next_row = q_values.setdefault(next_key, np.zeros(action_count, dtype=np.float64))
        admissible = _admissible_actions(name, next_key, env, config)
        target = reward + discount * float(np.max(next_row[admissible]))
    row[action] += alpha * (target - row[action])


def _epsilon_greedy(
    q_values: dict[tuple[int, ...], np.ndarray],
    state_key: tuple[int, ...],
    action_count: int,
    admissible: list[int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.choice(admissible))
    return _greedy_action(q_values, state_key, action_count, admissible)


def _greedy_action(q_values: dict[tuple[int, ...], np.ndarray], state_key: tuple[int, ...], action_count: int, admissible: list[int]) -> int:
    row = q_values.get(state_key)
    if row is None:
        row = np.zeros(action_count, dtype=np.float64)
    scores = row[admissible]
    return int(admissible[int(np.argmax(scores))])


def _episode_spans(day: SyntheticDay, config: DiagnosticsConfig, limit: int | None) -> list[tuple[int, int]]:
    clock = day.price["timestamp"].dt.strftime("%H:%M:%S")
    mask = np.zeros(len(clock), dtype=bool)
    for raw in config.stable_windows:
        start, end = raw.split("-", maxsplit=1)
        mask |= (clock >= start) & (clock <= end)
    idx = np.flatnonzero(mask)
    spans: list[tuple[int, int]] = []
    block_start = 0
    for pos in range(1, len(idx) + 1):
        boundary = pos == len(idx) or idx[pos] != idx[pos - 1] + 1
        if not boundary:
            continue
        block = idx[block_start:pos]
        for offset in range(0, len(block), config.episode_length):
            window = block[offset : offset + config.episode_length]
            if len(window) == config.episode_length:
                spans.append((int(window[0]), int(window[-1]) + 1))
                if limit is not None and len(spans) >= limit:
                    return spans
        block_start = pos
    return spans


def _inventory_bucket(inventory: int, config: DiagnosticsConfig) -> int:
    units = inventory / max(config.trade_unit, 1)
    if units <= -4:
        return 0
    if units <= -2:
        return 1
    if units < 0:
        return 2
    if units == 0:
        return 3
    if units < 2:
        return 4
    if units < 4:
        return 5
    return 6


def _lob_inventory_bucket(inventory: int, config: DiagnosticsConfig) -> int:
    units = inventory / max(config.trade_unit, 1)
    if units <= -4:
        return 0
    if units < 0:
        return 1
    if units == 0:
        return 2
    if units < 4:
        return 3
    return 4


def _time_bucket(env: PaperTradingEnv, config: DiagnosticsConfig) -> int:
    progress = (env.event_idx - env.episode_start) / max(env.episode_stop - env.episode_start, 1)
    return int(np.clip(np.floor(progress * config.tabular_time_bins), 0, config.tabular_time_bins - 1))


def _side_speed(env: PaperTradingEnv, aggressor_side: str, config: DiagnosticsConfig) -> int:
    lower = max(env.event_idx - config.tabular_lob_lookback, env.episode_start)
    if env.day.trades.empty:
        return 0
    timestamps = set(env.day.price.iloc[lower : env.event_idx + 1]["timestamp"])
    trades = env.day.trades
    block = trades[trades["timestamp"].isin(timestamps)]
    if block.empty:
        return 0
    recent_volume = float(block.loc[block["aggressor_side"] == aggressor_side, "size"].sum())
    quote_idx = max(env.event_idx - config.latency, 0)
    depth_col = "bid1_volume" if aggressor_side == "A" else "ask1_volume"
    top_depth = float(env.day.bid.iloc[quote_idx][depth_col] if aggressor_side == "A" else env.day.ask.iloc[quote_idx][depth_col])
    return int(recent_volume > top_depth)


def _mid_change_bucket(env: PaperTradingEnv, config: DiagnosticsConfig) -> int:
    lookback = max(int(config.tabular_lob_lookback), 1)
    start = max(env.event_idx - lookback, env.episode_start)
    if start >= env.event_idx:
        return 2
    tick = config.symbol_spec.tick_size
    delta_ticks = (float(env.day.price.iloc[env.event_idx]["midprice"]) - float(env.day.price.iloc[start]["midprice"])) / max(tick, 1e-12)
    if delta_ticks <= -config.tabular_mid_change_large_ticks:
        return 0
    if delta_ticks <= -config.tabular_mid_change_small_ticks:
        return 1
    if delta_ticks >= config.tabular_mid_change_large_ticks:
        return 4
    if delta_ticks >= config.tabular_mid_change_small_ticks:
        return 3
    return 2


def _action_count(name: str) -> int:
    if name == "Inventory-RL":
        return len(INVENTORY_ACTIONS)
    if name == "LOB-RL":
        return 4
    raise ValueError(f"Unknown tabular policy {name}")


def _slug(name: str) -> str:
    return name.lower().replace("-", "_")


def _linear_schedule(start: float, end: float, step: int, steps: int) -> float:
    if steps <= 1:
        return float(end)
    weight = step / float(steps - 1)
    return float((1.0 - weight) * start + weight * end)


def _save_qtable(path: Path, q_values: dict[tuple[int, ...], np.ndarray]) -> None:
    states = np.asarray([json.dumps(state) for state in q_values], dtype=object)
    q = np.vstack([q_values[state] for state in q_values]) if q_values else np.zeros((0, 0), dtype=np.float64)
    np.savez_compressed(path, states=states, q=q)


def _load_qtable(path: Path, action_count: int) -> dict[tuple[int, ...], np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    states = payload["states"]
    q = payload["q"]
    result: dict[tuple[int, ...], np.ndarray] = {}
    for raw_state, row in zip(states, q, strict=True):
        state = tuple(int(item) for item in json.loads(str(raw_state)))
        values = np.asarray(row, dtype=np.float64)
        if values.shape != (action_count,):
            raise ValueError(f"Q-table action count mismatch for {path}: expected {action_count}, got {values.shape}")
        result[state] = values
    return result


def _daily_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (policy, day), group in frame.groupby(["policy", "day"]):
        turnover = float(group["turnover"].sum())
        pnl = float(group["pnl"].sum())
        rows.append(
            {
                "policy": policy,
                "day": day,
                "pnl": pnl,
                "nd_pnl": float(group["nd_pnl"].sum()),
                "avg_abs_position": float(group["avg_abs_position"].mean()),
                "profit_ratio": pnl / max(turnover, 1e-8),
                "turnover": turnover,
                "episodes": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy", "day"])


def _assumptions(name: str, config: DiagnosticsConfig) -> dict[str, object]:
    if name == "Inventory-RL":
        return {
            "state": "inventory bucket x remaining-time bucket",
            "actions": "9 bid/ask offset pairs from {0,1,2} ticks",
            "learning": "tabular Q-learning, epsilon-greedy train, greedy eval",
            "reward": config.reward_mode,
        }
    return {
        "state": "bid speed, ask speed, mid-change bucket, inventory bucket, cumulative-PnL bucket",
        "actions": "neither, ask only, bid only, both at touch; strong-inventory restrictions applied",
        "learning": "tabular Q-learning, epsilon-greedy train, greedy eval",
        "reward": config.reward_mode,
    }
