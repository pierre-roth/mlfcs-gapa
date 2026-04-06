"""
Minimal reward distortion diagnostic.
Runs a random policy for a few episodes and prints reward component breakdown.
No model training, no backbone — finishes in seconds.
"""
import sys
import numpy as np
from lobmmx.config import RLTrainConfig
from lobmmx.pipeline import load_symbol_splits
from lobmmx.env import MarketMakingEnv

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
symbol = sys.argv[2] if len(sys.argv) > 2 else "AAPL"

config = RLTrainConfig(
    data_dir=data_dir,
    symbols=[symbol],
    mode="smoke",
    train_days=2,
    val_days=1,
    test_days=1,
    random_initial_inventory=True,
)
config.apply_mode_defaults()

splits = load_symbol_splits(config, symbol)
day = splits["test"][0]

env = MarketMakingEnv(day, config, reward_mode=config.reward_mode)
episodes = env.selected_episodes(limit=5)

print(f"\nSymbol: {symbol}  reward_mode: {config.reward_mode}  terminal_inventory_cost_scale: {config.terminal_inventory_cost_scale}")
print(f"initial_inventory_max: {config.initial_inventory_max}  allow_terminal_inventory: {config.allow_terminal_inventory}\n")

for i, span in enumerate(episodes):
    env.reset(span)
    done = False
    while not done:
        action = np.random.uniform(0, 1, size=3).astype(np.float32)
        _, _, done, _ = env.step(action)
    env.episode_result(method="random", episode_index=i)
