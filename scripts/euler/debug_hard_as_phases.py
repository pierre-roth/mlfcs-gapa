"""Phase-timed diagnostic for AS hard-window panel runs on Euler."""

from __future__ import annotations

import argparse
import faulthandler
import sys
import time
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv

from mlfcs_gapa.data.synthetic import SyntheticLobConfig, generate_synthetic_lob_day
from mlfcs_gapa.extensions.as_guidance import ASGuidanceConfig, make_as_strategy
from mlfcs_gapa.extensions.as_guided_env import ASGuidedMarketMakingEnv
from mlfcs_gapa.extensions.as_guided_panel import _make_ppo_model, _merge_lob_datasets


faulthandler.enable(file=sys.stderr, all_threads=True)
faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def timed(name: str, fn):
    start = time.perf_counter()
    log(f"BEGIN {name}")
    output = fn()
    log(f"END {name} {time.perf_counter() - start:.2f}s")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--events-per-day", type=int, default=2000)
    parser.add_argument("--train-days", type=int, default=2)
    args = parser.parse_args()

    class Config:
        variant = "hard_as"
        episode_events = 240
        n_envs = args.n_envs
        encoder_checkpoint = None
        freeze_encoder = False
        device = args.device

    log(
        "config "
        f"window={args.window} device={args.device} "
        f"timesteps={args.timesteps} n_envs={args.n_envs}"
    )

    def build_data():
        datasets = [
            generate_synthetic_lob_day(
                SyntheticLobConfig(
                    stock="000001",
                    day=f"2019-11-{index + 1:02d}",
                    n_events=args.events_per_day,
                    base_price=16.45,
                    seed=101 + index,
                )
            )
            for index in range(args.train_days)
        ]
        return _merge_lob_datasets(datasets, day="train")

    train_dataset = timed("build_data", build_data)
    as_strategy = timed(
        "make_as_strategy", lambda: make_as_strategy(train_dataset, episode_events=240)
    )
    guidance = ASGuidanceConfig(
        mode="hard",
        hard_window_bias=args.window,
        hard_window_spread=args.window,
        base_reward="profit",
    )

    def build_env():
        def make_env(rank: int):
            def factory():
                return ASGuidedMarketMakingEnv(
                    train_dataset,
                    as_strategy=as_strategy,
                    guidance=guidance,
                    episode_events=240,
                    latency_events=1,
                    normalize_actions=True,
                    random_episode_starts=True,
                    seed=30101 + rank,
                )

            return factory

        return DummyVecEnv([make_env(rank) for rank in range(args.n_envs)])

    env = timed("build_env", build_env)
    model = timed("make_ppo_model", lambda: _make_ppo_model(env, Config(), seed=30101))
    timed("learn", lambda: model.learn(total_timesteps=args.timesteps, progress_bar=False))
    output_dir = Path(
        f"/cluster/work/math/piroth/mlfcs-gapa/runs/extensions/"
        f"as-guided-debug/phases_w{args.window}_{args.device}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timed("save", lambda: model.save(output_dir / "ppo_model"))
    log("DONE")


if __name__ == "__main__":
    main()
