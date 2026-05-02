from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from warehouse_env import SmartWarehouseEnv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--obstacle-density", type=float, default=0.18)
    parser.add_argument("--orders", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--dynamic-obstacles", action="store_true", default=False)
    parser.add_argument(
        "--no-dynamic-obstacles",
        dest="dynamic_obstacles",
        action="store_false",
    )
    parser.add_argument("--model-path", type=str, default="models/ppo_warehouse")

    return parser.parse_args()


def main():
    args = parse_args()

    Path("models").mkdir(exist_ok=True)

    def make_env():
        return SmartWarehouseEnv(
            grid_size=args.grid_size,
            obstacle_density=args.obstacle_density,
            n_orders=args.orders,
            dynamic_obstacles=args.dynamic_obstacles,
        )

    env = make_vec_env(make_env, n_envs=args.n_envs)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=lambda p: 3e-4 * p,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=args.timesteps)

    model.save(args.model_path)

    print(f"Saved trained PPO model to {args.model_path}.zip")


if __name__ == "__main__":
    main()