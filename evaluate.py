from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from baselines import astar_action, greedy_action, random_action
from warehouse_env import SmartWarehouseEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--obstacle-density", type=float, default=0.18)
    parser.add_argument("--orders", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model-path", type=str, default="models/ppo_warehouse")
    parser.add_argument("--no-baselines", action="store_true")
    return parser.parse_args()


def run_episodes(policy, env_kwargs: dict, n_episodes: int) -> dict:
    results = defaultdict(list)

    for ep in range(n_episodes):
        env = SmartWarehouseEnv(**env_kwargs, seed=ep)
        obs, _ = env.reset()
        done = False

        total_reward = 0.0

        while not done:
            action = policy(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        results["reward"].append(total_reward)
        results["completed"].append(info["completed_orders"])
        results["total_orders"].append(info["total_orders"])
        results["collisions"].append(info["collisions"])
        results["steps"].append(env.steps)
        results["success"].append(int(info["completed_orders"] == info["total_orders"]))

    return results


def summarize(name: str, results: dict):
    n = len(results["reward"])
    success_rate = np.mean(results["success"]) * 100
    avg_reward = np.mean(results["reward"])
    avg_orders = np.mean(results["completed"])
    total_orders = results["total_orders"][0]
    avg_collisions = np.mean(results["collisions"])
    avg_steps = np.mean(results["steps"])

    print(f"\n{'='*40}")
    print(f"  {name}  ({n} episodes)")
    print(f"{'='*40}")
    print(f"  Success rate:      {success_rate:.1f}%")
    print(f"  Avg reward:        {avg_reward:.2f}")
    print(f"  Avg orders done:   {avg_orders:.2f} / {total_orders}")
    print(f"  Avg collisions:    {avg_collisions:.2f}")
    print(f"  Avg steps:         {avg_steps:.1f}")


def main():
    args = parse_args()

    env_kwargs = dict(
        grid_size=args.grid_size,
        obstacle_density=args.obstacle_density,
        n_orders=args.orders,
    )

    model_path = Path(args.model_path)
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"No model found at {model_path}.zip — train first.")

    model = PPO.load(model_path)

    def ppo_policy(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    print(f"\nEvaluating over {args.episodes} episodes  "
          f"(grid={args.grid_size}, orders={args.orders})")

    results = run_episodes(ppo_policy, env_kwargs, args.episodes)
    summarize("PPO (trained)", results)

    if not args.no_baselines:
        for name, fn in [
            ("A* baseline", lambda obs, env: astar_action(env)),
            ("Greedy baseline", lambda obs, env: greedy_action(env)),
            ("Random baseline", lambda obs, env: random_action(env)),
        ]:
            results = run_episodes(fn, env_kwargs, args.episodes)
            summarize(name, results)


if __name__ == "__main__":
    main()
