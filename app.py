from __future__ import annotations

from pathlib import Path
from typing import List

import gradio as gr
from stable_baselines3 import PPO

from baselines import astar_action, greedy_action, random_action
from warehouse_env import SmartWarehouseEnv


MODEL_PATH = Path("models/ppo_warehouse.zip")


def run_demo(
    agent_type: str,
    grid_size: int,
    obstacle_density: float,
    orders: int,
    dynamic_obstacles: bool,
    seed: int,
):
    env = SmartWarehouseEnv(
        grid_size=int(grid_size),
        obstacle_density=float(obstacle_density),
        n_orders=int(orders),
        dynamic_obstacles=bool(dynamic_obstacles),
    )

    obs, info = env.reset(seed=int(seed))

    frames: List[str] = [env.render()]
    total_reward = 0.0
    done = False

    model = None

    if agent_type == "PPO":
        if not MODEL_PATH.exists():
            return (
                "No trained PPO model found. Run `python train.py` first, or choose A*, Greedy, or Random.",
                "",
            )

        try:
            model = PPO.load(str(MODEL_PATH))
        except Exception as exc:
            return (
                f"Could not load PPO model: {exc}\n\n"
                "Tip: PPO models only work with the same observation size they were trained on. "
                "Use the same grid size as training.",
                "",
            )

    while not done:
        if agent_type == "Random":
            action = random_action(env)
        elif agent_type == "Greedy":
            action = greedy_action(env)
        elif agent_type == "A*":
            action = astar_action(env)
        elif agent_type == "PPO":
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

        frames.append(env.render())

    summary = f"""
Agent: {agent_type}
Success: {info["completed_orders"] == info["total_orders"]}
Completed Orders: {info["completed_orders"]}/{info["total_orders"]}
Steps: {env.steps}
Collisions: {info["collisions"]}
Invalid Moves: {info["invalid_moves"]}
Total Reward: {total_reward:.2f}
"""

    route_text = "\n\n".join(
        f"Step {i}\n{frame}" for i, frame in enumerate(frames[:80])
    )

    if len(frames) > 80:
        route_text += f"\n\n...truncated {len(frames) - 80} additional steps..."

    return summary.strip(), route_text


with gr.Blocks(title="Smart Warehouse Routing Agent") as demo:
    gr.Markdown(
        """
# Smart Warehouse Routing Agent

Compare routing strategies for a simulated warehouse robot completing pickup/dropoff tasks.

- **Random** = no strategy
- **Greedy** = always move closer to the target
- **A\*** = traditional path planning
- **PPO** = reinforcement learning policy, available after training
"""
    )

    with gr.Row():
        agent_type = gr.Dropdown(
            choices=["A*", "Greedy", "Random", "PPO"],
            value="A*",
            label="Agent",
        )

        grid_size = gr.Slider(
            minimum=5,
            maximum=15,
            value=8,
            step=1,
            label="Warehouse Size",
        )

        obstacle_density = gr.Slider(
            minimum=0.05,
            maximum=0.40,
            value=0.18,
            step=0.01,
            label="Shelf / Obstacle Density",
        )

        orders = gr.Slider(
            minimum=1,
            maximum=6,
            value=3,
            step=1,
            label="Number of Orders",
        )

    with gr.Row():
        dynamic_obstacles = gr.Checkbox(
            value=True,
            label="Dynamic blocked aisles",
        )

        seed = gr.Number(
            value=42,
            precision=0,
            label="Random Seed",
        )

    run_button = gr.Button("Run Simulation")

    summary_output = gr.Textbox(
        label="Simulation Summary",
        lines=8,
    )

    route_output = gr.Textbox(
        label="Warehouse Route",
        lines=28,
    )

    run_button.click(
        fn=run_demo,
        inputs=[
            agent_type,
            grid_size,
            obstacle_density,
            orders,
            dynamic_obstacles,
            seed,
        ],
        outputs=[
            summary_output,
            route_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()