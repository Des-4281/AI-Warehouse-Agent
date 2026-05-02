from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import List

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from baselines import astar_action, greedy_action, random_action
from warehouse_env import (
    AGENT, DROPOFF, EMPTY, PICKUP, SHELF,
    SmartWarehouseEnv,
)


MODEL_PATH = Path("models/ppo_warehouse.zip")

CELL_COLORS = {
    EMPTY:    "#e8e8e8",
    SHELF:    "#4a4a4a",
    AGENT:    "#2196F3",
    PICKUP:   "#4CAF50",
    DROPOFF:  "#FF5722",
}

MAX_FRAMES = 150


def render_frame(env: SmartWarehouseEnv, step: int, total_reward: float) -> Image.Image:
    display = env._display_grid()
    gs = env.grid_size
    cell_px = max(40, 400 // gs)
    fig_in = gs * cell_px / 100

    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=100)
    ax.set_xlim(0, gs)
    ax.set_ylim(0, gs)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#1e1e1e")

    font_size = max(6, int(cell_px * 0.28))

    for r in range(gs):
        for c in range(gs):
            val = int(display[r, c])
            color = CELL_COLORS.get(val, "#ffffff")
            rect = plt.Rectangle(
                [c, gs - r - 1], 1, 1,
                color=color, ec="#1e1e1e", lw=1.0,
            )
            ax.add_patch(rect)

            label = {AGENT: "A", PICKUP: "P", DROPOFF: "D"}.get(val)
            if label:
                ax.text(
                    c + 0.5, gs - r - 0.5, label,
                    ha="center", va="center",
                    fontsize=font_size, fontweight="bold", color="white",
                )

    ax.set_title(
        f"Step {step}  |  Reward {total_reward:+.1f}",
        fontsize=9, color="white", pad=4,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def run_demo(
    agent_type: str,
    grid_size: int,
    obstacle_density: float,
    orders: int,
    seed: int,
):
    env = SmartWarehouseEnv(
        grid_size=int(grid_size),
        obstacle_density=float(obstacle_density),
        n_orders=int(orders),
        seed=int(seed),
    )

    obs, _ = env.reset()

    model = None

    if agent_type == "PPO":
        if not MODEL_PATH.exists():
            return None, "No trained PPO model found. Run `python train.py` first."

        try:
            model = PPO.load(str(MODEL_PATH))
        except Exception as exc:
            return None, f"Could not load PPO model: {exc}"

        if model.observation_space.shape != env.observation_space.shape:
            trained_size = int((model.observation_space.shape[0] - 6) ** 0.5)
            return None, (
                f"Grid size mismatch: model was trained on grid_size={trained_size} "
                f"but current grid_size={grid_size}. "
                f"Set the slider to {trained_size} or retrain."
            )

    frames: List[Image.Image] = []
    total_reward = 0.0
    done = False
    step = 0

    frames.append(render_frame(env, step, total_reward))

    while not done:
        if agent_type == "Random":
            action = random_action(env)
        elif agent_type == "Greedy":
            action = greedy_action(env)
        elif agent_type == "A*":
            action = astar_action(env)
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1

        frames.append(render_frame(env, step, total_reward))

    # Subsample if episode is very long so the GIF stays responsive
    if len(frames) > MAX_FRAMES:
        indices = np.linspace(0, len(frames) - 1, MAX_FRAMES, dtype=int)
        frames = [frames[i] for i in indices]

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    frames[0].save(
        tmp.name,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=120,
    )

    summary = (
        f"Agent:            {agent_type}\n"
        f"Success:          {info['completed_orders'] == info['total_orders']}\n"
        f"Orders completed: {info['completed_orders']} / {info['total_orders']}\n"
        f"Steps taken:      {env.steps}\n"
        f"Collisions:       {info['collisions']}\n"
        f"Invalid moves:    {info['invalid_moves']}\n"
        f"Total reward:     {total_reward:.2f}"
    )

    return tmp.name, summary


with gr.Blocks(title="Smart Warehouse Routing Agent") as demo:
    gr.Markdown(
        """
# Smart Warehouse Routing Agent

Compare routing strategies for a simulated warehouse robot completing pickup/dropoff tasks.

**Legend:** `A` = Agent (blue) &nbsp; `P` = Pickup (green) &nbsp; `D` = Dropoff (orange) &nbsp; dark = Shelf
"""
    )

    PPO_GRID = 8
    PPO_ORDERS = 3

    with gr.Row():
        agent_type = gr.Dropdown(
            choices=["A*", "Greedy", "Random", "PPO"],
            value="A*",
            label="Agent",
        )
        grid_size = gr.Slider(minimum=5, maximum=15, value=8, step=1, label="Warehouse Size")
        obstacle_density = gr.Slider(minimum=0.05, maximum=0.40, value=0.18, step=0.01, label="Obstacle Density")
        orders = gr.Slider(minimum=1, maximum=6, value=3, step=1, label="Number of Orders")

    with gr.Row():
        seed = gr.Number(value=42, precision=0, label="Random Seed")

    ppo_note = gr.Markdown(visible=False)

    def on_agent_change(agent):
        if agent == "PPO":
            return (
                gr.update(value=PPO_GRID, interactive=False),
                gr.update(value=PPO_ORDERS, interactive=False),
                gr.update(visible=True, value=(
                    f"_PPO is locked to its training settings: grid size {PPO_GRID}, {PPO_ORDERS} orders._"
                )),
            )
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(visible=False),
        )

    agent_type.change(
        fn=on_agent_change,
        inputs=agent_type,
        outputs=[grid_size, orders, ppo_note],
    )

    run_button = gr.Button("Run Simulation", variant="primary")

    with gr.Row():
        gif_output = gr.Image(label="Warehouse Animation", type="filepath")
        summary_output = gr.Textbox(label="Result Summary", lines=9)

    run_button.click(
        fn=run_demo,
        inputs=[agent_type, grid_size, obstacle_density, orders, seed],
        outputs=[gif_output, summary_output],
    )


if __name__ == "__main__":
    demo.launch()
