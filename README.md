# Smart Warehouse Routing Agent

A reinforcement learning warehouse-routing simulation built with Gymnasium, Stable-Baselines3 PPO, and Gradio.

This project trains a PPO agent to complete multiple pickup/dropoff orders in a fixed 8x8 warehouse layout while avoiding shelves. The learned PPO policy is evaluated against classical and heuristic baselines including A*, Greedy, and Random agents.

## Project Overview

The environment represents a simplified warehouse floor plan:

- 8x8 grid layout
- Static shelf obstacles
- 3 pickup orders
- 1 dropoff station
- Fixed warehouse layout across episodes
- Discrete movement actions: up, right, down, left

The agent starts at the dropoff station, travels to each pickup location, returns each item to the dropoff station, and repeats until all orders are completed.

## Why This Project Exists

This project started as a basic reinforcement learning grid-navigation task and was adapted into a more practical warehouse-routing simulation.

The goal was to make the task feel closer to a real operational problem: a warehouse robot completing multiple fulfillment steps while avoiding obstacles and comparing learned behavior against traditional routing strategies.

## Agents Compared

### PPO Agent

The PPO agent is trained using Stable-Baselines3. It learns a policy from environment observations and reward feedback.

### A* Baseline

A* is the classical planning baseline. It uses direct access to the current warehouse map and computes a path to the active target.

### Greedy Baseline

The Greedy agent moves toward the current target based on Manhattan distance.

### Random Baseline

The Random agent samples actions without strategy.

## Current Scope

This version uses a fixed warehouse layout.

That means PPO is evaluated on the same stable warehouse floor plan it trains on. This is intentional for version one because the goal is to test whether PPO can learn the controlled multi-order routing task before adding layout generalization.

This project does not currently claim that PPO generalizes across unseen warehouse layouts.

A future extension would train across multiple generated warehouse layouts and evaluate on held-out layouts to test generalization.

## Design Decision: Fixed Layout vs. Random Layouts

An earlier version generated new warehouse layouts between episodes. That made the PPO evaluation much harder because the agent was being asked to generalize across layouts before it had learned the base routing task.

The current version scopes the problem to one fixed warehouse layout so the training and evaluation setup is aligned:

1. Train PPO on a stable 8x8 warehouse.
2. Evaluate PPO on the same routing task.
3. Compare PPO against A*, Greedy, and Random baselines.
4. Use the result as a controlled benchmark before expanding to randomized layouts.

This mirrors a common machine learning workflow:

- First, solve the controlled task.
- Then, test generalization under more variation.

## Environment Details

Observation space:

- Flattened normalized grid
- Agent position
- Current target position
- Carrying status
- Current order progress

Action space:

```
0 = up
1 = right
2 = down
3 = left
```

Reward structure:

- Small step penalty
- Reward for moving closer to the target
- Penalty for moving farther from the target
- Penalty for invalid moves/collisions
- Reward for pickup
- Reward for delivery
- Completion bonus when all orders are finished

## Installation

```bash
pip install -r requirements.txt
```

## Train PPO

```bash
python train.py
```

Default training settings:

- grid_size = 8
- orders = 3
- obstacle_density = 0.18
- timesteps = 1,000,000
- seed = 42

You can also override settings manually:

```bash
python train.py --timesteps 500000 --seed 42
```

## Evaluate Agents

```bash
python evaluate.py
```

This evaluates PPO against:

- A*
- Greedy
- Random

Reported metrics include:

- Success rate
- Average reward
- Average completed orders
- Average collisions
- Average steps

## Run the Gradio Demo

```bash
python app.py
```

The app visualizes the warehouse route as an animation and compares agent behavior interactively.

## Example Project Framing

This project demonstrates:

- Custom Gymnasium environment design
- Reinforcement learning with PPO
- Reward shaping
- Baseline comparison
- Debugging train/evaluation mismatch
- Controlled evaluation design
- Interactive Gradio visualization

## Limitations

This version uses one fixed warehouse layout. PPO performance should be interpreted as success on the trained warehouse task, not proof of generalization across all possible warehouse layouts.

A* may still outperform PPO in path efficiency because it directly plans using the map. PPO is included as a learned policy comparison, not as a guaranteed replacement for classical planning.

## Production Considerations

This project is a controlled reinforcement learning prototype, not a production warehouse robotics system.

In a real warehouse, the physical layout would usually remain mostly stable. Shelves, aisles, packing stations, and restricted zones do not typically change every episode. What changes more often is the operational metadata: active orders, item locations, robot state, order priority, temporary blocked aisles, congestion, timing constraints, and task queues.

For that reason, a production-style version would likely keep a known warehouse map and dynamically inject task metadata into the routing or decision system. That way, the model or planner is operating against a stable physical space while the tasks change around it, rather than retraining from scratch every time new orders or operational conditions change.

A practical architecture might combine:

- Classical path planning, such as A* or Dijkstra, for reliable low-level routing on a known map
- Rules or safety constraints for invalid moves, blocked zones, restricted areas, and collision prevention
- ML/RL components for higher-level optimization, such as task sequencing, congestion-aware routing, batching, or multi-agent coordination

The current project focuses on the first controlled milestone: proving that PPO can learn a fixed 8x8 multi-order routing task and comparing that learned policy against A*, Greedy, and Random baselines.

A future production-oriented extension would add dynamic order metadata, held-out layout evaluation, route-efficiency metrics, and more realistic operational constraints.

## Future Improvements

The current version focuses on a fixed 8x8 warehouse layout with three fixed pickup/dropoff orders. This keeps the first reinforcement learning milestone controlled and easier to evaluate.

A more production-oriented next step would not be to randomize the entire warehouse layout immediately. In a real warehouse, the physical map usually remains mostly stable. The dynamic part is the operational metadata: active orders, item locations, priorities, robot state, blocked aisles, congestion, timing constraints, and task queues.

Future improvements could include:

- **Dynamic order metadata:** keep the same warehouse map but vary item targets, order sequence, task priority, and dropoff requirements.
- **Live task injection:** allow the environment to receive new tasks without retraining the model from scratch each time.
- **Route-efficiency metrics:** track total steps, excess distance compared to A*, collision rate, invalid moves, and completion time.
- **Temporary blocked zones:** add controlled blocked aisles or congestion after the fixed-layout task is stable.
- **Hybrid planning architecture:** use A* or Dijkstra for reliable low-level routing and ML/RL for higher-level decisions like task sequencing, batching, or congestion-aware routing.
- **Held-out layout evaluation:** later, train across multiple seeded layouts and evaluate on unseen layouts to test true generalization.
- **Multi-agent extension:** simulate multiple warehouse agents and study collision avoidance, task allocation, and throughput.

## Summary

This project trains a PPO agent to solve a fixed 8x8 warehouse routing task with three pickup/dropoff orders. The agent is evaluated against A*, Greedy, and Random baselines to compare learned behavior with classical and heuristic routing approaches.
