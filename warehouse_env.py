from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


Position = Tuple[int, int]

EMPTY   = 0
SHELF   = 1
AGENT   = 2
PICKUP  = 3
DROPOFF = 4

MAX_GRID_VALUE = DROPOFF

ACTION_TO_DELTA: Dict[int, Position] = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}


@dataclass
class StepInfo:
    completed_orders: int
    total_orders: int
    collisions: int
    invalid_moves: int
    picked_up: bool
    delivered: bool
    target: Position


class SmartWarehouseEnv(gym.Env):
    """
    Warehouse fulfillment routing environment.

    The shelf layout and order locations are fixed at construction time and
    do not change between episodes. reset() only resets episode state.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size: int = 8,
        obstacle_density: float = 0.18,
        n_orders: int = 3,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if grid_size < 5:
            raise ValueError("grid_size must be at least 5.")

        if not 0 <= obstacle_density < 0.45:
            raise ValueError("obstacle_density must be between 0 and 0.45.")

        if n_orders < 1:
            raise ValueError("n_orders must be at least 1.")

        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.n_orders = n_orders
        self.max_steps = max_steps or grid_size * grid_size * n_orders * 5
        self.rng = np.random.default_rng(seed)

        obs_len = (grid_size * grid_size) + 6

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_len,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(4)

        self.dropoff_pos: Position = (0, 0)
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.orders: List[Position] = []

        # Episode state (reset each episode)
        self.agent_pos: Position = self.dropoff_pos
        self.current_order_idx = 0
        self.carrying = False
        self.steps = 0
        self.collisions = 0
        self.invalid_moves = 0
        self.visited: set[Position] = set()

        self._generate_valid_layout()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.steps = 0
        self.collisions = 0
        self.invalid_moves = 0
        self.current_order_idx = 0
        self.carrying = False
        self.agent_pos = self.dropoff_pos
        self.visited = {self.agent_pos}

        return self._get_observation(), self._get_info(False, False)

    def step(self, action: int):
        self.steps += 1
        picked_up = False
        delivered = False

        old_distance = self._manhattan(self.agent_pos, self._current_target())

        row_delta, col_delta = ACTION_TO_DELTA[int(action)]
        candidate = (
            self.agent_pos[0] + row_delta,
            self.agent_pos[1] + col_delta,
        )

        reward = -0.002

        if not self._is_walkable(candidate):
            reward -= 0.1
            self.collisions += 1
            self.invalid_moves += 1
        else:
            self.agent_pos = candidate
            self.visited.add(self.agent_pos)

            new_distance = self._manhattan(self.agent_pos, self._current_target())

            if new_distance < old_distance:
                reward += 0.03
            elif new_distance > old_distance:
                reward -= 0.03

        if (
            not self.carrying
            and self.current_order_idx < len(self.orders)
            and self.agent_pos == self.orders[self.current_order_idx]
        ):
            self.carrying = True
            picked_up = True
            reward += 1.0

        if self.carrying and self.agent_pos == self.dropoff_pos:
            self.carrying = False
            delivered = True
            self.current_order_idx += 1
            reward += 2.0

            if self.current_order_idx >= len(self.orders):
                reward += 10.0

        terminated = self.current_order_idx >= len(self.orders)
        truncated = self.steps >= self.max_steps

        if truncated and not terminated:
            reward -= 1.0

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info(picked_up, delivered),
        )

    def render(self):
        display = self._display_grid()

        symbols = {
            EMPTY:   ".",
            SHELF:   "X",
            AGENT:   "A",
            PICKUP:  "P",
            DROPOFF: "D",
        }

        rows = []
        for r in range(self.grid_size):
            rows.append(
                " ".join(symbols[int(display[r, c])] for c in range(self.grid_size))
            )

        return "\n".join(rows)

    def _generate_valid_layout(self):
        attempts = 0

        while True:
            attempts += 1

            if attempts > 1_000:
                raise RuntimeError("Could not generate a valid warehouse layout.")

            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            self.grid[self.dropoff_pos] = DROPOFF

            all_cells = [
                (r, c)
                for r in range(self.grid_size)
                for c in range(self.grid_size)
                if (r, c) != self.dropoff_pos
            ]

            n_shelves = int(len(all_cells) * self.obstacle_density)

            shelf_indices = self.rng.choice(
                len(all_cells),
                size=n_shelves,
                replace=False,
            )

            shelves = {all_cells[i] for i in shelf_indices}

            for pos in shelves:
                self.grid[pos] = SHELF

            open_cells = [cell for cell in all_cells if cell not in shelves]

            if len(open_cells) < self.n_orders:
                continue

            order_indices = self.rng.choice(
                len(open_cells),
                size=self.n_orders,
                replace=False,
            )

            self.orders = [open_cells[i] for i in order_indices]

            reachable = self._reachable_cells(self.dropoff_pos, shelves)

            if all(order in reachable for order in self.orders):
                break

    def _get_observation(self) -> np.ndarray:
        display = self._display_grid()

        flat_grid = (display.flatten() / MAX_GRID_VALUE).astype(np.float32)

        target = self._current_target()

        meta = np.array(
            [
                self.agent_pos[0] / (self.grid_size - 1),
                self.agent_pos[1] / (self.grid_size - 1),
                target[0] / (self.grid_size - 1),
                target[1] / (self.grid_size - 1),
                1.0 if self.carrying else 0.0,
                self.current_order_idx / max(1, self.n_orders),
            ],
            dtype=np.float32,
        )

        return np.concatenate([flat_grid, meta]).astype(np.float32)

    def _display_grid(self) -> np.ndarray:
        display = self.grid.copy()

        for idx, pickup in enumerate(self.orders):
            if idx >= self.current_order_idx:
                display[pickup] = PICKUP

        display[self.dropoff_pos] = DROPOFF
        display[self.agent_pos] = AGENT

        return display

    def _current_target(self) -> Position:
        if self.current_order_idx >= len(self.orders):
            return self.dropoff_pos

        if self.carrying:
            return self.dropoff_pos

        return self.orders[self.current_order_idx]

    def _get_info(self, picked_up: bool, delivered: bool) -> dict:
        return StepInfo(
            completed_orders=self.current_order_idx,
            total_orders=self.n_orders,
            collisions=self.collisions,
            invalid_moves=self.invalid_moves,
            picked_up=picked_up,
            delivered=delivered,
            target=self._current_target(),
        ).__dict__

    def _is_walkable(self, pos: Position) -> bool:
        r, c = pos

        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return False

        if self.grid[pos] == SHELF:
            return False

        return True

    def _reachable_cells(self, start: Position, shelves: set[Position]) -> set[Position]:
        queue = deque([start])
        seen = {start}

        while queue:
            pos = queue.popleft()

            for delta in ACTION_TO_DELTA.values():
                nxt = (pos[0] + delta[0], pos[1] + delta[1])
                r, c = nxt

                if (
                    0 <= r < self.grid_size
                    and 0 <= c < self.grid_size
                    and nxt not in shelves
                    and nxt not in seen
                ):
                    seen.add(nxt)
                    queue.append(nxt)

        return seen

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
