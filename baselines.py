from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from warehouse_env import ACTION_TO_DELTA, SHELF, SmartWarehouseEnv


Position = Tuple[int, int]


def random_action(env: SmartWarehouseEnv) -> int:
    return int(env.action_space.sample())


def greedy_action(env: SmartWarehouseEnv) -> int:
    target = env._current_target()

    best_action = None
    best_distance = float("inf")

    for action, delta in ACTION_TO_DELTA.items():
        candidate = (
            env.agent_pos[0] + delta[0],
            env.agent_pos[1] + delta[1],
        )

        if env._is_walkable(candidate):
            distance = abs(candidate[0] - target[0]) + abs(candidate[1] - target[1])

            if distance < best_distance:
                best_distance = distance
                best_action = action

    if best_action is None:
        return int(env.action_space.sample())

    return int(best_action)


def astar_action(env: SmartWarehouseEnv) -> int:
    path = astar_path(env, env.agent_pos, env._current_target())

    if path and len(path) >= 2:
        next_pos = path[1]

        delta = (
            next_pos[0] - env.agent_pos[0],
            next_pos[1] - env.agent_pos[1],
        )

        for action, action_delta in ACTION_TO_DELTA.items():
            if delta == action_delta:
                return int(action)

    return greedy_action(env)


def astar_path(
    env: SmartWarehouseEnv,
    start: Position,
    goal: Position,
) -> Optional[List[Position]]:
    def heuristic(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = []
    heapq.heappush(open_heap, (0, start))

    came_from: Dict[Position, Optional[Position]] = {start: None}
    g_score: Dict[Position, float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            return reconstruct_path(came_from, current)

        for delta in ACTION_TO_DELTA.values():
            neighbor = (
                current[0] + delta[0],
                current[1] + delta[1],
            )

            if not is_walkable_for_search(env, neighbor):
                continue

            tentative_g = g_score[current] + 1.0

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f_score = tentative_g + heuristic(neighbor, goal)

                heapq.heappush(open_heap, (f_score, neighbor))

    return None


def is_walkable_for_search(env: SmartWarehouseEnv, pos: Position) -> bool:
    r, c = pos

    if r < 0 or r >= env.grid_size or c < 0 or c >= env.grid_size:
        return False

    if env.grid[pos] == SHELF:
        return False

    if pos in env.dynamic_blocks:
        return False

    return True


def reconstruct_path(
    came_from: Dict[Position, Optional[Position]],
    current: Position,
) -> List[Position]:
    path = [current]

    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[index]
        path.append(current)

    return list(reversed(path))