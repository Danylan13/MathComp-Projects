from __future__ import annotations

import heapq
from dataclasses import dataclass


GridPoint = tuple[int, int]


@dataclass
class SearchResult:
    path: list[GridPoint]
    total_cost: int
    visited_nodes: int


def make_grid() -> tuple[int, int, set[GridPoint]]:
    rows, cols = 8, 10
    blocked = {
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3),
        (5, 4),
        (5, 5),
        (2, 7),
        (3, 7),
        (4, 7),
    }
    return rows, cols, blocked


def neighbors(node: GridPoint, rows: int, cols: int, blocked: set[GridPoint]) -> list[GridPoint]:
    row, col = node
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [
        candidate
        for candidate in candidates
        if 0 <= candidate[0] < rows and 0 <= candidate[1] < cols and candidate not in blocked
    ]


def reconstruct_path(came_from: dict[GridPoint, GridPoint | None], goal: GridPoint) -> list[GridPoint]:
    path: list[GridPoint] = []
    current: GridPoint | None = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    return list(reversed(path))


def dijkstra(
    start: GridPoint, goal: GridPoint, rows: int, cols: int, blocked: set[GridPoint]
) -> SearchResult:
    frontier: list[tuple[int, GridPoint]] = [(0, start)]
    came_from: dict[GridPoint, GridPoint | None] = {start: None}
    cost_so_far: dict[GridPoint, int] = {start: 0}
    visited_nodes = 0

    while frontier:
        current_cost, current = heapq.heappop(frontier)
        visited_nodes += 1

        if current == goal:
            return SearchResult(
                path=reconstruct_path(came_from, goal),
                total_cost=current_cost,
                visited_nodes=visited_nodes,
            )

        for neighbor in neighbors(current, rows, cols, blocked):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(frontier, (new_cost, neighbor))

    raise ValueError("No path found")


def manhattan(a: GridPoint, b: GridPoint) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(
    start: GridPoint, goal: GridPoint, rows: int, cols: int, blocked: set[GridPoint]
) -> SearchResult:
    frontier: list[tuple[int, int, GridPoint]] = [(0, manhattan(start, goal), start)]
    came_from: dict[GridPoint, GridPoint | None] = {start: None}
    cost_so_far: dict[GridPoint, int] = {start: 0}
    visited_nodes = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        visited_nodes += 1

        if current == goal:
            return SearchResult(
                path=reconstruct_path(came_from, goal),
                total_cost=cost_so_far[goal],
                visited_nodes=visited_nodes,
            )

        for neighbor in neighbors(current, rows, cols, blocked):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heuristic = manhattan(neighbor, goal)
                priority = new_cost + heuristic
                came_from[neighbor] = current
                heapq.heappush(frontier, (priority, heuristic, neighbor))

    raise ValueError("No path found")


def main() -> None:
    rows, cols, blocked = make_grid()
    start = (0, 0)
    goal = (7, 9)

    dijkstra_result = dijkstra(start, goal, rows, cols, blocked)
    astar_result = a_star(start, goal, rows, cols, blocked)

    print("Route Planning")
    print("-" * 60)
    print(
        f"Dijkstra: cost={dijkstra_result.total_cost}, "
        f"visited_nodes={dijkstra_result.visited_nodes}"
    )
    print(f"Path: {dijkstra_result.path}")
    print()
    print(
        f"A*:       cost={astar_result.total_cost}, "
        f"visited_nodes={astar_result.visited_nodes}"
    )
    print(f"Path: {astar_result.path}")


if __name__ == "__main__":
    main()
