# Route Planning

## Goal

Plan a route in a city-like grid while avoiding blocked cells and minimizing travel cost.

The project compares:

- Dijkstra's algorithm
- A* search with a Manhattan heuristic

## Why It Matters

This is a compact example of graph algorithms applied to robotics, logistics, and navigation systems.

## Concepts

- weighted graphs
- shortest paths
- heuristic search
- path reconstruction
- computational trade-offs

## Run

```bash
python 03_route_planning/src/route_planner.py
```

## Output

The script prints:

- path cost
- number of explored nodes
- the final path for each algorithm
