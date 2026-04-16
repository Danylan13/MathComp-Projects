from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


EDGE_PATH = Path(__file__).resolve().parents[1] / "data" / "network_edges.csv"
DEMAND_PATH = Path(__file__).resolve().parents[1] / "data" / "demand_scenarios.csv"


@dataclass
class Edge:
    target: str
    latency_ms: float
    failure_prob: float
    capacity_mbps: float


@dataclass
class PathReport:
    nodes: list[str]
    latency_ms: float
    bottleneck_mbps: float
    reliability: float
    score: float


def load_graph() -> dict[str, list[Edge]]:
    raw = np.genfromtxt(EDGE_PATH, delimiter=",", names=True, dtype=None, encoding="utf-8")
    graph: dict[str, list[Edge]] = {}

    for row in raw:
        source = str(row["source"])
        target = str(row["target"])
        latency = float(row["latency_ms"])
        failure = float(row["failure_prob"])
        capacity = float(row["capacity_mbps"])

        graph.setdefault(source, []).append(Edge(target, latency, failure, capacity))
        graph.setdefault(target, []).append(Edge(source, latency, failure, capacity))

    return graph


def load_demands() -> np.ndarray:
    return np.genfromtxt(DEMAND_PATH, delimiter=",", names=True, dtype=None, encoding="utf-8")


def apply_scenario(graph: dict[str, list[Edge]], scenario: str) -> dict[str, list[Edge]]:
    updated: dict[str, list[Edge]] = {
        node: [Edge(edge.target, edge.latency_ms, edge.failure_prob, edge.capacity_mbps) for edge in edges]
        for node, edges in graph.items()
    }

    if scenario == "maintenance":
        for node, target in [("Kyiv", "Dnipro"), ("Dnipro", "Kyiv")]:
            updated[node] = [edge for edge in updated[node] if edge.target != target]

    if scenario == "peak":
        for node, edges in updated.items():
            updated[node] = [
                Edge(edge.target, edge.latency_ms, edge.failure_prob, edge.capacity_mbps * 0.75)
                for edge in edges
            ]

    return updated


def enumerate_paths(
    graph: dict[str, list[Edge]],
    current: str,
    goal: str,
    visited: set[str],
    path: list[str],
    max_hops: int,
) -> list[list[str]]:
    if current == goal:
        return [path.copy()]
    if len(path) > max_hops:
        return []

    results: list[list[str]] = []
    for edge in graph.get(current, []):
        if edge.target in visited:
            continue
        visited.add(edge.target)
        path.append(edge.target)
        results.extend(enumerate_paths(graph, edge.target, goal, visited, path, max_hops))
        path.pop()
        visited.remove(edge.target)
    return results


def path_metrics(graph: dict[str, list[Edge]], nodes: list[str]) -> PathReport:
    latency = 0.0
    bottleneck = float("inf")
    reliability = 1.0

    for source, target in zip(nodes[:-1], nodes[1:]):
        edge = next(edge for edge in graph[source] if edge.target == target)
        latency += edge.latency_ms
        bottleneck = min(bottleneck, edge.capacity_mbps)
        reliability *= 1.0 - edge.failure_prob

    score = latency + 220.0 * (1.0 - reliability)
    return PathReport(nodes, latency, bottleneck, reliability, score)


def best_feasible_path(
    graph: dict[str, list[Edge]],
    origin: str,
    destination: str,
    required_mbps: float,
    max_latency_ms: float,
) -> PathReport | None:
    candidates = enumerate_paths(graph, origin, destination, {origin}, [origin], max_hops=6)
    feasible_reports = []

    for nodes in candidates:
        report = path_metrics(graph, nodes)
        if report.bottleneck_mbps >= required_mbps and report.latency_ms <= max_latency_ms:
            feasible_reports.append(report)

    if not feasible_reports:
        return None

    feasible_reports.sort(key=lambda report: (report.score, report.latency_ms, -report.bottleneck_mbps))
    return feasible_reports[0]


def main() -> None:
    base_graph = load_graph()
    demands = load_demands()

    print("Network Reliability Analysis")
    print("-" * 72)

    for scenario in ["baseline", "maintenance", "peak"]:
        scenario_graph = apply_scenario(base_graph, scenario)
        print(f"[Scenario: {scenario}]")
        scenario_demands = demands[demands["scenario"] == scenario]

        for demand in scenario_demands:
            origin = str(demand["origin"])
            destination = str(demand["destination"])
            required = float(demand["required_mbps"])
            max_latency = float(demand["max_latency_ms"])
            report = best_feasible_path(scenario_graph, origin, destination, required, max_latency)

            if report is None:
                print(
                    f"  {origin} -> {destination} | required={required:.0f} Mbps "
                    f"| max_latency={max_latency:.0f} ms | infeasible"
                )
                continue

            print(
                f"  {origin} -> {destination} | path={' -> '.join(report.nodes)} | "
                f"latency={report.latency_ms:.1f} ms | bottleneck={report.bottleneck_mbps:.0f} Mbps | "
                f"reliability={report.reliability:.4f}"
            )
        print()


if __name__ == "__main__":
    main()
