from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EDGE_PATH = Path(__file__).resolve().parents[1] / "data" / "network_edges.csv"
DEMAND_PATH = Path(__file__).resolve().parents[1] / "data" / "demand_scenarios.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


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
    scenario: str = ""
    origin: str = ""
    destination: str = ""


NODE_POSITIONS = {
    "Kyiv": (0.0, 0.4),
    "Lviv": (-2.4, 1.0),
    "Odessa": (-0.8, -1.6),
    "Kharkiv": (2.6, 0.8),
    "Dnipro": (1.3, -0.4),
    "Poltava": (1.8, 0.3),
    "Zaporizhzhia": (1.5, -1.2),
    "Warsaw": (-3.2, 2.0),
    "Berlin": (-4.7, 2.3),
}


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


def best_latency_path(
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
    feasible_reports.sort(key=lambda report: (report.latency_ms, report.score))
    return feasible_reports[0]


def simulate_path_availability(report: PathReport, trials: int = 5000, seed: int = 17) -> float:
    rng = np.random.default_rng(seed)
    outcomes = rng.random(trials)
    return float(np.mean(outcomes < report.reliability))


def save_route_summary(reports: list[PathReport]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "route_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scenario",
                "origin",
                "destination",
                "path",
                "latency_ms",
                "bottleneck_mbps",
                "analytic_reliability",
                "simulated_availability",
                "latency_only_path",
                "latency_only_reliability",
            ]
        )
        for index, report in enumerate(reports):
            latency_baseline = getattr(report, "latency_baseline", None)
            writer.writerow(
                [
                    report.scenario,
                    report.origin,
                    report.destination,
                    " -> ".join(report.nodes),
                    f"{report.latency_ms:.3f}",
                    f"{report.bottleneck_mbps:.3f}",
                    f"{report.reliability:.6f}",
                    f"{simulate_path_availability(report, seed=17 + index):.6f}",
                    "" if latency_baseline is None else " -> ".join(latency_baseline.nodes),
                    "" if latency_baseline is None else f"{latency_baseline.reliability:.6f}",
                ]
            )


def save_network_plot(base_graph: dict[str, list[Edge]], highlighted_paths: list[PathReport]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    drawn_edges: set[tuple[str, str]] = set()

    for source, edges in base_graph.items():
        x0, y0 = NODE_POSITIONS[source]
        for edge in edges:
            key = tuple(sorted((source, edge.target)))
            if key in drawn_edges:
                continue
            drawn_edges.add(key)
            x1, y1 = NODE_POSITIONS[edge.target]
            ax.plot([x0, x1], [y0, y1], color="#c7c7c7", linewidth=1.5, zorder=1)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    for index, report in enumerate(highlighted_paths):
        color = colors[index % len(colors)]
        for source, target in zip(report.nodes[:-1], report.nodes[1:]):
            x0, y0 = NODE_POSITIONS[source]
            x1, y1 = NODE_POSITIONS[target]
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=3.2, zorder=2)

    for node, (x, y) in NODE_POSITIONS.items():
        ax.scatter(x, y, s=180, color="#1f3c88", zorder=3)
        ax.text(x, y + 0.14, node, ha="center", va="bottom", fontsize=9)

    ax.set_title("Network Topology with Selected Scenario Paths")
    ax.axis("off")
    fig.savefig(OUTPUT_DIR / "network_paths.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base_graph = load_graph()
    demands = load_demands()
    chosen_paths: list[PathReport] = []

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

            report.scenario = scenario
            report.origin = origin
            report.destination = destination
            report.latency_baseline = best_latency_path(
                scenario_graph, origin, destination, required, max_latency
            )
            chosen_paths.append(report)
            simulated_availability = simulate_path_availability(report, seed=17 + len(chosen_paths))
            print(
                f"  {origin} -> {destination} | path={' -> '.join(report.nodes)} | "
                f"latency={report.latency_ms:.1f} ms | bottleneck={report.bottleneck_mbps:.0f} Mbps | "
                f"reliability={report.reliability:.4f} | simulated_availability={simulated_availability:.4f} | "
                f"latency_only_reliability={report.latency_baseline.reliability:.4f}"
            )
        print()

    if chosen_paths:
        save_route_summary(chosen_paths)
        save_network_plot(base_graph, chosen_paths)
        print(f"Saved plot: {OUTPUT_DIR / 'network_paths.png'}")
        print(f"Saved table: {OUTPUT_DIR / 'route_summary.csv'}")


if __name__ == "__main__":
    main()
