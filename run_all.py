from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ProjectRun:
    name: str
    script: Path
    expected_outputs: list[Path]


PROJECTS = [
    ProjectRun(
        name="Energy Load Forecasting",
        script=ROOT / "01_energy_load_forecasting" / "src" / "forecasting_pipeline.py",
        expected_outputs=[
            ROOT / "01_energy_load_forecasting" / "outputs" / "forecast_diagnostics.png",
            ROOT / "01_energy_load_forecasting" / "outputs" / "alpha_search.csv",
            ROOT / "01_energy_load_forecasting" / "outputs" / "benchmark_summary.csv",
        ],
    ),
    ProjectRun(
        name="Predictive Maintenance",
        script=ROOT / "02_predictive_maintenance" / "src" / "maintenance_model.py",
        expected_outputs=[
            ROOT / "02_predictive_maintenance" / "outputs" / "maintenance_diagnostics.png",
            ROOT / "02_predictive_maintenance" / "outputs" / "threshold_search.csv",
            ROOT / "02_predictive_maintenance" / "outputs" / "test_alerts.csv",
        ],
    ),
    ProjectRun(
        name="Network Reliability",
        script=ROOT / "03_network_reliability" / "src" / "network_analysis.py",
        expected_outputs=[
            ROOT / "03_network_reliability" / "outputs" / "network_paths.png",
            ROOT / "03_network_reliability" / "outputs" / "route_summary.csv",
        ],
    ),
    ProjectRun(
        name="Portfolio Optimization",
        script=ROOT / "04_portfolio_optimization" / "src" / "portfolio_analysis.py",
        expected_outputs=[
            ROOT / "04_portfolio_optimization" / "outputs" / "portfolio_backtest.png",
            ROOT / "04_portfolio_optimization" / "outputs" / "efficient_frontier.csv",
            ROOT / "04_portfolio_optimization" / "outputs" / "stress_test.csv",
        ],
    ),
    ProjectRun(
        name="Kalman Sensor Fusion",
        script=ROOT / "05_kalman_sensor_fusion" / "src" / "kalman_fusion.py",
        expected_outputs=[
            ROOT / "05_kalman_sensor_fusion" / "outputs" / "trajectory_comparison.png",
            ROOT / "05_kalman_sensor_fusion" / "outputs" / "state_estimates.csv",
        ],
    ),
]


def run_project(project: ProjectRun) -> tuple[bool, float]:
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(project.script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start

    print(f"\n=== {project.name} ===")
    print(result.stdout.strip())
    if result.stderr.strip():
        print("\n[stderr]")
        print(result.stderr.strip())

    outputs_ok = all(path.exists() for path in project.expected_outputs)
    success = result.returncode == 0 and outputs_ok
    print(f"\nstatus={'OK' if success else 'FAILED'} | runtime={elapsed:.2f}s")

    if not outputs_ok:
        missing = [str(path.relative_to(ROOT)) for path in project.expected_outputs if not path.exists()]
        print("missing outputs:")
        for path in missing:
            print(f"  - {path}")

    return success, elapsed


def main() -> int:
    print("Running all projects and validating generated artifacts...")
    results: list[tuple[str, bool, float]] = []

    for project in PROJECTS:
        success, elapsed = run_project(project)
        results.append((project.name, success, elapsed))

    print("\n=== Summary ===")
    overall_success = True
    for name, success, elapsed in results:
        overall_success &= success
        print(f"{name:<24} {'OK' if success else 'FAILED':<7} {elapsed:>6.2f}s")

    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
