from __future__ import annotations

from pathlib import Path
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def energy_summary() -> tuple[str, float, float]:
    rows = read_csv_dicts(ROOT / "01_energy_load_forecasting" / "outputs" / "benchmark_summary.csv")
    lag24 = next(row for row in rows if row["method"] == "lag_24_baseline")
    ridge = next(row for row in rows if row["method"] == "ridge_regression")
    baseline = float(lag24["rmse_mw"])
    model = float(ridge["rmse_mw"])
    return ("Energy forecasting", model, (baseline - model) / baseline * 100.0)


def maintenance_summary() -> tuple[str, float, float]:
    rows = read_csv_dicts(ROOT / "02_predictive_maintenance" / "outputs" / "model_comparison.csv")
    mahalanobis = next(row for row in rows if row["model"] == "mahalanobis")
    logistic = next(row for row in rows if row["model"] == "logistic_regression")
    baseline = float(mahalanobis["f1"])
    model = float(logistic["f1"])
    return ("Predictive maintenance", model, (model - baseline) / max(baseline, 1e-9) * 100.0)


def network_summary() -> tuple[str, float, float]:
    rows = read_csv_dicts(ROOT / "03_network_reliability" / "outputs" / "route_summary.csv")
    selected = np.mean([float(row["analytic_reliability"]) for row in rows])
    latency_only = np.mean([float(row["latency_only_reliability"]) for row in rows])
    return ("Network reliability", float(selected), (selected - latency_only) * 100.0)


def portfolio_summary() -> tuple[str, float, float]:
    rows = read_csv_dicts(ROOT / "04_portfolio_optimization" / "outputs" / "strategy_comparison.csv")
    optimized = next(row for row in rows if row["strategy"] == "optimized_static")
    equal = next(row for row in rows if row["strategy"] == "equal_weight")
    optimized_dd = abs(float(optimized["max_drawdown"]))
    equal_dd = abs(float(equal["max_drawdown"]))
    reduction = (equal_dd - optimized_dd) / max(equal_dd, 1e-9) * 100.0
    return ("Portfolio optimization", float(optimized["sharpe_ratio"]), reduction)


def kalman_summary() -> tuple[str, float, float]:
    rows = read_csv_dicts(ROOT / "05_kalman_sensor_fusion" / "outputs" / "robustness_summary.csv")
    baseline = next(row for row in rows if row["scenario"] == "baseline_filter")
    dropout = next(row for row in rows if row["scenario"] == "gps_dropout_every_5")
    baseline_rmse = float(baseline["position_rmse"])
    dropout_rmse = float(dropout["position_rmse"])
    robustness = (dropout_rmse - baseline_rmse) / max(baseline_rmse, 1e-9) * 100.0
    return ("Kalman sensor fusion", baseline_rmse, robustness)


def save_metrics_table(rows: list[tuple[str, float, float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "overview_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["project", "headline_metric", "improvement_or_stress_pct"])
        for project, metric, delta in rows:
            writer.writerow([project, f"{metric:.6f}", f"{delta:.6f}"])


def save_dashboard(rows: list[tuple[str, float, float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    projects = [row[0] for row in rows]
    metrics = np.array([row[1] for row in rows], dtype=float)
    deltas = np.array([row[2] for row in rows], dtype=float)
    x = np.arange(len(projects))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    bars = axes[0].bar(x, metrics, color=["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"])
    axes[0].set_title("Project Headline Metrics")
    axes[0].set_ylabel("Metric value")
    axes[0].set_xticks(x, projects, rotation=15, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, metrics):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")

    colors = np.where(deltas >= 0.0, "#117a65", "#c0392b")
    bars = axes[1].bar(x, deltas, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Relative Gain or Stress Penalty")
    axes[1].set_ylabel("Percent")
    axes[1].set_xticks(x, projects, rotation=15, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, deltas):
        offset = 0.8 if value >= 0 else -1.2
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.1f}%", ha="center")

    fig.savefig(OUTPUT_DIR / "overview_dashboard.png", dpi=160)
    plt.close(fig)


def main() -> None:
    rows = [
        energy_summary(),
        maintenance_summary(),
        network_summary(),
        portfolio_summary(),
        kalman_summary(),
    ]
    save_metrics_table(rows)
    save_dashboard(rows)
    print("Repository overview")
    print("-" * 72)
    for project, metric, delta in rows:
        print(f"{project:<24} metric={metric:.4f} delta={delta:.2f}%")
    print()
    print(f"Saved outputs: {OUTPUT_DIR / 'overview_metrics.csv'}, {OUTPUT_DIR / 'overview_dashboard.png'}")


if __name__ == "__main__":
    main()
