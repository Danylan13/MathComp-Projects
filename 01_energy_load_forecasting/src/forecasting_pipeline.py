from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "energy_load.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


@dataclass
class RidgeModel:
    weights: np.ndarray
    bias: float
    feature_names: list[str]


def load_dataset() -> np.ndarray:
    return np.genfromtxt(DATA_PATH, delimiter=",", names=True, dtype=None, encoding="utf-8")


def build_design_matrix(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    timestamps = [datetime.fromisoformat(value) for value in raw["timestamp"]]
    loads = raw["load_mw"].astype(float)

    rows: list[list[float]] = []
    targets: list[float] = []
    labels: list[str] = []
    feature_names = [
        "temperature_c",
        "humidity_pct",
        "wind_kph",
        "is_holiday",
        "industrial_index",
        "time_index",
        "sin_hour",
        "cos_hour",
        "sin_dayofweek",
        "cos_dayofweek",
        "is_morning_peak",
        "is_evening_peak",
        "lag_1",
        "lag_24",
    ]

    for idx in range(24, len(raw)):
        ts = timestamps[idx]
        hour_angle = 2 * np.pi * ts.hour / 24
        dow_angle = 2 * np.pi * ts.weekday() / 7
        rows.append(
            [
                float(raw["temperature_c"][idx]),
                float(raw["humidity_pct"][idx]),
                float(raw["wind_kph"][idx]),
                float(raw["is_holiday"][idx]),
                float(raw["industrial_index"][idx]),
                float(idx),
                float(np.sin(hour_angle)),
                float(np.cos(hour_angle)),
                float(np.sin(dow_angle)),
                float(np.cos(dow_angle)),
                float(7 <= ts.hour <= 10),
                float(18 <= ts.hour <= 21),
                float(loads[idx - 1]),
                float(loads[idx - 24]),
            ]
        )
        targets.append(float(loads[idx]))
        labels.append(ts.strftime("%Y-%m-%d %H:%M"))

    return np.asarray(rows), np.asarray(targets), feature_names, np.asarray(labels)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) == 0, 1.0, train.std(axis=0))
    return (train - mean) / std, (test - mean) / std


def fit_ridge_regression(
    x: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float = 0.4
) -> RidgeModel:
    x_augmented = np.column_stack([x, np.ones(len(x))])
    regularizer = np.eye(x_augmented.shape[1]) * alpha
    regularizer[-1, -1] = 0.0
    solution = np.linalg.solve(x_augmented.T @ x_augmented + regularizer, x_augmented.T @ y)
    return RidgeModel(weights=solution[:-1], bias=float(solution[-1]), feature_names=feature_names)


def fit_ordinary_least_squares(x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> RidgeModel:
    x_augmented = np.column_stack([x, np.ones(len(x))])
    solution = np.linalg.pinv(x_augmented) @ y
    return RidgeModel(weights=solution[:-1], bias=float(solution[-1]), feature_names=feature_names)


def predict(x: np.ndarray, model: RidgeModel) -> np.ndarray:
    return x @ model.weights + model.bias


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def tune_alpha(
    x_train: np.ndarray, y_train: np.ndarray, feature_names: list[str], candidates: list[float]
) -> tuple[float, list[tuple[float, float]]]:
    metrics: list[tuple[float, float]] = []
    validation_window = 16
    fold_starts = [
        len(x_train) - 3 * validation_window,
        len(x_train) - 2 * validation_window,
        len(x_train) - validation_window,
    ]

    for alpha in candidates:
        fold_scores = []
        for split_index in fold_starts:
            x_fit, x_val = x_train[:split_index], x_train[split_index : split_index + validation_window]
            y_fit, y_val = y_train[:split_index], y_train[split_index : split_index + validation_window]
            x_fit_scaled, x_val_scaled = standardize(x_fit, x_val)
            model = fit_ridge_regression(x_fit_scaled, y_fit, feature_names, alpha=alpha)
            predictions = predict(x_val_scaled, model)
            fold_scores.append(rmse(y_val, predictions))
        metrics.append((alpha, float(np.mean(fold_scores))))

    best_alpha, _ = min(metrics, key=lambda item: item[1])
    return best_alpha, metrics


def save_validation_metrics(metrics: list[tuple[float, float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "alpha_search.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha", "validation_rmse"])
        for alpha, value in metrics:
            writer.writerow([alpha, f"{value:.6f}"])


def save_benchmark_table(rows: list[tuple[str, float, float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "benchmark_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "rmse_mw", "mape_pct"])
        for name, rmse_value, mape_value in rows:
            writer.writerow([name, f"{rmse_value:.6f}", f"{mape_value:.6f}"])


def save_plots(labels: np.ndarray, y_true: np.ndarray, baseline: np.ndarray, forecast: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_axis = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(x_axis, y_true, label="actual", linewidth=2.2, color="#1f3c88")
    axes[0].plot(x_axis, baseline, label="lag24 baseline", linestyle="--", color="#c0392b")
    axes[0].plot(x_axis, forecast, label="ridge forecast", color="#117a65")
    axes[0].set_title("Final 24-Hour Forecast Window")
    axes[0].set_ylabel("Load (MW)")
    axes[0].set_xticks(x_axis[::3], labels[::3], rotation=30, ha="right")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    residual_baseline = y_true - baseline
    residual_model = y_true - forecast
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].plot(x_axis, residual_baseline, label="baseline residual", linestyle="--", color="#c0392b")
    axes[1].plot(x_axis, residual_model, label="model residual", color="#117a65")
    axes[1].set_title("Residual Comparison")
    axes[1].set_ylabel("Residual (MW)")
    axes[1].set_xticks(x_axis[::3], labels[::3], rotation=30, ha="right")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.savefig(OUTPUT_DIR / "forecast_diagnostics.png", dpi=160)
    plt.close(fig)


def main() -> None:
    raw = load_dataset()
    x, y, feature_names, labels = build_design_matrix(raw)
    split_index = len(x) - 24

    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    test_labels = labels[split_index:]

    x_train_scaled, x_test_scaled = standardize(x_train, x_test)
    alpha_candidates = [0.05, 0.1, 0.4, 1.0, 3.0, 10.0, 25.0]
    best_alpha, validation_metrics = tune_alpha(x_train, y_train, feature_names, alpha_candidates)
    save_validation_metrics(validation_metrics)
    model = fit_ridge_regression(x_train_scaled, y_train, feature_names, alpha=best_alpha)
    ols_model = fit_ordinary_least_squares(x_train_scaled, y_train, feature_names)

    lag24_pred = x_test[:, -1]
    lag1_pred = x_test[:, -2]
    ols_pred = predict(x_test_scaled, ols_model)
    baseline_pred = x_test[:, -1]
    model_pred = predict(x_test_scaled, model)
    benchmark_rows = [
        ("lag_24_baseline", rmse(y_test, lag24_pred), mape(y_test, lag24_pred)),
        ("lag_1_baseline", rmse(y_test, lag1_pred), mape(y_test, lag1_pred)),
        ("ordinary_least_squares", rmse(y_test, ols_pred), mape(y_test, ols_pred)),
        ("ridge_regression", rmse(y_test, model_pred), mape(y_test, model_pred)),
    ]
    save_benchmark_table(benchmark_rows)

    ranked_coefficients = sorted(
        zip(model.feature_names, model.weights),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    save_plots(test_labels, y_test, baseline_pred, model_pred)

    print("Energy Load Forecasting")
    print("-" * 72)
    print(f"Train samples: {len(x_train)} | Test samples: {len(x_test)}")
    print(f"Selected alpha: {best_alpha:.2f}")
    print(f"Naive lag-24 RMSE: {rmse(y_test, lag24_pred):.3f} MW")
    print(f"Lag-1 baseline RMSE: {rmse(y_test, lag1_pred):.3f} MW")
    print(f"OLS model RMSE:      {rmse(y_test, ols_pred):.3f} MW")
    print(f"Ridge model RMSE:    {rmse(y_test, model_pred):.3f} MW")
    print(f"Naive lag-24 MAPE: {mape(y_test, lag24_pred):.3f}%")
    print(f"Lag-1 baseline MAPE: {mape(y_test, lag1_pred):.3f}%")
    print(f"OLS model MAPE:      {mape(y_test, ols_pred):.3f}%")
    print(f"Ridge model MAPE:    {mape(y_test, model_pred):.3f}%")
    print()
    print("Most influential standardized coefficients:")
    for name, value in ranked_coefficients[:6]:
        print(f"  {name:<16} {value:>10.3f}")
    print()
    print("Final 24-hour forecast window:")
    for label, actual, baseline, forecast in zip(test_labels, y_test, baseline_pred, model_pred):
        print(
            f"  {label} | actual={actual:7.2f} | lag24={baseline:7.2f} "
            f"| ridge={forecast:7.2f}"
        )
    print()
    print(f"Saved plot: {OUTPUT_DIR / 'forecast_diagnostics.png'}")
    print(f"Saved tables: {OUTPUT_DIR / 'alpha_search.csv'}, {OUTPUT_DIR / 'benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
