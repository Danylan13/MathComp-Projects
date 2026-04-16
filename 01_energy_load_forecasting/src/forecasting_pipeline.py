from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "energy_load.csv"


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


def predict(x: np.ndarray, model: RidgeModel) -> np.ndarray:
    return x @ model.weights + model.bias


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def main() -> None:
    raw = load_dataset()
    x, y, feature_names, labels = build_design_matrix(raw)
    split_index = len(x) - 24

    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    test_labels = labels[split_index:]

    x_train_scaled, x_test_scaled = standardize(x_train, x_test)
    model = fit_ridge_regression(x_train_scaled, y_train, feature_names)

    baseline_pred = x_test[:, -1]
    model_pred = predict(x_test_scaled, model)

    ranked_coefficients = sorted(
        zip(model.feature_names, model.weights),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    print("Energy Load Forecasting")
    print("-" * 72)
    print(f"Train samples: {len(x_train)} | Test samples: {len(x_test)}")
    print(f"Naive lag-24 RMSE: {rmse(y_test, baseline_pred):.3f} MW")
    print(f"Ridge model RMSE:  {rmse(y_test, model_pred):.3f} MW")
    print(f"Naive lag-24 MAPE: {mape(y_test, baseline_pred):.3f}%")
    print(f"Ridge model MAPE:  {mape(y_test, model_pred):.3f}%")
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


if __name__ == "__main__":
    main()
