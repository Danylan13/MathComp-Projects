from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingResult:
    weights: np.ndarray
    bias: float
    losses: list[float]


def make_dataset(seed: int = 7, samples: int = 180) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    area = rng.normal(65, 18, size=samples)
    rooms = rng.integers(1, 5, size=samples)
    age = rng.integers(0, 40, size=samples)
    distance_to_metro = rng.normal(1.8, 0.7, size=samples)
    features = np.column_stack([area, rooms, age, distance_to_metro])

    noise = rng.normal(0, 12000, size=samples)
    price = (
        45000
        + 3100 * area
        + 17000 * rooms
        - 900 * age
        - 14000 * distance_to_metro
        + noise
    )
    return features, price


def train_test_split(
    features: np.ndarray, target: np.ndarray, ratio: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cutoff = int(len(features) * ratio)
    return (
        features[:cutoff],
        features[cutoff:],
        target[:cutoff],
        target[cutoff:],
    )


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std == 0, 1, std)
    return (train - mean) / std, (test - mean) / std, mean, std


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mse(y_true, y_pred))


def fit_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.06,
    epochs: int = 400,
) -> TrainingResult:
    samples, dimensions = x.shape
    weights = np.zeros(dimensions)
    bias = 0.0
    losses: list[float] = []

    for _ in range(epochs):
        predictions = x @ weights + bias
        error = predictions - y

        grad_w = (2 / samples) * (x.T @ error)
        grad_b = 2 * float(error.mean())

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        losses.append(mse(y, predictions))

    return TrainingResult(weights=weights, bias=bias, losses=losses)


def fit_closed_form(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    x_augmented = np.column_stack([x, np.ones(len(x))])
    solution = np.linalg.pinv(x_augmented.T @ x_augmented) @ x_augmented.T @ y
    return solution[:-1], float(solution[-1])


def predict(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return x @ weights + bias


def main() -> None:
    features, target = make_dataset()
    x_train, x_test, y_train, y_test = train_test_split(features, target)
    x_train_scaled, x_test_scaled, _, _ = standardize(x_train, x_test)

    gd_model = fit_gradient_descent(x_train_scaled, y_train)
    cf_weights, cf_bias = fit_closed_form(x_train_scaled, y_train)

    gd_train_pred = predict(x_train_scaled, gd_model.weights, gd_model.bias)
    gd_test_pred = predict(x_test_scaled, gd_model.weights, gd_model.bias)
    cf_test_pred = predict(x_test_scaled, cf_weights, cf_bias)

    print("Optimization Lab")
    print("-" * 60)
    print(f"Gradient Descent train RMSE: {rmse(y_train, gd_train_pred):.2f}")
    print(f"Gradient Descent test RMSE:  {rmse(y_test, gd_test_pred):.2f}")
    print(f"Closed Form test RMSE:       {rmse(y_test, cf_test_pred):.2f}")
    print()
    print("Gradient Descent weights:")
    for index, value in enumerate(gd_model.weights, start=1):
        print(f"  w{index}: {value:,.2f}")
    print(f"  bias: {gd_model.bias:,.2f}")
    print()
    print("First 5 losses:")
    for loss in gd_model.losses[:5]:
        print(f"  {loss:,.2f}")
    print("Last 5 losses:")
    for loss in gd_model.losses[-5:]:
        print(f"  {loss:,.2f}")


if __name__ == "__main__":
    main()
