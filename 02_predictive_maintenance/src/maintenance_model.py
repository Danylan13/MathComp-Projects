from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sensor_windows.csv"


@dataclass
class MetricSet:
    precision: float
    recall: float
    f1: float
    confusion: tuple[int, int, int, int]


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    raw = np.genfromtxt(DATA_PATH, delimiter=",", names=True, dtype=None, encoding="utf-8")
    feature_names = [
        "temp_mean",
        "vibration_rms",
        "pressure_mean",
        "current_draw",
        "acoustic_rms",
        "throughput_tph",
    ]
    x = np.column_stack([raw[name].astype(float) for name in feature_names])
    y = raw["failure_in_24h"].astype(int)
    window_ids = raw["window_id"].astype(int)
    return x, y, window_ids, feature_names


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) == 0, 1.0, train.std(axis=0))
    return (train - mean) / std, (test - mean) / std, mean, std


def mahalanobis_scores(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    inverse_covariance = np.linalg.pinv(covariance)
    deltas = x - mean
    return np.einsum("ij,jk,ik->i", deltas, inverse_covariance, deltas)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


def fit_logistic_regression(
    x: np.ndarray, y: np.ndarray, learning_rate: float = 0.08, epochs: int = 4000, l2: float = 0.3
) -> tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1])
    bias = 0.0

    for _ in range(epochs):
        logits = x @ weights + bias
        probabilities = sigmoid(logits)
        error = probabilities - y

        grad_w = (x.T @ error) / len(x) + l2 * weights / len(x)
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias


def classify_logistic(x: np.ndarray, weights: np.ndarray, bias: float, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    probabilities = sigmoid(x @ weights + bias)
    return (probabilities >= threshold).astype(int), probabilities


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricSet:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MetricSet(precision, recall, f1, (tp, fp, tn, fn))


def main() -> None:
    x, y, window_ids, feature_names = load_dataset()
    split_index = 32
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    test_window_ids = window_ids[split_index:]

    x_train_scaled, x_test_scaled, _, _ = standardize(x_train, x_test)
    healthy_train = x_train_scaled[y_train == 0]
    healthy_mean = healthy_train.mean(axis=0)
    healthy_covariance = np.cov(healthy_train, rowvar=False)

    train_health_scores = mahalanobis_scores(healthy_train, healthy_mean, healthy_covariance)
    test_health_scores = mahalanobis_scores(x_test_scaled, healthy_mean, healthy_covariance)
    mahalanobis_threshold = float(np.quantile(train_health_scores, 0.95))
    mahalanobis_pred = (test_health_scores >= mahalanobis_threshold).astype(int)

    weights, bias = fit_logistic_regression(x_train_scaled, y_train)
    logistic_pred, logistic_prob = classify_logistic(x_test_scaled, weights, bias)

    mahalanobis_metrics = compute_metrics(y_test, mahalanobis_pred)
    logistic_metrics = compute_metrics(y_test, logistic_pred)

    ranked_features = sorted(zip(feature_names, weights), key=lambda item: abs(item[1]), reverse=True)
    top_windows = np.argsort(logistic_prob)[::-1][:6]

    print("Predictive Maintenance")
    print("-" * 72)
    print(f"Train windows: {len(x_train)} | Test windows: {len(x_test)}")
    print(f"Mahalanobis threshold: {mahalanobis_threshold:.3f}")
    print()
    print("Mahalanobis detector:")
    print(
        f"  precision={mahalanobis_metrics.precision:.3f} "
        f"recall={mahalanobis_metrics.recall:.3f} "
        f"f1={mahalanobis_metrics.f1:.3f}"
    )
    print(f"  confusion(tp, fp, tn, fn)={mahalanobis_metrics.confusion}")
    print()
    print("Logistic regression:")
    print(
        f"  precision={logistic_metrics.precision:.3f} "
        f"recall={logistic_metrics.recall:.3f} "
        f"f1={logistic_metrics.f1:.3f}"
    )
    print(f"  confusion(tp, fp, tn, fn)={logistic_metrics.confusion}")
    print()
    print("Strongest logistic coefficients:")
    for name, value in ranked_features[:5]:
        print(f"  {name:<16} {value:>9.3f}")
    print()
    print("Highest-risk test windows:")
    for idx in top_windows:
        print(
            f"  window={test_window_ids[idx]:>2} | probability={logistic_prob[idx]:.3f} "
            f"| actual={y_test[idx]} | mahalanobis={test_health_scores[idx]:.3f}"
        )


if __name__ == "__main__":
    main()
