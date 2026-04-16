from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionReport:
    anomaly_indices: np.ndarray
    z_scores: np.ndarray
    reconstruction_errors: np.ndarray


def simulate_sensor_stream(seed: int = 21, steps: int = 240) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(steps)

    temperature = 70 + 2.5 * np.sin(time / 20) + rng.normal(0, 0.4, size=steps)
    pressure = 100 + 1.8 * np.sin(time / 18 + 0.3) + rng.normal(0, 0.5, size=steps)
    vibration = 4 + 0.2 * np.sin(time / 7) + rng.normal(0, 0.15, size=steps)

    stream = np.column_stack([temperature, pressure, vibration])

    anomaly_slice = slice(150, 160)
    stream[anomaly_slice, 0] += 8.0
    stream[anomaly_slice, 1] -= 6.0
    stream[anomaly_slice, 2] += np.linspace(1.0, 2.5, anomaly_slice.stop - anomaly_slice.start)
    return stream


def rolling_z_score(stream: np.ndarray, window: int = 25) -> np.ndarray:
    scores = np.zeros(len(stream))
    for idx in range(window, len(stream)):
        history = stream[idx - window : idx]
        mean = history.mean(axis=0)
        std = np.where(history.std(axis=0) == 0, 1, history.std(axis=0))
        z = np.abs((stream[idx] - mean) / std)
        scores[idx] = float(z.max())
    return scores


def pca_reconstruction_error(stream: np.ndarray, n_components: int = 2) -> np.ndarray:
    centered = stream - stream.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_vectors = eigenvectors[:, np.argsort(eigenvalues)[::-1][:n_components]]
    projected = centered @ principal_vectors
    reconstructed = projected @ principal_vectors.T
    error = centered - reconstructed
    return np.linalg.norm(error, axis=1)


def detect_anomalies(stream: np.ndarray) -> DetectionReport:
    z_scores = rolling_z_score(stream)
    reconstruction_errors = pca_reconstruction_error(stream)

    z_threshold = float(z_scores.mean() + 2.8 * z_scores.std())
    reconstruction_threshold = float(
        reconstruction_errors.mean() + 2.5 * reconstruction_errors.std()
    )

    mask = (z_scores > z_threshold) | (reconstruction_errors > reconstruction_threshold)
    return DetectionReport(
        anomaly_indices=np.flatnonzero(mask),
        z_scores=z_scores,
        reconstruction_errors=reconstruction_errors,
    )


def main() -> None:
    stream = simulate_sensor_stream()
    report = detect_anomalies(stream)

    print("Sensor Anomaly Detection")
    print("-" * 60)
    print(f"Detected anomalies: {len(report.anomaly_indices)}")
    print("Top suspicious timestamps:")

    combined_score = report.z_scores + 3.0 * report.reconstruction_errors
    strongest = np.argsort(combined_score)[-8:][::-1]
    for idx in strongest:
        print(
            f"  t={idx:03d} | z-score={report.z_scores[idx]:.2f} "
            f"| reconstruction_error={report.reconstruction_errors[idx]:.3f}"
        )


if __name__ == "__main__":
    main()
