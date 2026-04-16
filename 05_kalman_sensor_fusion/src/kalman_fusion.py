from __future__ import annotations

from pathlib import Path

import numpy as np


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tracking_measurements.csv"


def load_measurements() -> np.ndarray:
    return np.genfromtxt(DATA_PATH, delimiter=",", names=True, dtype=None, encoding="utf-8")


def run_kalman_filter(measurements: np.ndarray) -> np.ndarray:
    dt = float(measurements["t"][1] - measurements["t"][0])
    transition = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    observation = np.eye(4)
    process_noise = np.diag([0.20, 0.20, 0.08, 0.08])
    measurement_noise = np.diag([2.6, 2.6, 0.18, 0.18])

    state = np.array(
        [
            float(measurements["gps_x"][0]),
            float(measurements["gps_y"][0]),
            float(measurements["imu_vx"][0]),
            float(measurements["imu_vy"][0]),
        ]
    )
    covariance = np.diag([5.0, 5.0, 1.0, 1.0])

    estimates = []
    for row in measurements:
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_noise

        measurement = np.array(
            [
                float(row["gps_x"]),
                float(row["gps_y"]),
                float(row["imu_vx"]),
                float(row["imu_vy"]),
            ]
        )
        innovation = measurement - observation @ state
        innovation_covariance = observation @ covariance @ observation.T + measurement_noise
        kalman_gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)

        state = state + kalman_gain @ innovation
        covariance = (np.eye(4) - kalman_gain @ observation) @ covariance
        estimates.append(state.copy())

    return np.asarray(estimates)


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((reference - estimate) ** 2, axis=1))))


def main() -> None:
    measurements = load_measurements()
    true_positions = np.column_stack([measurements["true_x"], measurements["true_y"]]).astype(float)
    gps_positions = np.column_stack([measurements["gps_x"], measurements["gps_y"]]).astype(float)
    estimates = run_kalman_filter(measurements)
    filtered_positions = estimates[:, :2]

    raw_rmse = rmse(true_positions, gps_positions)
    filtered_rmse = rmse(true_positions, filtered_positions)
    improvement = (raw_rmse - filtered_rmse) / raw_rmse * 100.0

    print("Kalman Sensor Fusion")
    print("-" * 72)
    print(f"Raw GPS position RMSE: {raw_rmse:.3f}")
    print(f"Filtered position RMSE: {filtered_rmse:.3f}")
    print(f"Relative improvement: {improvement:.2f}%")
    print()
    print("State snapshots:")
    for idx in [0, 1, 2, len(estimates) - 3, len(estimates) - 2, len(estimates) - 1]:
        state = estimates[idx]
        print(
            f"  t={measurements['t'][idx]:>4.1f} | x={state[0]:7.3f} y={state[1]:7.3f} "
            f"| vx={state[2]:6.3f} vy={state[3]:6.3f}"
        )


if __name__ == "__main__":
    main()
