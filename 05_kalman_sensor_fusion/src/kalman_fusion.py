from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tracking_measurements.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def load_measurements(data_path: Path) -> np.ndarray:
    return np.genfromtxt(data_path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def run_kalman_filter(measurements: np.ndarray, gps_dropout_every: int = 0) -> np.ndarray:
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
    predicted_states = []
    predicted_covariances = []
    filtered_covariances = []
    for row in measurements:
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_noise
        predicted_states.append(state.copy())
        predicted_covariances.append(covariance.copy())

        if gps_dropout_every and len(estimates) % gps_dropout_every == 0:
            observation_step = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            measurement = np.array([float(row["imu_vx"]), float(row["imu_vy"])])
            measurement_noise_step = np.diag([0.18, 0.18])
        else:
            observation_step = observation
            measurement = np.array(
                [
                    float(row["gps_x"]),
                    float(row["gps_y"]),
                    float(row["imu_vx"]),
                    float(row["imu_vy"]),
                ]
            )
            measurement_noise_step = measurement_noise

        innovation = measurement - observation_step @ state
        innovation_covariance = observation_step @ covariance @ observation_step.T + measurement_noise_step
        kalman_gain = covariance @ observation_step.T @ np.linalg.inv(innovation_covariance)
        state = state + kalman_gain @ innovation
        covariance = (np.eye(4) - kalman_gain @ observation_step) @ covariance
        estimates.append(state.copy())
        filtered_covariances.append(covariance.copy())

    return (
        np.asarray(estimates),
        np.asarray(predicted_states),
        np.asarray(predicted_covariances),
        np.asarray(filtered_covariances),
        transition,
    )


def run_rts_smoother(
    filtered_states: np.ndarray,
    predicted_states: np.ndarray,
    predicted_covariances: np.ndarray,
    filtered_covariances: np.ndarray,
    transition: np.ndarray,
) -> np.ndarray:
    smoothed_states = filtered_states.copy()
    smoothed_covariances = filtered_covariances.copy()
    for index in range(len(filtered_states) - 2, -1, -1):
        gain = filtered_covariances[index] @ transition.T @ np.linalg.inv(predicted_covariances[index + 1])
        smoothed_states[index] = filtered_states[index] + gain @ (
            smoothed_states[index + 1] - predicted_states[index + 1]
        )
        smoothed_covariances[index] = filtered_covariances[index] + gain @ (
            smoothed_covariances[index + 1] - predicted_covariances[index + 1]
        ) @ gain.T
    return smoothed_states


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((reference - estimate) ** 2, axis=1))))


def save_main_plot(
    true_positions: np.ndarray,
    gps_positions: np.ndarray,
    filtered_positions: np.ndarray,
    smoothed_positions: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(true_positions[:, 0], true_positions[:, 1], label="true path", linewidth=2.4, color="#1f3c88")
    axes[0].scatter(gps_positions[:, 0], gps_positions[:, 1], label="gps", s=18, color="#c0392b", alpha=0.6)
    axes[0].plot(filtered_positions[:, 0], filtered_positions[:, 1], label="kalman", linewidth=2.0, color="#117a65")
    axes[0].plot(smoothed_positions[:, 0], smoothed_positions[:, 1], label="smoother", linewidth=2.0, color="#8e44ad")
    axes[0].set_title("Trajectory Reconstruction")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    raw_error = np.linalg.norm(true_positions - gps_positions, axis=1)
    filtered_error = np.linalg.norm(true_positions - filtered_positions, axis=1)
    smoothed_error = np.linalg.norm(true_positions - smoothed_positions, axis=1)
    axes[1].plot(raw_error, label="gps error", color="#c0392b", linewidth=2.0)
    axes[1].plot(filtered_error, label="kalman error", color="#117a65", linewidth=2.0)
    axes[1].plot(smoothed_error, label="smoother error", color="#8e44ad", linewidth=2.0)
    axes[1].set_title("Position Error Over Time")
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("Euclidean error")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.savefig(output_dir / "trajectory_comparison.png", dpi=160)
    plt.close(fig)


def save_state_estimates(
    measurements: np.ndarray,
    filtered_positions: np.ndarray,
    smoothed_positions: np.ndarray,
    dropout_positions: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "state_estimates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "filtered_x", "filtered_y", "smoothed_x", "smoothed_y", "dropout_x", "dropout_y"])
        for index in range(len(measurements)):
            writer.writerow(
                [
                    f"{float(measurements['t'][index]):.1f}",
                    f"{filtered_positions[index, 0]:.6f}",
                    f"{filtered_positions[index, 1]:.6f}",
                    f"{smoothed_positions[index, 0]:.6f}",
                    f"{smoothed_positions[index, 1]:.6f}",
                    f"{dropout_positions[index, 0]:.6f}",
                    f"{dropout_positions[index, 1]:.6f}",
                ]
            )


def save_robustness_summary(
    measurements: np.ndarray,
    true_positions: np.ndarray,
    output_dir: Path,
    dropout_steps: list[int],
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = [("baseline_filter", rmse(true_positions, run_kalman_filter(measurements)[0][:, :2]))]
    for every in dropout_steps:
        estimates, _, _, _, _ = run_kalman_filter(measurements, gps_dropout_every=every)
        rows.append((f"gps_dropout_every_{every}", rmse(true_positions, estimates[:, :2])))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "robustness_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "position_rmse"])
        for name, value in rows:
            writer.writerow([name, f"{value:.6f}"])
    return rows


def save_dropout_plot(rows: list[tuple[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row[0].replace("gps_dropout_every_", "dropout/") for row in rows]
    values = np.asarray([row[1] for row in rows])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(labels, values, marker="o", linewidth=2.2, color="#1f77b4")
    ax.set_title("Kalman Filter Sensitivity to GPS Dropout")
    ax.set_ylabel("Position RMSE")
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "dropout_sensitivity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalman sensor fusion and smoothing with dropout robustness checks.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to tracking measurements CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated outputs.")
    parser.add_argument("--dropout-every", type=int, default=5, help="Default GPS dropout period used in the main stress test.")
    parser.add_argument(
        "--dropout-sweep",
        type=int,
        nargs="*",
        default=[3, 5, 8],
        help="Dropout intervals used for robustness summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measurements = load_measurements(args.data_path)
    true_positions = np.column_stack([measurements["true_x"], measurements["true_y"]]).astype(float)
    gps_positions = np.column_stack([measurements["gps_x"], measurements["gps_y"]]).astype(float)
    estimates, predicted_states, predicted_covariances, filtered_covariances, transition = run_kalman_filter(measurements)
    dropout_estimates, _, _, _, _ = run_kalman_filter(measurements, gps_dropout_every=args.dropout_every)
    smoothed_states = run_rts_smoother(estimates, predicted_states, predicted_covariances, filtered_covariances, transition)

    filtered_positions = estimates[:, :2]
    dropout_positions = dropout_estimates[:, :2]
    smoothed_positions = smoothed_states[:, :2]
    raw_rmse = rmse(true_positions, gps_positions)
    filtered_rmse = rmse(true_positions, filtered_positions)
    dropout_rmse = rmse(true_positions, dropout_positions)
    smoothed_rmse = rmse(true_positions, smoothed_positions)
    improvement = (raw_rmse - filtered_rmse) / raw_rmse * 100.0
    smoother_gain = (filtered_rmse - smoothed_rmse) / filtered_rmse * 100.0

    save_state_estimates(measurements, filtered_positions, smoothed_positions, dropout_positions, args.output_dir)
    save_main_plot(true_positions, gps_positions, filtered_positions, smoothed_positions, args.output_dir)
    robustness_rows = save_robustness_summary(measurements, true_positions, args.output_dir, args.dropout_sweep)
    save_dropout_plot(robustness_rows, args.output_dir)

    print("Kalman Sensor Fusion")
    print("-" * 72)
    print(f"Raw GPS position RMSE: {raw_rmse:.3f}")
    print(f"Filtered position RMSE: {filtered_rmse:.3f}")
    print(f"Filter RMSE with GPS dropout: {dropout_rmse:.3f}")
    print(f"Smoothed position RMSE: {smoothed_rmse:.3f}")
    print(f"Relative improvement: {improvement:.2f}%")
    print(f"Smoother gain over filter: {smoother_gain:.2f}%")
    print()
    print("Dropout robustness sweep:")
    for scenario, value in robustness_rows:
        print(f"  {scenario:<20} rmse={value:.3f}")
    print()
    print("State snapshots:")
    for idx in [0, 1, 2, len(estimates) - 3, len(estimates) - 2, len(estimates) - 1]:
        state = estimates[idx]
        print(
            f"  t={measurements['t'][idx]:>4.1f} | x={state[0]:7.3f} y={state[1]:7.3f} "
            f"| vx={state[2]:6.3f} vy={state[3]:6.3f}"
        )
    print()
    print(
        f"Saved plots: {args.output_dir / 'trajectory_comparison.png'}, {args.output_dir / 'dropout_sensitivity.png'}"
    )
    print(
        f"Saved tables: {args.output_dir / 'state_estimates.csv'}, {args.output_dir / 'robustness_summary.csv'}"
    )


if __name__ == "__main__":
    main()
