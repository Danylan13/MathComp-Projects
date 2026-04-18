# Kalman Sensor Fusion

![Trajectory comparison](outputs/trajectory_comparison.png)

## Problem

Recover the trajectory of a moving object from noisy GPS position measurements and IMU-derived velocity estimates.

## Data

File: `data/tracking_measurements.csv`

Columns:

- `true_x`, `true_y`
- `gps_x`, `gps_y`
- `imu_vx`, `imu_vy`

## Method

A linear Kalman filter is used with a constant-velocity state model. The state vector contains position and velocity in two dimensions, and the measurement vector combines GPS and IMU observations. A Rauch-Tung-Striebel smoother is then applied on top of the filtered states to improve the final trajectory estimate.

The project compares:

- raw GPS position error
- filtered trajectory error

## Key Results

- Kalman filtering reduces RMSE from `1.793` to `1.579`
- RTS smoothing reduces RMSE further to `1.206`
- The smoother improves on the filtered trajectory by `23.62%`

## Benchmarks

| Method | Position RMSE |
| --- | ---: |
| Raw GPS | 1.793 |
| Kalman filter | 1.579 |
| RTS smoother | 1.206 |

See [RESULTS.md](RESULTS.md) and `outputs/state_estimates.csv` for the trajectory summary.

## Run

```bash
python 05_kalman_sensor_fusion/src/kalman_fusion.py
```

## Output

The script reports:

- raw and filtered RMSE
- smoothed RMSE and smoother gain over the filter
- state estimates at the start and end of the trajectory
- improvement from sensor fusion
- saved trajectory plot in `outputs/trajectory_comparison.png`
