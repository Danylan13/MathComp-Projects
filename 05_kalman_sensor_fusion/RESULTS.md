# Kalman Sensor Fusion Results

## Executive Summary

Kalman filtering improves the raw GPS trajectory, and RTS smoothing improves it further. The project now also reports how error changes across several GPS-dropout schedules, which makes the robustness story much clearer.

## Trajectory Error Summary

| Method | Position RMSE |
| --- | ---: |
| Raw GPS | 1.793 |
| Kalman filter | 1.579 |
| Kalman filter with GPS dropout | 1.607 |
| RTS smoother | 1.206 |

## Dropout Robustness

| Scenario | Position RMSE |
| --- | ---: |
| baseline_filter | 1.579 |
| gps_dropout_every_3 | 1.578 |
| gps_dropout_every_5 | 1.607 |
| gps_dropout_every_8 | 1.621 |

## What Improved

- Filtering improves raw GPS by `11.96%`
- RTS smoothing improves the filtered estimate by `23.62%`
- Even under repeated GPS dropout, the filter remains close to the nominal filtered error level

## Additional Analysis

- `dropout_sensitivity.png` visualizes the stability of the filter as position updates become less frequent
- `state_estimates.csv` stores filtered, smoothed, and dropout-stress trajectories side by side
- `robustness_summary.csv` provides a compact machine-readable version of the stress sweep

## Why This Matters

This project now demonstrates a key engineering point: sensor fusion is valuable precisely when measurements are noisy or intermittent, not only when the signal is clean.

## Limitations

- The motion model is constant-velocity and therefore intentionally simple
- The GPS dropout patterns are periodic rather than event-driven
- The current setup does not model bias drift in the IMU channels

## Artifacts

- `outputs/trajectory_comparison.png`
- `outputs/dropout_sensitivity.png`
- `outputs/state_estimates.csv`
- `outputs/robustness_summary.csv`
