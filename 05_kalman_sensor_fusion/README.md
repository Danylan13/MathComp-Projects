# Kalman Sensor Fusion

## Problem

Recover the trajectory of a moving object from noisy GPS position measurements and IMU-derived velocity estimates.

## Data

File: `data/tracking_measurements.csv`

Columns:

- `true_x`, `true_y`
- `gps_x`, `gps_y`
- `imu_vx`, `imu_vy`

## Method

A linear Kalman filter is used with a constant-velocity state model. The state vector contains position and velocity in two dimensions, and the measurement vector combines GPS and IMU observations.

The project compares:

- raw GPS position error
- filtered trajectory error

## Run

```bash
python 05_kalman_sensor_fusion/src/kalman_fusion.py
```

## Output

The script reports:

- raw and filtered RMSE
- state estimates at the start and end of the trajectory
- improvement from sensor fusion
