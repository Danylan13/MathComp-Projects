# Kalman Sensor Fusion Results

## Trajectory Error Summary

| Method | Position RMSE |
| --- | ---: |
| Raw GPS | 1.793 |
| Kalman filter | 1.579 |
| RTS smoother | 1.206 |

## Notes

- Filtering improves raw GPS by `11.96%`
- The RTS smoother improves the filtered trajectory by `23.62%`
- Artifacts: `outputs/trajectory_comparison.png`, `outputs/state_estimates.csv`
