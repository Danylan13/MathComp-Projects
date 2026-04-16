# Sensor Anomaly Detection

## Goal

Detect unusual behavior in a simulated industrial sensor stream using two signals:

- rolling z-score
- PCA reconstruction error

## Why It Matters

This mirrors a practical monitoring problem from embedded and industrial systems: identifying faulty or drifting sensors before a failure becomes critical.

## Concepts

- multivariate time series
- rolling statistics
- dimensionality reduction
- anomaly scoring
- threshold-based alerting

## Run

```bash
python 02_sensor_anomaly_detection/src/sensor_monitor.py
```

## Output

The script reports:

- number of detected anomalies
- top suspicious timestamps
- score details for the strongest alerts
