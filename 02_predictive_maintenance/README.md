# Predictive Maintenance

## Problem

Estimate whether an industrial machine will fail within the next 24 hours from aggregated sensor windows.

## Data

File: `data/sensor_windows.csv`

Columns:

- `temp_mean`
- `vibration_rms`
- `pressure_mean`
- `current_draw`
- `acoustic_rms`
- `throughput_tph`
- `failure_in_24h`

## Method

Two detection strategies are implemented:

- Mahalanobis distance relative to healthy operating windows
- regularized logistic regression trained on labeled examples

This makes it possible to compare unsupervised health monitoring against a supervised classifier.

## Run

```bash
python 02_predictive_maintenance/src/maintenance_model.py
```

## Output

The script reports:

- precision, recall, and F1 score
- confusion matrices for both methods
- the highest-risk windows in the test segment
