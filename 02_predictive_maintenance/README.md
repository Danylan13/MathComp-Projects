# Predictive Maintenance

![Maintenance diagnostics](outputs/maintenance_diagnostics.png)

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

The classifier threshold is not fixed by hand: it is selected on a validation slice to maximize F1 score before final evaluation on the test windows.

## Key Results

- Logistic regression outperforms the Mahalanobis detector on F1 score: `0.571` vs `0.400`
- The selected logistic threshold is `0.30`
- The model preserves full recall on the test split while reducing false alarms relative to the unsupervised detector

## Benchmarks

| Method | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Mahalanobis detector | 0.250 | 1.000 | 0.400 |
| Logistic regression | 0.400 | 1.000 | 0.571 |

See [RESULTS.md](RESULTS.md), `outputs/threshold_search.csv`, and `outputs/test_alerts.csv` for the detailed comparison.

## Run

```bash
python 02_predictive_maintenance/src/maintenance_model.py
```

## Output

The script reports:

- precision, recall, and F1 score
- selected logistic threshold and threshold search table
- confusion matrices for both methods
- the highest-risk windows in the test segment
- saved diagnostic plot in `outputs/maintenance_diagnostics.png`
- includes a threshold-sweep view of test-set F1 behavior
