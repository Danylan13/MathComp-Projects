# Predictive Maintenance Results

## Executive Summary

The supervised classifier beats the unsupervised Mahalanobis detector on the test split while preserving full recall. The project also now makes the operational trade-off visible through business cost and a synthetic sensor-drift stress test.

## Benchmark Summary

| Method | Precision | Recall | F1 | Business Cost |
| --- | ---: | ---: | ---: | ---: |
| Mahalanobis detector | 0.250 | 1.000 | 0.400 | 12.0 |
| Logistic regression | 0.400 | 1.000 | 0.571 | 6.0 |
| Logistic with sensor drift | 0.308 | 1.000 | 0.471 | 9.0 |

## What Improved

- Logistic regression improves F1 from `0.400` to `0.571`
- The selected decision threshold is `0.30`
- Operational false-alarm cost is cut in half relative to the unsupervised baseline

## Additional Analysis

- `maintenance_tradeoffs.png` shows how precision, recall, and business cost move as the classification threshold changes
- `drift_robustness.csv` captures how the model behaves when several sensor channels experience calibration drift
- `model_comparison.csv` stores a compact benchmark table that can be reused outside the notebook or app context

## Why This Matters

Predictive maintenance is rarely judged by F1 alone. Teams care about missed failures, unnecessary interventions, and sensitivity to sensor degradation. This project now surfaces all three.

## Limitations

- The classifier is still trained on a small synthetic dataset
- Full recall is preserved partly because the chosen threshold is aggressive, which keeps false alarms nontrivial
- The drift experiment is synthetic rather than learned from historical recalibration events

## Artifacts

- `outputs/maintenance_diagnostics.png`
- `outputs/maintenance_tradeoffs.png`
- `outputs/threshold_search.csv`
- `outputs/test_alerts.csv`
- `outputs/model_comparison.csv`
- `outputs/drift_robustness.csv`
