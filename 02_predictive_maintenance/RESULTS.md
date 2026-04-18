# Predictive Maintenance Results

## Benchmark Summary

| Method | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Mahalanobis detector | 0.250 | 1.000 | 0.400 |
| Logistic regression | 0.400 | 1.000 | 0.571 |

## Notes

- Selected logistic threshold: `0.30`
- Strongest supervised signals: `pressure_mean`, `throughput_tph`, `vibration_rms`
- Artifacts: `outputs/maintenance_diagnostics.png`, `outputs/threshold_search.csv`, `outputs/test_alerts.csv`
