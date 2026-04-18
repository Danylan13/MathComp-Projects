# Energy Load Forecasting Results

## Benchmark Summary

| Method | Test RMSE (MW) | Test MAPE |
| --- | ---: | ---: |
| Lag-24 baseline | 38.853 | 8.909% |
| Lag-1 baseline | 14.616 | 2.424% |
| Ordinary least squares | 18.724 | 4.270% |
| Ridge regression | 15.314 | 3.388% |

## Notes

- Best regularization strength selected on time-series validation: `alpha = 10.0`
- Main drivers in the fitted standardized model: `temperature_c`, `is_evening_peak`, `lag_1`
- Artifacts: `outputs/forecast_diagnostics.png`, `outputs/alpha_search.csv`, `outputs/benchmark_summary.csv`
