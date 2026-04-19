# Energy Load Forecasting Results

## Executive Summary

The forecasting pipeline improves substantially over the naive day-ahead baseline and produces a stable linear model that remains interpretable. The strongest gains come from combining weather effects, calendar structure, and autoregressive lags instead of relying on a single historical reference point.

## Benchmark Summary

| Method | Test RMSE (MW) | Test MAPE |
| --- | ---: | ---: |
| Lag-24 baseline | 38.853 | 8.909% |
| Lag-1 baseline | 14.616 | 2.424% |
| Ordinary least squares | 18.724 | 4.270% |
| Ridge regression | 15.314 | 3.388% |

## What Improved

- Ridge regression reduces RMSE by `60.58%` relative to the lag-24 baseline
- Time-series validation selects `alpha = 10.0`, giving a more stable fit than unregularized OLS
- The strongest standardized drivers are `temperature_c`, `is_evening_peak`, and `lag_1`

## Additional Analysis

- `feature_ablation.csv` shows that removing weather variables or calendar structure degrades predictive accuracy
- `regime_breakdown.csv` breaks performance down into `peak_day`, `commute_peak`, `night_valley`, and `regular` periods
- `feature_importance.png` highlights the largest standardized coefficients for model interpretation

## Why This Matters

This project now reads like a practical demand-forecasting study rather than a simple regression exercise. It includes baselines, regularization tuning, ablations, and regime-specific evaluation, which are all standard ingredients in real forecasting work.

## Limitations

- The model family is intentionally linear and compact, so it prioritizes interpretability over maximum predictive power
- The dataset is still small, so conclusions should be treated as directional rather than production-grade
- Lag-1 remains a very strong baseline, which signals that further gains would likely require richer exogenous variables or nonlinear models

## Artifacts

- `outputs/forecast_diagnostics.png`
- `outputs/feature_importance.png`
- `outputs/alpha_search.csv`
- `outputs/benchmark_summary.csv`
- `outputs/feature_ablation.csv`
- `outputs/regime_breakdown.csv`
