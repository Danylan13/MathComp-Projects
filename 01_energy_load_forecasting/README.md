# Energy Load Forecasting

## Problem

Forecast hourly power demand from weather measurements, industrial activity, calendar effects, and autoregressive lags.

## Data

File: `data/energy_load.csv`

Columns:

- `timestamp`
- `temperature_c`
- `humidity_pct`
- `wind_kph`
- `is_holiday`
- `industrial_index`
- `load_mw`

## Method

The pipeline builds a feature matrix with:

- current weather and operating variables
- cyclic hour and day-of-week encodings
- lag-1 and lag-24 demand features

The model is a ridge regressor evaluated on a walk-forward split. Results are compared to a naive day-ahead baseline.

## Run

```bash
python 01_energy_load_forecasting/src/forecasting_pipeline.py
```

## Output

The script reports:

- baseline vs model RMSE and MAPE
- the most influential coefficients
- the final 24-hour forecast window
- saved diagnostic plot in `outputs/forecast_diagnostics.png`
