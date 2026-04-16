# Applied Mathematics and Computer Engineering Projects

This repository contains five compact but data-driven projects built around realistic engineering analysis tasks. The focus is on mathematical modeling, numerical methods, signal and network analysis, and decision-making under uncertainty.

Each project includes:

- a concrete problem statement
- input data stored in the repository
- reproducible Python code
- quantitative evaluation
- generated output artifacts such as plots or diagnostics
- concise documentation of methods and assumptions

## Projects

| Project | Problem | Main Methods |
| --- | --- | --- |
| `01_energy_load_forecasting` | Forecast short-term energy demand from weather and industrial activity data | time-series features, ridge regression, walk-forward evaluation |
| `02_predictive_maintenance` | Predict equipment failures from aggregated sensor windows | Mahalanobis distance, logistic regression, classification metrics |
| `03_network_reliability` | Route telecom traffic through a network under latency, capacity, and outage constraints | graph search, constrained routing, reliability scoring |
| `04_portfolio_optimization` | Build and backtest a portfolio allocation from historical asset prices | return estimation, covariance modeling, grid-search optimization, tail risk |
| `05_kalman_sensor_fusion` | Estimate a moving object's state from noisy GPS and IMU measurements | linear state-space models, Kalman filtering, RMSE analysis |

## Repository Layout

```text
MathComp-Projects/
|-- README.md
|-- requirements.txt
|-- 01_energy_load_forecasting/
|-- 02_predictive_maintenance/
|-- 03_network_reliability/
|-- 04_portfolio_optimization/
`-- 05_kalman_sensor_fusion/
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run any project from the repository root:

```bash
python 01_energy_load_forecasting/src/forecasting_pipeline.py
python 02_predictive_maintenance/src/maintenance_model.py
python 03_network_reliability/src/network_analysis.py
python 04_portfolio_optimization/src/portfolio_analysis.py
python 05_kalman_sensor_fusion/src/kalman_fusion.py
```

## Design Principles

- keep every project self-contained
- use transparent numerical methods instead of black-box tooling
- prefer measurable outputs over visual polish
- document assumptions directly in code and README files

## Possible Extensions

- add unit tests for numerical routines
- introduce richer datasets or public benchmarks
- export results to figures and reports
- compare current baselines against more advanced models
