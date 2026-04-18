# Applied Mathematics and Computer Engineering Projects

![Tests](https://github.com/Danylan13/MathComp-Projects/actions/workflows/tests.yml/badge.svg)

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

## Highlights

- energy forecasting improves test RMSE from `38.853 MW` to `15.314 MW`
- predictive maintenance compares unsupervised monitoring against supervised classification
- network reliability includes Monte Carlo availability checks for routed traffic
- portfolio analysis exports efficient frontier and stress scenarios
- sensor fusion compares raw GPS, Kalman filtering, and RTS smoothing

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

Or generate and validate all project artifacts in one pass:

```bash
python run_all.py
```

## Project Gallery

### 01 Energy Load Forecasting

Short-term load prediction with time-series baselines, OLS, and ridge regression.

![Energy Load Forecasting](01_energy_load_forecasting/outputs/forecast_diagnostics.png)

Quick files:
- [README](01_energy_load_forecasting/README.md)
- [Results](01_energy_load_forecasting/RESULTS.md)

### 02 Predictive Maintenance

Failure-risk scoring from aggregated sensor windows with Mahalanobis distance and logistic regression.

![Predictive Maintenance](02_predictive_maintenance/outputs/maintenance_diagnostics.png)

Quick files:
- [README](02_predictive_maintenance/README.md)
- [Results](02_predictive_maintenance/RESULTS.md)

### 03 Network Reliability

Reliability-aware routing on a constrained backbone network with Monte Carlo availability checks.

![Network Reliability](03_network_reliability/outputs/network_paths.png)

Quick files:
- [README](03_network_reliability/README.md)
- [Results](03_network_reliability/RESULTS.md)

### 04 Portfolio Optimization

Backtested allocation with efficient frontier export, tail-risk metrics, and stress scenarios.

![Portfolio Optimization](04_portfolio_optimization/outputs/portfolio_backtest.png)

Quick files:
- [README](04_portfolio_optimization/README.md)
- [Results](04_portfolio_optimization/RESULTS.md)

### 05 Kalman Sensor Fusion

Trajectory reconstruction with Kalman filtering, RTS smoothing, and dropout robustness checks.

![Kalman Sensor Fusion](05_kalman_sensor_fusion/outputs/trajectory_comparison.png)

Quick files:
- [README](05_kalman_sensor_fusion/README.md)
- [Results](05_kalman_sensor_fusion/RESULTS.md)

## Design Principles

- keep every project self-contained
- use transparent numerical methods instead of black-box tooling
- prefer measurable outputs over visual polish
- document assumptions directly in code and README files

## Testing

Run the unit test suite from the repository root:

```bash
python -m unittest discover -s tests
```

## Possible Extensions

- introduce richer datasets or public benchmarks
- export results to figures and reports
- compare current baselines against more advanced models
- package the projects as small interactive demos
