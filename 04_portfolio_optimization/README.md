# Portfolio Optimization and Risk

## Problem

Construct a portfolio from historical price data, optimize asset weights under a return constraint, and compare the resulting strategy against an equal-weight benchmark.

## Data

File: `data/asset_prices.csv`

Assets:

- `ALPHA`
- `BETA`
- `GAMMA`
- `DELTA`

## Method

The project:

- converts price history into log returns
- estimates mean returns and covariance on a training window
- searches the simplex for a minimum-variance allocation with a return floor
- backtests the allocation on a holdout segment
- computes volatility, Value at Risk, Conditional Value at Risk, and maximum drawdown

## Run

```bash
python 04_portfolio_optimization/src/portfolio_analysis.py
```

## Output

The script reports:

- optimized weights
- benchmark comparison
- out-of-sample tail-risk metrics
- backtest wealth trajectory summary
- saved backtest plot in `outputs/portfolio_backtest.png`
