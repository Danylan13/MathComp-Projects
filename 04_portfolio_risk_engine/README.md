# Portfolio Risk Engine

## Goal

Estimate risk for a small portfolio using Monte Carlo simulation of correlated asset returns.

## Why It Matters

This project shows how probability, covariance, and simulation can be turned into a compact quantitative tool for decision support.

## Concepts

- random vectors
- covariance matrices
- Monte Carlo simulation
- Value at Risk
- Conditional Value at Risk
- stress testing

## Run

```bash
python 04_portfolio_risk_engine/src/risk_engine.py
```

## Output

The script prints:

- expected portfolio return
- volatility
- 95% VaR
- 95% CVaR
- scenario stress results
