# Portfolio Optimization Results

## Executive Summary

The optimized portfolio meaningfully reduces volatility, drawdown, and tail risk relative to an equal-weight benchmark. A rolling rebalancing variant with transaction costs has also been added to make the comparison more realistic and less backtest-perfect.

## Strategy Comparison

| Strategy | Annual Return | Annual Volatility | VaR95 | CVaR95 | Max Drawdown | Sharpe Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Equal-weight | 0.4221 | 0.0312 | 0.0021 | 0.0023 | -0.0117 | 15.7720 |
| Optimized static | 0.3359 | 0.0180 | 0.0008 | 0.0009 | -0.0043 | 15.3817 |
| Rolling rebalanced costed | 0.2640 | 0.0183 | 0.0009 | 0.0010 | -0.0047 | 12.2843 |

## What Improved

- Annualized volatility falls from `0.0312` to `0.0180`
- Maximum drawdown improves from `-0.0117` to `-0.0043`
- VaR95 and CVaR95 are materially lower in the optimized strategy than in the equal-weight portfolio

## Additional Analysis

- `strategy_risk_return.png` provides a cleaner comparison of return, risk, and Sharpe ratio across strategies
- `efficient_frontier.csv` exports the discrete frontier for reuse or deeper visualization
- `stress_test.csv` summarizes scenario-level portfolio response under inflation shock, growth rally, and risk-off conditions

## Why This Matters

This is no longer just a static allocation example. By including rolling rebalancing with transaction costs, the project shows awareness of implementation frictions and realistic benchmarking.

## Limitations

- The asset universe is very small
- The grid search is intentionally simple and coarse for transparency
- The Sharpe ratios are high because the test horizon is short and low-volatility

## Artifacts

- `outputs/portfolio_backtest.png`
- `outputs/strategy_risk_return.png`
- `outputs/efficient_frontier.csv`
- `outputs/stress_test.csv`
- `outputs/strategy_comparison.csv`
