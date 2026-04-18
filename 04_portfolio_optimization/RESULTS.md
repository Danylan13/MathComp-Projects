# Portfolio Optimization Results

## Strategy Comparison

| Strategy | Annual Return | Annual Volatility | VaR95 | CVaR95 | Max Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal-weight | 0.4221 | 0.0312 | 0.0021 | 0.0023 | -0.0117 |
| Optimized | 0.3359 | 0.0180 | 0.0008 | 0.0009 | -0.0043 |

## Notes

- Optimized weights: `ALPHA 0.30`, `BETA 0.00`, `GAMMA 0.20`, `DELTA 0.50`
- Optimization trades some upside for materially lower drawdown and tail risk
- Artifacts: `outputs/portfolio_backtest.png`, `outputs/efficient_frontier.csv`, `outputs/stress_test.csv`
