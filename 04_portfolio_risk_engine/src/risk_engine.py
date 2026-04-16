from __future__ import annotations

import numpy as np


def portfolio_setup() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.array([0.35, 0.25, 0.20, 0.20])
    mean_returns = np.array([0.010, 0.007, 0.012, 0.005])
    covariance = np.array(
        [
            [0.0030, 0.0012, 0.0016, 0.0008],
            [0.0012, 0.0022, 0.0011, 0.0007],
            [0.0016, 0.0011, 0.0035, 0.0010],
            [0.0008, 0.0007, 0.0010, 0.0015],
        ]
    )
    return weights, mean_returns, covariance


def monte_carlo_returns(
    mean_returns: np.ndarray, covariance: np.ndarray, simulations: int = 20000, seed: int = 11
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean_returns, covariance, size=simulations)


def risk_metrics(portfolio_returns: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    loss_distribution = -portfolio_returns
    var = float(np.quantile(loss_distribution, alpha))
    cvar = float(loss_distribution[loss_distribution >= var].mean())
    return var, cvar


def stress_test(weights: np.ndarray) -> dict[str, float]:
    scenarios = {
        "rate_shock": np.array([-0.08, -0.05, -0.09, -0.03]),
        "tech_selloff": np.array([-0.12, -0.06, -0.15, -0.02]),
        "defensive_rotation": np.array([-0.02, 0.01, -0.01, 0.03]),
    }
    return {name: float(weights @ scenario) for name, scenario in scenarios.items()}


def main() -> None:
    weights, mean_returns, covariance = portfolio_setup()
    simulated_asset_returns = monte_carlo_returns(mean_returns, covariance)
    portfolio_returns = simulated_asset_returns @ weights

    expected_return = float(portfolio_returns.mean())
    volatility = float(portfolio_returns.std())
    var_95, cvar_95 = risk_metrics(portfolio_returns)
    stress_results = stress_test(weights)

    print("Portfolio Risk Engine")
    print("-" * 60)
    print(f"Expected return: {expected_return:.4f}")
    print(f"Volatility:      {volatility:.4f}")
    print(f"95% VaR:         {var_95:.4f}")
    print(f"95% CVaR:        {cvar_95:.4f}")
    print()
    print("Stress scenarios:")
    for name, result in stress_results.items():
        print(f"  {name}: {result:.4f}")


if __name__ == "__main__":
    main()
