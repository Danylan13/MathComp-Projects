from __future__ import annotations

from itertools import product
from pathlib import Path
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "asset_prices.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def load_prices() -> tuple[np.ndarray, list[str]]:
    raw = np.genfromtxt(DATA_PATH, delimiter=",", names=True, dtype=None, encoding="utf-8")
    asset_names = [name for name in raw.dtype.names if name != "date"]
    prices = np.column_stack([raw[name].astype(float) for name in asset_names])
    return prices, asset_names


def log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices), axis=0)


def generate_weight_grid(step: float = 0.1) -> np.ndarray:
    levels = np.arange(0.0, 1.0 + step / 2, step)
    weights = []
    for combo in product(levels, repeat=4):
        if np.isclose(sum(combo), 1.0):
            weights.append(combo)
    return np.asarray(weights, dtype=float)


def choose_min_variance_portfolio(train_returns: np.ndarray) -> np.ndarray:
    mean_returns = train_returns.mean(axis=0)
    covariance = np.cov(train_returns, rowvar=False)
    grid = generate_weight_grid(0.1)
    return_floor = float(np.mean(train_returns @ np.full(train_returns.shape[1], 1 / train_returns.shape[1])))

    feasible_weights = []
    for weights in grid:
        expected_return = float(weights @ mean_returns)
        variance = float(weights @ covariance @ weights)
        if expected_return >= return_floor:
            feasible_weights.append((variance, weights))

    if feasible_weights:
        feasible_weights.sort(key=lambda item: item[0])
        return feasible_weights[0][1]

    min_variance_index = int(np.argmin([weights @ covariance @ weights for weights in grid]))
    return grid[min_variance_index]


def backtest(returns: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    simple_returns = np.exp(returns @ weights) - 1.0
    wealth = np.cumprod(1.0 + simple_returns)
    return simple_returns, wealth


def max_drawdown(wealth: np.ndarray) -> float:
    peaks = np.maximum.accumulate(wealth)
    drawdowns = wealth / peaks - 1.0
    return float(drawdowns.min())


def risk_metrics(simple_returns: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    losses = -simple_returns
    var = float(np.quantile(losses, alpha))
    cvar = float(losses[losses >= var].mean())
    return var, cvar


def annualized_summary(simple_returns: np.ndarray) -> tuple[float, float, float, float]:
    avg_daily = float(simple_returns.mean())
    vol_daily = float(simple_returns.std())
    annual_return = (1.0 + avg_daily) ** 252 - 1.0
    annual_vol = vol_daily * np.sqrt(252)
    var_95, cvar_95 = risk_metrics(simple_returns)
    return annual_return, annual_vol, var_95, cvar_95


def frontier_points(train_returns: np.ndarray) -> list[tuple[float, float, tuple[float, ...]]]:
    covariance = np.cov(train_returns, rowvar=False)
    mean_returns = train_returns.mean(axis=0)
    points = []
    for weights in generate_weight_grid(0.1):
        expected_return = float(weights @ mean_returns)
        volatility = float(np.sqrt(weights @ covariance @ weights))
        points.append((expected_return, volatility, tuple(weights.tolist())))
    return points


def save_tables(asset_names: list[str], optimized_weights: np.ndarray, points: list[tuple[float, float, tuple[float, ...]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "efficient_frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected_return", "volatility", *asset_names])
        for expected_return, volatility, weights in points:
            writer.writerow([f"{expected_return:.6f}", f"{volatility:.6f}", *[f"{weight:.2f}" for weight in weights]])

    stress_scenarios = {
        "inflation_shock": np.array([-0.030, -0.014, -0.045, -0.006]),
        "growth_rally": np.array([0.024, 0.011, 0.032, 0.008]),
        "risk_off": np.array([-0.020, -0.018, -0.027, -0.009]),
    }
    with (OUTPUT_DIR / "stress_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "optimized_return"])
        for name, vector in stress_scenarios.items():
            writer.writerow([name, f"{float(optimized_weights @ vector):.6f}"])


def save_plots(
    optimized_wealth: np.ndarray,
    equal_wealth: np.ndarray,
    frontier: list[tuple[float, float, tuple[float, ...]]],
    optimized_weights: np.ndarray,
    equal_weights: np.ndarray,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_axis = np.arange(1, len(optimized_wealth) + 1)
    optimized_drawdown = optimized_wealth / np.maximum.accumulate(optimized_wealth) - 1.0
    equal_drawdown = equal_wealth / np.maximum.accumulate(equal_wealth) - 1.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)
    axes[0].plot(x_axis, optimized_wealth, label="optimized", linewidth=2.2, color="#117a65")
    axes[0].plot(x_axis, equal_wealth, label="equal-weight", linewidth=2.0, color="#c0392b")
    axes[0].set_title("Out-of-Sample Wealth Trajectory")
    axes[0].set_ylabel("Wealth index")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(x_axis, optimized_drawdown, label="optimized drawdown", linewidth=2.2, color="#117a65")
    axes[1].plot(x_axis, equal_drawdown, label="equal-weight drawdown", linewidth=2.0, color="#c0392b")
    axes[1].set_title("Drawdown Comparison")
    axes[1].set_xlabel("Test day")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    frontier_returns = np.array([point[0] for point in frontier])
    frontier_vols = np.array([point[1] for point in frontier])
    frontier_weights = [np.array(point[2]) for point in frontier]

    optimized_idx = int(np.argmin([np.linalg.norm(weights - optimized_weights) for weights in frontier_weights]))
    equal_idx = int(np.argmin([np.linalg.norm(weights - equal_weights) for weights in frontier_weights]))

    axes[2].scatter(frontier_vols, frontier_returns, s=20, color="#95a5a6", alpha=0.7)
    axes[2].scatter(frontier_vols[optimized_idx], frontier_returns[optimized_idx], s=80, color="#117a65", label="optimized")
    axes[2].scatter(frontier_vols[equal_idx], frontier_returns[equal_idx], s=80, color="#c0392b", label="equal-weight")
    axes[2].set_title("Discrete Efficient Frontier")
    axes[2].set_xlabel("Volatility")
    axes[2].set_ylabel("Expected return")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.savefig(OUTPUT_DIR / "portfolio_backtest.png", dpi=160)
    plt.close(fig)


def main() -> None:
    prices, asset_names = load_prices()
    returns = log_returns(prices)
    split_index = 42
    train_returns = returns[:split_index]
    test_returns = returns[split_index:]

    optimized_weights = choose_min_variance_portfolio(train_returns)
    equal_weights = np.full(len(asset_names), 1.0 / len(asset_names))

    optimized_simple, optimized_wealth = backtest(test_returns, optimized_weights)
    equal_simple, equal_wealth = backtest(test_returns, equal_weights)

    optimized_summary = annualized_summary(optimized_simple)
    equal_summary = annualized_summary(equal_simple)
    points = frontier_points(train_returns)
    save_tables(asset_names, optimized_weights, points)
    save_plots(optimized_wealth, equal_wealth, points, optimized_weights, equal_weights)

    print("Portfolio Optimization and Risk")
    print("-" * 72)
    print("Optimized weights:")
    for name, weight in zip(asset_names, optimized_weights):
        print(f"  {name:<6} {weight:.2f}")
    print()
    print("Out-of-sample comparison:")
    print(
        f"  optimized  annual_return={optimized_summary[0]:.4f} "
        f"annual_vol={optimized_summary[1]:.4f} "
        f"VaR95={optimized_summary[2]:.4f} CVaR95={optimized_summary[3]:.4f} "
        f"max_drawdown={max_drawdown(optimized_wealth):.4f}"
    )
    print(
        f"  equal-wt   annual_return={equal_summary[0]:.4f} "
        f"annual_vol={equal_summary[1]:.4f} "
        f"VaR95={equal_summary[2]:.4f} CVaR95={equal_summary[3]:.4f} "
        f"max_drawdown={max_drawdown(equal_wealth):.4f}"
    )
    print()
    print("Final wealth on test window:")
    print(f"  optimized={optimized_wealth[-1]:.4f}")
    print(f"  equal-wt ={equal_wealth[-1]:.4f}")
    print()
    print(f"Saved plot: {OUTPUT_DIR / 'portfolio_backtest.png'}")
    print(f"Saved tables: {OUTPUT_DIR / 'efficient_frontier.csv'}, {OUTPUT_DIR / 'stress_test.csv'}")


if __name__ == "__main__":
    main()
