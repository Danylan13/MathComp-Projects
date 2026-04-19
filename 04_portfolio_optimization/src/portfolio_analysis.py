from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "asset_prices.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def load_prices(data_path: Path) -> tuple[np.ndarray, list[str]]:
    raw = np.genfromtxt(data_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
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


def choose_min_variance_portfolio(train_returns: np.ndarray, grid_step: float = 0.1) -> np.ndarray:
    mean_returns = train_returns.mean(axis=0)
    covariance = np.cov(train_returns, rowvar=False)
    grid = generate_weight_grid(grid_step)
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


def sharpe_ratio(simple_returns: np.ndarray, risk_free_daily: float = 0.00005) -> float:
    excess = simple_returns - risk_free_daily
    vol = float(excess.std())
    if vol == 0.0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / vol)


def frontier_points(train_returns: np.ndarray, grid_step: float = 0.1) -> list[tuple[float, float, tuple[float, ...]]]:
    covariance = np.cov(train_returns, rowvar=False)
    mean_returns = train_returns.mean(axis=0)
    points = []
    for weights in generate_weight_grid(grid_step):
        expected_return = float(weights @ mean_returns)
        volatility = float(np.sqrt(weights @ covariance @ weights))
        points.append((expected_return, volatility, tuple(weights.tolist())))
    return points


def rolling_backtest(
    returns: np.ndarray,
    train_window: int,
    grid_step: float,
    transaction_cost_bps: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    daily_returns: list[float] = []
    previous_weights = np.full(returns.shape[1], 1.0 / returns.shape[1])
    cost_scale = transaction_cost_bps / 10000.0
    for index in range(train_window, len(returns)):
        train_slice = returns[index - train_window : index]
        weights = choose_min_variance_portfolio(train_slice, grid_step=grid_step)
        turnover = float(np.sum(np.abs(weights - previous_weights)))
        gross_simple = float(np.exp(returns[index] @ weights) - 1.0)
        net_simple = gross_simple - turnover * cost_scale
        daily_returns.append(net_simple)
        previous_weights = weights
    wealth = np.cumprod(1.0 + np.asarray(daily_returns))
    return np.asarray(daily_returns), wealth


def save_tables(
    asset_names: list[str],
    optimized_weights: np.ndarray,
    points: list[tuple[float, float, tuple[float, ...]]],
    strategy_rows: list[tuple[str, float, float, float, float, float, float]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "efficient_frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected_return", "volatility", *asset_names])
        for expected_return, volatility, weights in points:
            writer.writerow([f"{expected_return:.6f}", f"{volatility:.6f}", *[f"{weight:.2f}" for weight in weights]])

    stress_scenarios = {
        "inflation_shock": np.array([-0.030, -0.014, -0.045, -0.006]),
        "growth_rally": np.array([0.024, 0.011, 0.032, 0.008]),
        "risk_off": np.array([-0.020, -0.018, -0.027, -0.009]),
    }
    with (output_dir / "stress_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "optimized_return"])
        for name, vector in stress_scenarios.items():
            writer.writerow([name, f"{float(optimized_weights @ vector):.6f}"])

    with (output_dir / "strategy_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["strategy", "annual_return", "annual_vol", "var95", "cvar95", "max_drawdown", "sharpe_ratio"])
        for row in strategy_rows:
            writer.writerow([row[0], *[f"{value:.6f}" for value in row[1:]]])


def save_main_plot(
    optimized_wealth: np.ndarray,
    equal_wealth: np.ndarray,
    frontier: list[tuple[float, float, tuple[float, ...]]],
    optimized_weights: np.ndarray,
    equal_weights: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(output_dir / "portfolio_backtest.png", dpi=160)
    plt.close(fig)


def save_strategy_plot(strategy_rows: list[tuple[str, float, float, float, float, float, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row[0] for row in strategy_rows]
    vols = np.asarray([row[2] for row in strategy_rows])
    returns = np.asarray([row[1] for row in strategy_rows])
    sharpes = np.asarray([row[6] for row in strategy_rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(labels, returns, color=["#117a65", "#c0392b", "#1f77b4"])
    axes[0].set_title("Annualized Return by Strategy")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    scatter = axes[1].scatter(vols, returns, s=120, c=sharpes, cmap="viridis")
    for label, vol, ret in zip(labels, vols, returns):
        axes[1].text(vol, ret, f" {label}", va="center")
    axes[1].set_title("Risk-Return Map")
    axes[1].set_xlabel("Annualized volatility")
    axes[1].set_ylabel("Annualized return")
    axes[1].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[1], label="Sharpe ratio")
    fig.savefig(output_dir / "strategy_risk_return.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portfolio optimization, backtesting, and strategy comparison.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to price history CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated outputs.")
    parser.add_argument("--train-split", type=int, default=42, help="Number of observations in the train window.")
    parser.add_argument("--grid-step", type=float, default=0.1, help="Grid step for discrete weight search.")
    parser.add_argument("--rolling-window", type=int, default=30, help="Rolling train window for rebalancing benchmark.")
    parser.add_argument("--transaction-cost-bps", type=float, default=8.0, help="Transaction cost in basis points.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices, asset_names = load_prices(args.data_path)
    returns = log_returns(prices)
    train_returns = returns[: args.train_split]
    test_returns = returns[args.train_split :]
    optimized_weights = choose_min_variance_portfolio(train_returns, grid_step=args.grid_step)
    equal_weights = np.full(len(asset_names), 1.0 / len(asset_names))

    optimized_simple, optimized_wealth = backtest(test_returns, optimized_weights)
    equal_simple, equal_wealth = backtest(test_returns, equal_weights)
    rolling_simple, rolling_wealth = rolling_backtest(
        returns,
        train_window=args.rolling_window,
        grid_step=args.grid_step,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    optimized_summary = annualized_summary(optimized_simple)
    equal_summary = annualized_summary(equal_simple)
    rolling_summary = annualized_summary(rolling_simple)
    points = frontier_points(train_returns, grid_step=args.grid_step)
    strategy_rows = [
        (
            "optimized_static",
            optimized_summary[0],
            optimized_summary[1],
            optimized_summary[2],
            optimized_summary[3],
            max_drawdown(optimized_wealth),
            sharpe_ratio(optimized_simple),
        ),
        (
            "equal_weight",
            equal_summary[0],
            equal_summary[1],
            equal_summary[2],
            equal_summary[3],
            max_drawdown(equal_wealth),
            sharpe_ratio(equal_simple),
        ),
        (
            "rolling_rebalanced_costed",
            rolling_summary[0],
            rolling_summary[1],
            rolling_summary[2],
            rolling_summary[3],
            max_drawdown(rolling_wealth),
            sharpe_ratio(rolling_simple),
        ),
    ]

    save_tables(asset_names, optimized_weights, points, strategy_rows, args.output_dir)
    save_main_plot(optimized_wealth, equal_wealth, points, optimized_weights, equal_weights, args.output_dir)
    save_strategy_plot(strategy_rows, args.output_dir)

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
    print(
        f"  rolling    annual_return={rolling_summary[0]:.4f} "
        f"annual_vol={rolling_summary[1]:.4f} "
        f"VaR95={rolling_summary[2]:.4f} CVaR95={rolling_summary[3]:.4f} "
        f"max_drawdown={max_drawdown(rolling_wealth):.4f}"
    )
    print()
    print("Final wealth on test window:")
    print(f"  optimized={optimized_wealth[-1]:.4f}")
    print(f"  equal-wt ={equal_wealth[-1]:.4f}")
    print(f"  rolling  ={rolling_wealth[-1]:.4f}")
    print()
    print(
        f"Saved plots: {args.output_dir / 'portfolio_backtest.png'}, {args.output_dir / 'strategy_risk_return.png'}"
    )
    print(
        "Saved tables: "
        f"{args.output_dir / 'efficient_frontier.csv'}, "
        f"{args.output_dir / 'stress_test.csv'}, "
        f"{args.output_dir / 'strategy_comparison.csv'}"
    )


if __name__ == "__main__":
    main()
