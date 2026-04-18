from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "04_portfolio_optimization" / "src" / "portfolio_analysis.py"
SPEC = importlib.util.spec_from_file_location("portfolio_analysis", MODULE_PATH)
portfolio_analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = portfolio_analysis
SPEC.loader.exec_module(portfolio_analysis)


class PortfolioOptimizationTests(unittest.TestCase):
    def test_generate_weight_grid_sums_to_one(self) -> None:
        grid = portfolio_analysis.generate_weight_grid(0.5)
        self.assertTrue(np.allclose(grid.sum(axis=1), np.ones(len(grid))))

    def test_max_drawdown_is_zero_for_monotone_wealth(self) -> None:
        wealth = np.array([1.0, 1.1, 1.2, 1.3])
        self.assertAlmostEqual(portfolio_analysis.max_drawdown(wealth), 0.0)

    def test_risk_metrics_return_ordered_values(self) -> None:
        returns = np.array([0.02, -0.01, 0.01, -0.03, 0.00])
        var, cvar = portfolio_analysis.risk_metrics(returns)
        self.assertGreaterEqual(cvar, var)


if __name__ == "__main__":
    unittest.main()
