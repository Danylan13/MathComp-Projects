from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "01_energy_load_forecasting" / "src" / "forecasting_pipeline.py"
SPEC = importlib.util.spec_from_file_location("forecasting_pipeline", MODULE_PATH)
forecasting_pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = forecasting_pipeline
SPEC.loader.exec_module(forecasting_pipeline)


class EnergyLoadForecastingTests(unittest.TestCase):
    def test_standardize_centers_training_data(self) -> None:
        train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        test = np.array([[7.0, 8.0]])
        scaled_train, scaled_test = forecasting_pipeline.standardize(train, test)
        self.assertTrue(np.allclose(scaled_train.mean(axis=0), np.zeros(2)))
        self.assertEqual(scaled_test.shape, (1, 2))

    def test_rmse_and_mape_are_zero_for_perfect_prediction(self) -> None:
        y_true = np.array([10.0, 12.0, 14.0])
        y_pred = np.array([10.0, 12.0, 14.0])
        self.assertAlmostEqual(forecasting_pipeline.rmse(y_true, y_pred), 0.0)
        self.assertAlmostEqual(forecasting_pipeline.mape(y_true, y_pred), 0.0)

    def test_tune_alpha_returns_candidate_value(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(64, 4))
        y = 2 * x[:, 0] - 0.5 * x[:, 1] + 0.1
        alpha, metrics = forecasting_pipeline.tune_alpha(x, y, ["a", "b", "c", "d"], [0.1, 1.0, 10.0])
        self.assertIn(alpha, [0.1, 1.0, 10.0])
        self.assertEqual(len(metrics), 3)


if __name__ == "__main__":
    unittest.main()
