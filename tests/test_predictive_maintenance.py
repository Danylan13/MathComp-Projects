from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "02_predictive_maintenance" / "src" / "maintenance_model.py"
SPEC = importlib.util.spec_from_file_location("maintenance_model", MODULE_PATH)
maintenance_model = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = maintenance_model
SPEC.loader.exec_module(maintenance_model)


class PredictiveMaintenanceTests(unittest.TestCase):
    def test_compute_metrics_matches_expected_counts(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        metrics = maintenance_model.compute_metrics(y_true, y_pred)
        self.assertEqual(metrics.confusion, (1, 1, 1, 1))
        self.assertAlmostEqual(metrics.precision, 0.5)
        self.assertAlmostEqual(metrics.recall, 0.5)

    def test_sigmoid_maps_zero_to_half(self) -> None:
        values = np.array([0.0])
        result = maintenance_model.sigmoid(values)
        self.assertTrue(np.allclose(result, np.array([0.5])))

    def test_select_threshold_returns_candidate(self) -> None:
        probabilities = np.array([0.2, 0.4, 0.7, 0.9])
        y_true = np.array([0, 0, 1, 1])
        threshold, scores = maintenance_model.select_threshold(probabilities, y_true)
        self.assertGreaterEqual(threshold, 0.30)
        self.assertLessEqual(threshold, 0.80)
        self.assertTrue(len(scores) > 0)


if __name__ == "__main__":
    unittest.main()
