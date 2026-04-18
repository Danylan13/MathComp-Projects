from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "05_kalman_sensor_fusion" / "src" / "kalman_fusion.py"
SPEC = importlib.util.spec_from_file_location("kalman_fusion", MODULE_PATH)
kalman_fusion = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = kalman_fusion
SPEC.loader.exec_module(kalman_fusion)


class KalmanSensorFusionTests(unittest.TestCase):
    def test_rmse_zero_for_identical_inputs(self) -> None:
        reference = np.array([[0.0, 1.0], [2.0, 3.0]])
        estimate = np.array([[0.0, 1.0], [2.0, 3.0]])
        self.assertAlmostEqual(kalman_fusion.rmse(reference, estimate), 0.0)

    def test_rts_smoother_preserves_shape(self) -> None:
        filtered_states = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        predicted_states = filtered_states.copy()
        predicted_cov = np.array([np.eye(2) for _ in range(3)])
        filtered_cov = np.array([np.eye(2) for _ in range(3)])
        transition = np.eye(2)
        smoothed = kalman_fusion.run_rts_smoother(
            filtered_states, predicted_states, predicted_cov, filtered_cov, transition
        )
        self.assertEqual(smoothed.shape, filtered_states.shape)


if __name__ == "__main__":
    unittest.main()
