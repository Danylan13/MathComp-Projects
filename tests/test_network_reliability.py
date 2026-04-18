from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "03_network_reliability" / "src" / "network_analysis.py"
SPEC = importlib.util.spec_from_file_location("network_analysis", MODULE_PATH)
network_analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = network_analysis
SPEC.loader.exec_module(network_analysis)


class NetworkReliabilityTests(unittest.TestCase):
    def test_path_metrics_accumulate_latency_and_reliability(self) -> None:
        graph = {
            "A": [network_analysis.Edge("B", 10.0, 0.1, 100.0)],
            "B": [
                network_analysis.Edge("A", 10.0, 0.1, 100.0),
                network_analysis.Edge("C", 5.0, 0.2, 80.0),
            ],
            "C": [network_analysis.Edge("B", 5.0, 0.2, 80.0)],
        }
        report = network_analysis.path_metrics(graph, ["A", "B", "C"])
        self.assertAlmostEqual(report.latency_ms, 15.0)
        self.assertAlmostEqual(report.reliability, 0.9 * 0.8)
        self.assertAlmostEqual(report.bottleneck_mbps, 80.0)

    def test_simulate_path_availability_is_in_unit_interval(self) -> None:
        report = network_analysis.PathReport(["A", "B"], 10.0, 100.0, 0.95, 15.0)
        value = network_analysis.simulate_path_availability(report, trials=1000, seed=1)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
