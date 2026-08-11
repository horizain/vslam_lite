#!/usr/bin/env python3
"""benchmark.py 门限聚合回归测试。

这些测试不启动 SLAM，只验证多轮统计的方向性：min 门必须看各轮最小
值，max 门必须看各轮最大值。
"""
import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path

import benchmark
import compare_trajectories
import soak_test
import yaml


class BenchmarkGateTest(unittest.TestCase):
    def test_min_gate_uses_lowest_round(self):
        rounds = [{"valid_ratio": 0.98}, {"valid_ratio": 1.0}]
        aggregate = benchmark.aggregate(rounds)
        result = benchmark.check_gates(
            aggregate, {"valid_ratio": {"min": 0.99}}, True)
        self.assertEqual(result["valid_ratio"]["status"], "fail")
        self.assertEqual(result["valid_ratio"]["worst"], 0.98)

    def test_max_gate_uses_highest_round(self):
        rounds = [{"lost_count": 1}, {"lost_count": 31}]
        aggregate = benchmark.aggregate(rounds)
        result = benchmark.check_gates(
            aggregate, {"lost_count": {"max": 30}}, True)
        self.assertEqual(result["lost_count"]["status"], "fail")
        self.assertEqual(result["lost_count"]["worst"], 31.0)

    def test_configured_gate_cannot_skip_missing_metric(self):
        result = benchmark.check_gates(
            {}, {"ate_rmse": {"max": 40.0}, "jumps_10m": {"max": 0}}, True)
        self.assertEqual(result["ate_rmse"]["status"], "fail")
        self.assertEqual(result["jumps_10m"]["status"], "fail")

    def test_zero_processed_frames_fail_soak_gate(self):
        args = SimpleNamespace(deadline_ms=80, queue_capacity=3,
                               min_valid_ratio=0.99)
        metrics = {"frames_processed": 0, "pose_accepted": 0,
                   "latency_p99_ms": 0.0, "deadline_miss_ratio": 0.0,
                   "input_queue_hwm": 0, "backend_pending": 0}
        ok, report = soak_test.check_metrics(metrics, args, {"failures": []}, True)
        self.assertFalse(ok)
        self.assertTrue(any("valid_ratio" in f for f in report["failures"]))
        with self.assertRaises(RuntimeError):
            benchmark.validate_run_metrics(metrics)

    def test_empty_reference_cannot_pass_determinism_gate(self):
        self.assertFalse(compare_trajectories.compare_traj([], [], 1e-6, 1e-8))
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.csv"
            b = Path(tmp) / "b.csv"
            a.write_text("frame_id,state\n")
            b.write_text("frame_id,state\n")
            self.assertFalse(compare_trajectories.compare_status(a, b))

    def test_repository_gate_names_and_specs_parse_exactly(self):
        config_path = Path(__file__).resolve().parent.parent / "config/benchmark.yaml"
        with config_path.open() as f:
            gates = yaml.safe_load(f)["Gates"]
        self.assertIn("deadline_miss_ratio", gates)
        self.assertEqual(gates["deadline_miss_ratio"], {"max": 0.01})
        self.assertFalse(any("{" in str(name) for name in gates))


if __name__ == "__main__":
    unittest.main()
