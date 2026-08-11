#!/usr/bin/env python3
"""Self-tests for the dataset-independent loop recall oracle."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluate_loop_recall import LoopEvent, Pose, evaluate, main, read_events


def synthetic_poses():
    # First traversal x=0..9, second traversal x=9..0.  The second traversal
    # is a single continuous revisitation interval under the oracle settings.
    return [Pose(float(index), (float(index if index < 10 else 19 - index), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            for index in range(20)]


class LoopRecallTests(unittest.TestCase):
    def test_empty_events_fail_closed(self):
        report = evaluate(synthetic_poses(), [], max_distance=0.2, min_time_gap=2.0)
        self.assertEqual(report["status"], "no_events")
        self.assertEqual(report["metrics"]["events"], 0)
        self.assertEqual(report["metrics"]["recall"], 0.0)
        self.assertEqual(report["metrics"]["false_detections"], 0)

    def test_missed_loop_is_recall_failure(self):
        report = evaluate(synthetic_poses(), [], max_distance=0.2, min_time_gap=2.0)
        self.assertGreater(report["metrics"]["total_intervals"], 0)
        self.assertEqual(report["metrics"]["detected_intervals"], 0)

    def test_false_detection_and_valid_hit(self):
        events = [LoopEvent(13.0, 6.0), LoopEvent(3.0, 18.0)]
        report = evaluate(synthetic_poses(), events, max_distance=0.2, min_time_gap=2.0)
        self.assertEqual(report["metrics"]["positive_events"], 1)
        self.assertEqual(report["metrics"]["false_detections"], 1)
        self.assertEqual(report["metrics"]["event_hit_rate"], 0.5)
        self.assertEqual(report["metrics"]["recall"], 1.0)

    def test_continuous_events_are_deduplicated(self):
        events = [LoopEvent(13.0, 6.0), LoopEvent(14.0, 5.0), LoopEvent(15.0, 4.0)]
        report = evaluate(synthetic_poses(), events, max_distance=0.2, min_time_gap=2.0)
        self.assertEqual(report["metrics"]["positive_events"], 3)
        self.assertEqual(report["metrics"]["detected_intervals"], 1)
        self.assertEqual(report["metrics"]["total_intervals"], 1)
        self.assertEqual(report["metrics"]["recall"], 1.0)
        self.assertEqual(len(report["metrics"]["detection_delays"]), 1)

    def test_unlabelled_frame_id_log_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.log"
            path.write_text("Loop closed! kf#150 -> kf#1595\n", encoding="utf-8")
            self.assertEqual(read_events(path), [])

    def test_jsonl_with_comments_is_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("# comment\n{\"query_time\": 13, \"reference_time\": 6}\n"
                            '{"query_time": 14, "reference_time": 5}\n', encoding="utf-8")
            self.assertEqual(len(read_events(path)), 2)

    def test_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "gt.tum"
            events = root / "events.jsonl"
            output = root / "report.json"
            gt.write_text("\n".join(f"{pose.timestamp} {pose.position[0]} 0 0 0 0 0 1" for pose in synthetic_poses()) + "\n", encoding="utf-8")
            events.write_text('{"query_time": 13, "reference_time": 6}\n', encoding="utf-8")
            self.assertEqual(main(["--gt", str(gt), "--events", str(events), "--output", str(output),
                                   "--max-distance", "0.2", "--min-time-gap", "2"]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["metrics"]["positive_events"], 1)


if __name__ == "__main__":
    unittest.main()
