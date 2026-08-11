#!/usr/bin/env python3
"""Offline, dataset-independent loop-closure recall evaluation.

The evaluator deliberately consumes timestamps, rather than frame numbers.  A
GT TUM file is used only to create an offline oracle; it is never part of the
runtime loop-closure path.  Loop events are normally JSON/CSV records with a
query and reference timestamp, but the parser also accepts simple log lines
such as ``loop query_time=12.3 reference_time=4.1``.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class Pose:
    timestamp: float
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]  # x, y, z, w


@dataclass(frozen=True)
class LoopEvent:
    query_time: float
    reference_time: float
    detection_time: Optional[float] = None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def read_tum(path: str | Path) -> list[Pose]:
    poses: list[Pose] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.replace(",", " ").split()
            if len(fields) < 8:
                raise ValueError(f"{path}:{line_no}: expected timestamp tx ty tz qx qy qz qw")
            try:
                values = [float(item) for item in fields[:8]]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: non-numeric TUM pose") from exc
            if not all(math.isfinite(item) for item in values):
                raise ValueError(f"{path}:{line_no}: non-finite TUM pose")
            poses.append(Pose(values[0], tuple(values[1:4]), tuple(values[4:8])))
    poses.sort(key=lambda pose: pose.timestamp)
    return poses


_KEYS = {
    "query_time": ("query_time", "current_time", "query_timestamp", "current_timestamp", "query", "current"),
    "reference_time": ("reference_time", "match_time", "matched_time", "reference_timestamp", "match_timestamp", "reference", "match", "matched"),
    "detection_time": ("detection_time", "detected_time", "detection_timestamp", "detected_at"),
}


def _value_for(record: dict[str, Any], names: Iterable[str]) -> Optional[float]:
    lowered = {str(key).lower(): value for key, value in record.items()}
    for name in names:
        if name in lowered and _finite(lowered[name]):
            return float(lowered[name])
    return None


def _event_from_record(record: dict[str, Any]) -> Optional[LoopEvent]:
    query = _value_for(record, _KEYS["query_time"])
    reference = _value_for(record, _KEYS["reference_time"])
    if query is None or reference is None:
        return None
    detection = _value_for(record, _KEYS["detection_time"])
    return LoopEvent(query, reference, detection)


def read_events(path: str | Path) -> list[LoopEvent]:
    """Read timestamp pairs from JSON, CSV, JSONL, or explicit timestamp logs.

    Bare ``kf#12 -> kf#34`` lines are intentionally ignored: frame IDs have no
    dataset-independent meaning and cannot be matched to a timestamped GT.
    """
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    events: list[LoopEvent] = []
    if not stripped:
        return events
    if stripped[0] in "[{":
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            records = payload if isinstance(payload, list) else [payload]
            for record in records:
                if isinstance(record, dict):
                    event = _event_from_record(record)
                    if event is not None:
                        events.append(event)
            return events
        # A JSONL file starts with ``{`` too, but is not one JSON document.
        # Parse each record independently before falling back to text logs.
        if payload is None:
            jsonl_events: list[LoopEvent] = []
            jsonl_ok = True
            for line in stripped.splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    jsonl_ok = False
                    break
                if isinstance(record, dict):
                    event = _event_from_record(record)
                    if event is not None:
                        jsonl_events.append(event)
            if jsonl_ok:
                return jsonl_events

    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return events
    # Permit comments before JSONL records as well.
    jsonl_events = []
    jsonl_ok = True
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            jsonl_ok = False
            break
        if isinstance(record, dict):
            event = _event_from_record(record)
            if event is not None:
                jsonl_events.append(event)
    if jsonl_ok:
        return jsonl_events
    # CSV with a header.  Header names prevent accidental interpretation of IDs.
    try:
        first = next(csv.reader([lines[0]]))
    except csv.Error:
        first = []
    normalized = {item.strip().lower() for item in first}
    has_header = bool(normalized & set(sum((_KEYS[key] for key in _KEYS), ())))
    if has_header:
        for row in csv.DictReader(lines):
            event = _event_from_record(dict(row))
            if event is not None:
                events.append(event)
        return events

    # Explicit key=value log records.  This also handles comma/space separated
    # timestamp CSV rows when there are at least two numeric columns.
    key_patterns = {
        "query": r"(?:query_time|current_time|query_timestamp|current_timestamp|query|current)\s*[:=]\s*([-+0-9.eE]+)",
        "reference": r"(?:reference_time|match_time|matched_time|reference_timestamp|match_timestamp|reference|match|matched)\s*[:=]\s*([-+0-9.eE]+)",
        "detection": r"(?:detection_time|detected_time|detection_timestamp|detected_at)\s*[:=]\s*([-+0-9.eE]+)",
    }
    for line in lines:
        query_match = re.search(key_patterns["query"], line, re.IGNORECASE)
        reference_match = re.search(key_patterns["reference"], line, re.IGNORECASE)
        if query_match and reference_match:
            detection_match = re.search(key_patterns["detection"], line, re.IGNORECASE)
            events.append(LoopEvent(float(query_match.group(1)), float(reference_match.group(1)),
                                    float(detection_match.group(1)) if detection_match else None))
            continue
        fields = re.split(r"[,;\s]+", line.strip())
        if len(fields) >= 2 and all(_finite(item) for item in fields[:2]):
            events.append(LoopEvent(float(fields[0]), float(fields[1]),
                                    float(fields[2]) if len(fields) > 2 and _finite(fields[2]) else None))
    return events


def _quat_angle(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    norm_first = math.sqrt(sum(value * value for value in first))
    norm_second = math.sqrt(sum(value * value for value in second))
    if norm_first <= 1e-12 or norm_second <= 1e-12:
        return math.inf
    dot = abs(sum(a * b for a, b in zip(first, second))) / (norm_first * norm_second)
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _distance(first: Pose, second: Pose) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first.position, second.position)))


def _nearest_index(timestamps: list[float], timestamp: float, tolerance: float) -> Optional[int]:
    index = bisect.bisect_left(timestamps, timestamp)
    candidates = [item for item in (index - 1, index) if 0 <= item < len(timestamps)]
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(timestamps[item] - timestamp))
    return best if abs(timestamps[best] - timestamp) <= tolerance else None


def evaluate(poses: list[Pose], events: list[LoopEvent], *, max_distance: float,
             min_time_gap: float, orientation_threshold: Optional[float] = None,
             max_interval_gap: Optional[float] = None, time_tolerance: Optional[float] = None) -> dict[str, Any]:
    """Evaluate events against an offline spatial-temporal GT oracle."""
    if not poses:
        event_count = len(events)
        return {"status": "no_ground_truth", "loopable_intervals": [], "metrics": {
            "recall": 0.0, "event_hit_rate": 0.0, "hit_rate": 0.0, "precision": 0.0,
            "false_detection_rate": 0.0, "false_detections": event_count,
            "positive_events": 0, "detected_intervals": 0, "total_intervals": 0,
            "detection_delays": [], "loopable_samples": 0, "events": event_count}}
    timestamps = [pose.timestamp for pose in poses]
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    median_dt = statistics.median(deltas) if deltas else 0.0
    tolerance = time_tolerance if time_tolerance is not None else max(median_dt * 0.5, 1e-9)
    interval_gap = max_interval_gap if max_interval_gap is not None else (median_dt * 3.0 if median_dt > 0 else math.inf)

    loopable = [False] * len(poses)
    valid_references: list[list[int]] = [[] for _ in poses]
    for current in range(len(poses)):
        for reference in range(current):
            if poses[current].timestamp - poses[reference].timestamp < min_time_gap:
                continue
            if _distance(poses[current], poses[reference]) > max_distance:
                continue
            if orientation_threshold is not None and _quat_angle(poses[current].quaternion, poses[reference].quaternion) > orientation_threshold:
                continue
            loopable[current] = True
            valid_references[current].append(reference)

    intervals: list[dict[str, Any]] = []
    current_indices: list[int] = []
    for index, is_loopable in enumerate(loopable + [False]):
        if is_loopable:
            if current_indices and poses[index].timestamp - poses[current_indices[-1]].timestamp > interval_gap:
                intervals.append(_make_interval(poses, current_indices))
                current_indices = []
            current_indices.append(index)
        elif current_indices:
            intervals.append(_make_interval(poses, current_indices))
            current_indices = []

    interval_hits: dict[int, LoopEvent] = {}
    positive_events = 0
    false_detections = 0
    accepted_events: list[dict[str, Any]] = []
    for event in events:
        query_index = _nearest_index(timestamps, event.query_time, tolerance)
        reference_index = _nearest_index(timestamps, event.reference_time, tolerance)
        accepted = False
        interval_index: Optional[int] = None
        if query_index is not None and reference_index is not None and reference_index in valid_references[query_index]:
            accepted = True
            positive_events += 1
            for candidate, interval in enumerate(intervals):
                if interval["start_index"] <= query_index <= interval["end_index"]:
                    interval_index = candidate
                    if (candidate not in interval_hits or
                            _event_observation_time(event) < _event_observation_time(interval_hits[candidate])):
                        interval_hits[candidate] = event
                    break
        else:
            false_detections += 1
        accepted_events.append({"query_time": event.query_time, "reference_time": event.reference_time,
                                "accepted": accepted, "interval": interval_index})

    delays: list[float] = []
    for interval_index, event in interval_hits.items():
        interval = intervals[interval_index]
        observed_at = _event_observation_time(event)
        delays.append(max(0.0, observed_at - interval["start_time"]))
    count = len(events)
    metrics = {
        "recall": (len(interval_hits) / len(intervals)) if intervals else 0.0,
        "event_hit_rate": (positive_events / count) if count else 0.0,
        "hit_rate": (positive_events / count) if count else 0.0,
        "precision": (positive_events / count) if count else 0.0,
        "false_detection_rate": (false_detections / count) if count else 0.0,
        "false_detections": false_detections,
        "positive_events": positive_events,
        "events": count,
        "loopable_samples": sum(loopable),
        "detected_intervals": len(interval_hits),
        "total_intervals": len(intervals),
        "detection_delays": delays,
        "mean_detection_delay": statistics.mean(delays) if delays else None,
        "p95_detection_delay": _percentile(delays, 95.0) if delays else None,
    }
    return {"status": "ok" if events else "no_events", "parameters": {
        "max_distance": max_distance, "min_time_gap": min_time_gap,
        "orientation_threshold": orientation_threshold, "max_interval_gap": interval_gap,
        "time_tolerance": tolerance, "median_gt_dt": median_dt},
        "loopable_intervals": intervals, "events": accepted_events, "metrics": metrics}


def _make_interval(poses: list[Pose], indices: list[int]) -> dict[str, Any]:
    return {"start_time": poses[indices[0]].timestamp, "end_time": poses[indices[-1]].timestamp,
            "sample_count": len(indices), "start_index": indices[0], "end_index": indices[-1]}


def _event_observation_time(event: LoopEvent) -> float:
    return event.detection_time if event.detection_time is not None else event.query_time


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, help="ground-truth TUM trajectory")
    parser.add_argument("--events", required=True, help="loop event JSON/CSV/JSONL/log")
    parser.add_argument("--output", help="write JSON report here (stdout by default)")
    parser.add_argument("--max-distance", type=float, default=3.0)
    parser.add_argument("--min-time-gap", type=float, default=10.0)
    parser.add_argument("--orientation-threshold", type=float, default=None,
                        help="optional relative quaternion angle threshold in radians")
    parser.add_argument("--max-interval-gap", type=float, default=None,
                        help="maximum timestamp gap joining adjacent positive samples")
    parser.add_argument("--time-tolerance", type=float, default=None,
                        help="timestamp matching tolerance; defaults to half median GT dt")
    args = parser.parse_args(argv)
    try:
        report = evaluate(read_tum(args.gt), read_events(args.events),
                          max_distance=args.max_distance, min_time_gap=args.min_time_gap,
                          orientation_threshold=args.orientation_threshold,
                          max_interval_gap=args.max_interval_gap, time_tolerance=args.time_tolerance)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())
