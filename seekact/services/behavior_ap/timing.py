"""Auditable foreground wall-clock timing, independent of simulator time.

Use nested spans on the thread which created the ledger. Background encoder
timings belong in ``summarize(extra=...)`` diagnostics, not in foreground spans.
The inclusive category times overlap by definition; only exclusive category
times plus unclassified gaps partition the measured wall-clock total.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable, Iterable


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime | str) -> str:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(instant, datetime) or instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("utc_clock must return a timezone-aware datetime or ISO timestamp")
    return instant.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json_copy(value):
    """Validate metadata before starting a measured action and isolate mutation."""
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


def _union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    intervals = sorted((start, end) for start, end in intervals if end > start)
    if not intervals:
        return 0
    total, first, last = 0, intervals[0][0], intervals[0][1]
    for start, end in intervals[1:]:
        if start > last:
            total += last - first
            first, last = start, end
        else:
            last = max(last, end)
    return total + last - first


class WallClockLedger:
    """Persist real elapsed time and nested foreground spans as JSON evidence.

    ``clock`` returns monotonic integer nanoseconds (normally perf_counter_ns).
    ``utc_clock`` labels events but is never used to compute durations. Existing
    ledger files are not overwritten, so interrupted runs retain their evidence.
    ``expected_categories`` is the caller's coverage contract; missing names are
    reported without turning timing gaps into fabricated measurements.
    """

    def __init__(self, output_dir, *, clock: Callable[[], int] = time.perf_counter_ns,
                 utc_clock: Callable[[], datetime | str] = _utc_now,
                 expected_categories: Iterable[str] = ()):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "timing_events.jsonl"
        self.summary_path = self.output_dir / "wall_timing.json"
        if self.events_path.exists() or self.summary_path.exists():
            raise FileExistsError("Timing evidence already exists; use a fresh run directory")
        self._clock, self._utc_clock = clock, utc_clock
        self._lock = threading.RLock()
        self._owner = threading.get_ident()
        self._last_ns = None
        self._spans = {}
        self._stack = []
        self._event_count = 0
        self._finished_ns = None
        self._finished_utc = None
        self._finished_summary = None
        names = list(expected_categories)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Expected category names must be nonempty strings")
        self.expected_categories = sorted(set(names))
        self._started_ns, self._started_utc = self._stamp()
        # Exclusive creation avoids mixing evidence even if two runners race.
        with self.events_path.open("x", encoding="utf-8"):
            pass
        self._write_event("ledger_started", self._started_ns, self._started_utc,
                          expected_categories=self.expected_categories,
                          wall_clock_not_simulation_time=True)

    def _stamp(self):
        now = self._clock()
        if type(now) is not int:
            raise TypeError("Monotonic clock must return integer nanoseconds")
        if self._last_ns is not None and now < self._last_ns:
            raise ValueError("Monotonic clock moved backwards")
        self._last_ns = now
        return now, _utc_text(self._utc_clock())

    def _require_open(self):
        if self._finished_ns is not None:
            raise RuntimeError("Timing ledger is already finished")

    def _require_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("Foreground timing spans must run on the ledger owner thread; use extra diagnostics for background work")

    def _write_event(self, event, now, utc, **fields):
        self._event_count += 1
        row = {"event_index": self._event_count, "event": event,
               "perf_counter_ns": now, "timestamp_utc": utc,
               "elapsed_ns": now - self._started_ns, **fields}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        return deepcopy(row)

    def emit(self, event: str, **metadata):
        """Record a submitted/ACK/etc. event without inventing a timed span."""
        if not isinstance(event, str) or not event.strip():
            raise ValueError("Event name must be a nonempty string")
        metadata = _json_copy(metadata)
        with self._lock:
            self._require_open()
            now, utc = self._stamp()
            parent = self._stack[-1] if self._stack and threading.get_ident() == self._owner else None
            return self._write_event(event, now, utc, parent_id=parent, metadata=metadata)

    @contextmanager
    def span(self, name: str, **metadata):
        """Measure a synchronous nested span; exceptions are recorded and raised."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Span name must be a nonempty string")
        metadata = _json_copy(metadata)
        with self._lock:
            self._require_open()
            self._require_owner()
            now, utc = self._stamp()
            span_id = f"span-{len(self._spans) + 1:06d}"
            parent = self._stack[-1] if self._stack else None
            record = {"span_id": span_id, "parent_id": parent, "name": name,
                      "start_ns": now, "start_utc": utc, "end_ns": None,
                      "end_utc": None, "duration_ns": None, "status": "running",
                      "metadata": metadata}
            self._spans[span_id] = record
            self._stack.append(span_id)
            self._write_event("span_started", now, utc, **deepcopy(record))
        error = None
        try:
            yield span_id
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            raise
        finally:
            with self._lock:
                self._require_owner()
                if not self._stack or self._stack[-1] != span_id:
                    raise RuntimeError("Timing spans must exit in last-in-first-out order")
                now, utc = self._stamp()
                self._stack.pop()
                record.update(end_ns=now, end_utc=utc, duration_ns=now - record["start_ns"],
                              status="error" if error else "ok")
                if error:
                    record["error"] = error
                self._write_event("span_finished", now, utc, **deepcopy(record))

    def _summary(self, now, utc, extra):
        spans = list(self._spans.values())
        intervals = {span["span_id"]: (span["start_ns"], span["end_ns"] if span["end_ns"] is not None else now)
                     for span in spans}
        children_by_parent = {}
        intervals_by_name = {}
        for span in spans:
            interval = intervals[span["span_id"]]
            children_by_parent.setdefault(span["parent_id"], []).append(interval)
            intervals_by_name.setdefault(span["name"], []).append(interval)
        by_name = {}
        for span in spans:
            name = span["name"]
            row = by_name.setdefault(name, {"count": 0, "completed_count": 0, "error_count": 0,
                                           "running_count": 0, "inclusive_ns": 0, "exclusive_ns": 0})
            row["count"] += 1
            row["completed_count"] += span["status"] != "running"
            row["error_count"] += span["status"] == "error"
            row["running_count"] += span["status"] == "running"
            start, end = intervals[span["span_id"]]
            children = children_by_parent.get(span["span_id"], [])
            row["exclusive_ns"] += end - start - _union_ns(children)
        for name, row in by_name.items():
            # A recursively nested same-name category is counted once, not twice.
            row["inclusive_ns"] = _union_ns(intervals_by_name[name])
            row["inclusive_seconds"] = row["inclusive_ns"] / 1e9
            row["exclusive_seconds"] = row["exclusive_ns"] / 1e9
        total = now - self._started_ns
        covered = _union_ns(intervals.values())
        exclusive_total = sum(row["exclusive_ns"] for row in by_name.values())
        missing = sorted(set(self.expected_categories) - set(by_name))
        return {
            "schema_version": 1,
            "wall_clock_not_simulation_time": True,
            "duration_clock": "time.perf_counter_ns" if self._clock is time.perf_counter_ns else "injected_monotonic_nanoseconds",
            "timestamp_clock": "UTC labels only; not used for durations",
            "start_ns": self._started_ns, "end_ns": now,
            "start_utc": self._started_utc, "end_utc": utc,
            "total_ns": total, "total_seconds": total / 1e9,
            "classified_wall_ns": covered, "classified_wall_seconds": covered / 1e9,
            "idle_unclassified_ns": total - covered,
            "idle_unclassified_seconds": (total - covered) / 1e9,
            "by_name": dict(sorted(by_name.items())),
            "accounting": {
                "inclusive_categories_overlap_do_not_sum": True,
                "exclusive_category_total_ns": exclusive_total,
                "exclusive_plus_idle_equals_total": exclusive_total + total - covered == total,
                "background_diagnostics_included_in_totals": False,
            },
            "coverage": {"expected_categories": self.expected_categories,
                         "observed_categories": sorted(by_name), "missing_expected_categories": missing,
                         "all_expected_categories_observed": not missing,
                         "open_span_count": len(self._stack)},
            "finished": self._finished_ns is not None,
            "event_count": self._event_count,
            "diagnostics": extra,
        }

    def summarize(self, extra=None):
        """Return a snapshot. ``extra`` is diagnostics and never enters totals."""
        extra = _json_copy({} if extra is None else extra)
        with self._lock:
            if self._finished_ns is None:
                now, utc = self._stamp()
            else:
                now, utc = self._finished_ns, self._finished_utc
            return self._summary(now, utc, extra)

    def finish(self, extra=None):
        """Freeze elapsed time and atomically persist wall_timing.json once."""
        extra = _json_copy({} if extra is None else extra)
        with self._lock:
            if self._finished_summary is not None:
                return deepcopy(self._finished_summary)
            if self._stack:
                raise RuntimeError("Cannot finish timing while foreground spans are open")
            now, utc = self._stamp()
            self._finished_ns, self._finished_utc = now, utc
            self._write_event("ledger_finished", now, utc, total_ns=now - self._started_ns,
                              wall_clock_not_simulation_time=True)
            summary = self._summary(now, utc, extra)
            temporary = self.summary_path.with_name(self.summary_path.name + ".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.summary_path)
            self._finished_summary = summary
            return deepcopy(summary)
