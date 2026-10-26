import json
import threading
from datetime import datetime, timedelta, timezone

import tempfile
import unittest
from pathlib import Path

from services.behavior_ap.timing import WallClockLedger


class Clock:
    def __init__(self):
        self.ns = 0
        self.utc = datetime(2026, 10, 23, tzinfo=timezone.utc)

    def __call__(self):
        return self.ns

    def advance(self, seconds):
        self.ns += int(seconds * 1e9)
        self.utc += timedelta(seconds=seconds)


def ledger_at(tmp_path, **kwargs):
    clock = Clock()
    return WallClockLedger(tmp_path, clock=clock, utc_clock=lambda: clock.utc, **kwargs), clock


class BehaviorTimingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def test_nested_exclusive_accounting_and_idle_gaps(self):
        tmp_path = self.path
        ledger, clock = ledger_at(tmp_path, expected_categories=['initialization', 'observation', 'action_execution'])
        clock.advance(1)
        with ledger.span('initialization', task='radio') as outer:
            clock.advance(2)
            with ledger.span('observation') as inner:
                clock.advance(3)
            clock.advance(2)
        clock.advance(1)
        with ledger.span('action_execution'):
            clock.advance(4)
        clock.advance(2)
        summary = ledger.finish()
        assert summary['total_seconds'] == 15
        assert summary['classified_wall_seconds'] == 11
        assert summary['idle_unclassified_seconds'] == 4
        assert summary['by_name']['initialization']['inclusive_seconds'] == 7
        assert summary['by_name']['initialization']['exclusive_seconds'] == 4
        assert summary['by_name']['observation']['exclusive_seconds'] == 3
        assert summary['by_name']['action_execution']['exclusive_seconds'] == 4
        assert summary['coverage']['all_expected_categories_observed']
        assert summary['accounting']['exclusive_plus_idle_equals_total']
        events = [json.loads(line) for line in ledger.events_path.read_text().splitlines()]
        starts = {row['span_id']: row for row in events if row['event'] == 'span_started'}
        assert starts[inner]['parent_id'] == outer
        assert starts[outer]['parent_id'] is None
        assert all(row['timestamp_utc'].endswith('Z') for row in events)
        assert json.loads(ledger.summary_path.read_text()) == summary


    def test_exception_is_measured_and_not_swallowed(self):
        tmp_path = self.path
        ledger, clock = ledger_at(tmp_path)
        with self.assertRaisesRegex(RuntimeError, 'failed action'):
            with ledger.span('action_execution'):
                clock.advance(5)
                raise RuntimeError('failed action')
        summary = ledger.finish()
        assert summary['by_name']['action_execution']['error_count'] == 1
        assert summary['by_name']['action_execution']['exclusive_seconds'] == 5
        events = [json.loads(line) for line in ledger.events_path.read_text().splitlines()]
        end = next(row for row in events if row['event'] == 'span_finished')
        assert end['status'] == 'error'
        assert end['error']['type'] == 'RuntimeError'
        assert end['duration_ns'] == 5_000_000_000


    def test_background_video_diagnostics_never_add_to_wall_total(self):
        tmp_path = self.path
        ledger, clock = ledger_at(tmp_path)
        with ledger.span('action_execution'):
            clock.advance(6)
        with ledger.span('video_finalize'):
            clock.advance(2)
        extra = {'background_encoder_seconds': 900, 'queued_frames': 180}
        summary = ledger.finish(extra)
        assert summary['total_seconds'] == 8
        assert summary['classified_wall_seconds'] == 8
        assert summary['diagnostics'] == extra
        assert summary['accounting']['background_diagnostics_included_in_totals'] is False
        assert summary['wall_clock_not_simulation_time'] is True


    def test_recursive_same_name_inclusive_union(self):
        tmp_path = self.path
        ledger, clock = ledger_at(tmp_path)
        with ledger.span('observation'):
            clock.advance(1)
            with ledger.span('observation'):
                clock.advance(2)
            clock.advance(3)
        summary = ledger.finish()
        assert summary['by_name']['observation']['count'] == 2
        assert summary['by_name']['observation']['inclusive_seconds'] == 6
        assert summary['by_name']['observation']['exclusive_seconds'] == 6
        assert summary['accounting']['exclusive_plus_idle_equals_total']


    def test_events_do_not_turn_idle_time_into_spans(self):
        tmp_path = self.path
        ledger, clock = ledger_at(tmp_path, expected_categories=['action_execution'])
        clock.advance(2)
        event = ledger.emit('action_submitted', sequence_id=7)
        clock.advance(3)
        ledger.emit('action_ack', sequence_id=7)
        summary = ledger.finish()
        assert event['metadata']['sequence_id'] == 7
        assert event['parent_id'] is None
        assert summary['total_seconds'] == 5
        assert summary['idle_unclassified_seconds'] == 5
        assert summary['coverage']['missing_expected_categories'] == ['action_execution']


    def test_snapshot_inside_span_and_finish_requires_closed_spans(self):
        tmp_path = self.path
        ledger, clock = ledger_at(tmp_path)
        with ledger.span('agent') as span_id:
            clock.advance(2)
            event = ledger.emit('action_submitted', sequence_id=1)
            assert event['parent_id'] == span_id
            snapshot = ledger.summarize({'background_seconds': 100})
            assert snapshot['coverage']['open_span_count'] == 1
            assert snapshot['by_name']['agent']['running_count'] == 1
            assert snapshot['total_seconds'] == 2
            with self.assertRaisesRegex(RuntimeError, 'spans are open'):
                ledger.finish()
            clock.advance(1)
        final = ledger.finish()
        clock.advance(99)
        assert ledger.finish() == final
        assert ledger.summarize()['total_seconds'] == 3
        with self.assertRaisesRegex(RuntimeError, 'already finished'):
            ledger.emit('late')
        with self.assertRaisesRegex(RuntimeError, 'already finished'):
            with ledger.span('late'):
                pass


    def test_utc_clock_adjustment_does_not_change_duration(self):
        tmp_path = self.path
        ledger, clock = ledger_at(tmp_path)
        with ledger.span('initialization'):
            clock.advance(2)
            clock.utc -= timedelta(days=1)
        assert ledger.finish()['total_seconds'] == 2


    def test_existing_evidence_is_never_overwritten(self):
        tmp_path = self.path
        ledger, _ = ledger_at(tmp_path)
        original = ledger.events_path.read_bytes()
        with self.assertRaises(FileExistsError):
            ledger_at(tmp_path)
        assert ledger.events_path.read_bytes() == original


    def test_background_thread_cannot_create_foreground_span(self):
        tmp_path = self.path
        ledger, clock = ledger_at(tmp_path)
        errors = []
        def work():
            try:
                with ledger.span('encoder'):
                    pass
            except RuntimeError as exc:
                errors.append(str(exc))
        worker = threading.Thread(target=work)
        worker.start()
        worker.join()
        clock.advance(1)
        assert len(errors) == 1 and 'owner thread' in errors[0]
        assert ledger.finish()['idle_unclassified_seconds'] == 1


    def test_invalid_metadata_or_backward_clock_is_rejected(self):
        tmp_path = self.path
        ledger, clock = ledger_at(tmp_path)
        with self.assertRaises(ValueError):
            ledger.emit('invalid', value=float('nan'))
        clock.ns = -1
        with self.assertRaisesRegex(ValueError, 'backwards'):
            ledger.emit('invalid')

if __name__ == "__main__":
    unittest.main()
