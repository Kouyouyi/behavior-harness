"""Known CPU-only interval fixtures; no Agent, demonstration or simulator runs."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import report_behavior_experiment as report
from services.behavior_ap.timing import WallClockLedger


class BehaviorReportTimingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.origin = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def fixture(self, with_tools=True):
        clock = [0]
        ledger = WallClockLedger(self.root / 'worker_timing', clock=lambda: clock[0],
            utc_clock=lambda: self.origin + timedelta(seconds=clock[0] / 1e9),
            expected_categories=('gpu_wait', 'initialization', 'observation', 'action_execution', 'video_finalize'))
        def at(seconds):
            clock[0] = int(seconds * 1e9)
        at(1)
        with ledger.span('gpu_wait'):
            at(2)
        with ledger.span('initialization'):
            at(3)
        if with_tools:
            # Agent lives [5,20]. Clipped tools are [5,8], [10,16], [19,20].
            with ledger.span('tool_dispatch'):
                at(6)
                with ledger.span('observation'):
                    at(7)
                at(8)
            at(10)
            with ledger.span('tool_dispatch'):
                at(11)
                with ledger.span('action_execution'):
                    at(15)
                at(16)
            at(19)
            with ledger.span('tool_dispatch'):
                at(24)
        at(25)
        with ledger.span('video_finalize'):
            at(26)
        at(30)
        worker = ledger.finish()
        # Synthetic clock values deliberately follow the real evidence schema.
        worker['duration_clock'] = 'time.perf_counter_ns'
        agent = {'status': 'agent_completed', 'exit_code': 0,
                 'started_perf_counter_ns': 5_000_000_000, 'ended_perf_counter_ns': 20_000_000_000,
                 'agent_wall_seconds': 15,
                 'started_at_utc': (self.origin + timedelta(seconds=5)).isoformat(),
                 'ended_at_utc': (self.origin + timedelta(seconds=20)).isoformat()}
        events = report.load_timing_events(self.root / 'worker_timing/timing_events.jsonl')
        return agent, worker, events

    def analyze(self, agent, worker, events, **kwargs):
        return report.analyze_agent_tool_time(agent, worker, events,
            sealed=kwargs.pop('sealed', True), normal_close=kwargs.pop('normal_close', True))

    def test_known_intersection_union_and_residual_in_exact_nanoseconds(self):
        agent, worker, events = self.fixture()
        result = self.analyze(agent, worker, events)
        self.assertTrue(result['valid'], result['reasons'])
        self.assertEqual(result['tool_dispatch_intersection_union_ns'], 10_000_000_000)
        self.assertEqual(result['agent_residual_ns'], 5_000_000_000)
        self.assertEqual(result['observation_intersection_union_ns'], 1_000_000_000)
        self.assertEqual(result['action_execution_intersection_union_ns'], 4_000_000_000)
        self.assertTrue(result['accounting_exact_in_nanoseconds'])
        self.assertEqual(result['agent_wall_seconds'], 15)
        self.assertIn('不是纯 LLM', result['residual_scope'])
        self.assertEqual(report.interval_union_ns([(0, 4), (1, 2), (3, 5), (8, 9)]), 6)

    def test_pending_and_abnormal_close_do_not_publish_derived_numbers(self):
        agent, worker, events = self.fixture()
        for sealed, normal, expected in ((False, False, 'pending'), (True, False, 'unknown')):
            result = self.analyze(agent, worker, events, sealed=sealed, normal_close=normal)
            self.assertEqual(result['status'], expected)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['agent_residual_ns'])
            self.assertIsNone(result['tool_dispatch_intersection_union_seconds'])

    def test_missing_truncated_unpaired_or_inconsistent_events_are_unknown(self):
        agent, worker, events = self.fixture()
        broken = deepcopy(events)
        finished = next(row for row in broken if row.get('event') == 'span_finished' and row.get('name') == 'observation')
        finished['span_id'] = 'absent-parent'
        cases = [(None, events), (worker, None), (worker, events[:-1]), (worker, broken)]
        changed = deepcopy(worker); changed['by_name']['tool_dispatch']['count'] += 1
        cases.append((changed, events))
        for ledger, rows in cases:
            with self.subTest(ledger=ledger is not None, rows=len(rows or [])):
                result = self.analyze(agent, ledger, rows)
                self.assertEqual(result['status'], 'unknown')
                self.assertIsNone(result['agent_residual_seconds'])
        path = self.root / 'truncated.jsonl'
        path.write_bytes(b'{"event":"ledger_finished"}')
        self.assertIsNone(report.load_timing_events(path))

    def test_cross_clock_bad_duration_or_uncontained_agent_is_unknown(self):
        agent, worker, events = self.fixture()
        shifted_utc = dict(agent, started_at_utc=(self.origin + timedelta(seconds=105)).isoformat())
        wrong_wall = dict(agent, agent_wall_seconds=14)
        outside = dict(agent, started_perf_counter_ns=31_000_000_000, ended_perf_counter_ns=46_000_000_000)
        for candidate in (shifted_utc, wrong_wall, outside):
            result = self.analyze(candidate, worker, events)
            self.assertEqual(result['status'], 'unknown')
            self.assertIsNone(result['tool_dispatch_intersection_union_ns'])
        wrong_source = dict(worker, duration_clock='remote_different_clock')
        self.assertEqual(self.analyze(agent, wrong_source, events)['status'], 'unknown')

    def test_complete_empty_tool_coverage_is_zero_not_unknown(self):
        agent, worker, events = self.fixture(with_tools=False)
        result = self.analyze(agent, worker, events)
        self.assertTrue(result['valid'], result['reasons'])
        self.assertEqual(result['tool_dispatch_intersection_union_seconds'], 0)
        self.assertEqual(result['observation_intersection_union_seconds'], 0)
        self.assertEqual(result['agent_residual_seconds'], 15)

    def test_report_uses_recorded_total_and_separate_tool_diagnostics(self):
        agent, worker, events = self.fixture()
        output = self.root / 'report'
        trial = output / 'turning_on_radio/zero_shot_v1'
        (trial / 'worker_timing').mkdir(parents=True)
        (trial / 'agent').mkdir()
        (trial / 'supervisor_timing').mkdir()
        protocol = {'tasks': [{'task_name': 'turning_on_radio', 'task_index': 0, 'prompt': 'Turn on radio.'}],
                    'variant': 'original', 'render_profile': 'fast', 'split': 'public_test', 'instance_id': 301}
        (output / 'protocol.json').write_text(json.dumps(protocol))
        (trial / 'worker_timing/wall_timing.json').write_text(json.dumps(worker))
        (trial / 'worker_timing/timing_events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events))
        (trial / 'agent/agent_timing.json').write_text(json.dumps(agent))
        (trial / 'supervisor_timing/wall_timing.json').write_text(json.dumps({'total_seconds': 100, 'by_name': {}}))
        closed = {'status': 'closed', 'steps': 0, 'official_success': False,
                  'videos': {'passed': True, 'expected_frames': 0, 'status': 'no_frames', 'videos': {}}}
        (trial / 'episode_closed.json').write_text(json.dumps(closed))
        record = {'task': 'turning_on_radio', 'phase': 'zero_shot', 'attempt': 'v1',
                  'status': 'task_failure', 'evaluation_valid': True, 'official_success': False,
                  'steps': 0, 'agent_exit_code': 0, 'worker_exit_code': 0}
        (trial / 'trial_record.json').write_text(json.dumps(record))
        result = report.write_report(output)
        row = result['tasks'][0]['attempts'][0]
        self.assertEqual(row['timing']['supervisor_total_seconds'], 100)
        self.assertTrue(row['agent_timing_analysis']['valid'])
        self.assertEqual(row['agent_timing_analysis']['agent_residual_seconds'], 5)
        page = (output / 'index.html').read_text()
        self.assertIn('工具执行墙钟（含子跨度）', page)
        self.assertIn('不能再与工具并集相加', page)
        self.assertIn('30fps', page)
        self.assertIn('视频时长不是实际实验用时', page)
        self.assertNotIn('0.00 秒</strong>', page)
        # A running replacement keeps pending instead of publishing final residuals.
        pending = output / 'turning_on_radio/zero_shot_v2'
        pending.mkdir()
        (pending / 'trial_started.json').write_text(json.dumps({'task': 'turning_on_radio',
            'phase': 'zero_shot', 'attempt': 'v2', 'started_at_utc': '2099-01-01T00:00:00Z'}))
        refreshed = report.write_report(output)
        latest = refreshed['tasks'][0]['attempts'][-1]
        self.assertEqual(latest['status'], 'unsealed')
        self.assertEqual(latest['agent_timing_analysis']['status'], 'pending')

    def expert_fixture(self):
        task = {'task_name': 'putting_shoes_on_rack', 'fixed_expert': {
            'episode_index': 4400, 'raw_episode_id': 220020, 'frames': 7076}}
        directory = self.root / 'putting_shoes_on_rack/expert_episode_4400'
        directory.mkdir(parents=True)
        experience = directory / 'experience.md'
        experience.write_text('Expert contents must not be read by the reporter.')
        exported = {'status': 'export_verified', 'source_count': 1,
            'task_name': task['task_name'], 'fixed_expert': task['fixed_expert'],
            'wall_timing': {'finished': True, 'duration_clock': 'time.perf_counter_ns',
                'start_ns': 1_000_000_000, 'end_ns': 7_000_000_000, 'total_ns': 6_000_000_000,
                'total_seconds': 6, 'by_name': {name: {'exclusive_seconds': value} for name, value in
                    [('expert_export', 1), ('expert_transfer', 2), ('expert_verification', 3)]}},
            'remote_manifest': {'files': {name: {'sha256': 'a' * 64} for name in
                ('episode.parquet', 'annotation.json', 'videos/head.mp4',
                 'videos/left_wrist.mp4', 'videos/right_wrist.mp4')}}}
        text = json.dumps(exported)
        (directory / 'export_complete.json').write_text(text)
        analyzed = {'status': 'complete', 'source_count': 1, 'episode_index': 4400,
            'started_perf_counter_ns': 10_000_000_000, 'ended_perf_counter_ns': 14_000_000_000,
            'analysis_wall_seconds': 4, 'experience_path': str(experience),
            'experience_sha256': 'b' * 64,
            'export_complete_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'scope': 'Elapsed wall time including tool and orchestration waiting.'}
        return task, directory, analyzed

    def test_expert_stages_pending_without_completion_markers(self):
        task = {'task_name': 'putting_shoes_on_rack', 'fixed_expert': {'episode_index': 4400}}
        result = report.expert_stages(self.root, task)
        self.assertEqual(result['export']['status'], 'pending')
        self.assertEqual(result['analysis']['status'], 'pending')
        self.assertIsNone(result['analysis']['wall_seconds'])
        task, directory, _ = self.expert_fixture()
        result = report.expert_stages(self.root, task)
        self.assertEqual(result['export']['status'], 'complete')
        self.assertEqual(result['analysis']['status'], 'pending')
        self.assertIsNone(result['experience_link'])

    def test_expert_metadata_only_hash_links_and_independent_wall_times(self):
        task, directory, analyzed = self.expert_fixture()
        (directory / 'experience_analysis.json').write_text(json.dumps(analyzed))
        original_text, original_bytes = Path.read_text, Path.read_bytes
        allowed = {'export_complete.json', 'experience_analysis.json'}
        def text(path, *args, **kwargs):
            self.assertIn(path.name, allowed, 'Reporter tried reading expert contents')
            return original_text(path, *args, **kwargs)
        def raw(path, *args, **kwargs):
            self.assertIn(path.name, allowed, 'Reporter tried hashing expert contents')
            return original_bytes(path, *args, **kwargs)
        with patch.object(Path, 'read_text', text), patch.object(Path, 'read_bytes', raw):
            result = report.expert_stages(self.root, task)
        self.assertEqual(result['export']['status'], 'complete')
        self.assertEqual(result['analysis']['status'], 'complete')
        self.assertEqual(result['export']['wall_seconds'], 6)
        self.assertEqual(result['analysis']['wall_seconds'], 4)
        self.assertEqual(len(result['source_file_sha256']), 5)
        self.assertEqual(result['experience_link'], 'putting_shoes_on_rack/expert_episode_4400/experience.md')
        self.assertFalse(result['expert_content_read'])
        page = report.expert_panel(result)
        self.assertIn('不与零样本或重试墙钟相加', page)
        self.assertIn('不是纯模型推理', page)
        self.assertNotIn('Expert contents must', page)

    def test_expert_wrong_source_hash_clock_or_episode_is_not_complete(self):
        task, directory, analyzed = self.expert_fixture()
        for changes in ({'source_count': 2}, {'episode_index': 0},
                        {'export_complete_sha256': '0' * 64}, {'analysis_wall_seconds': 5}):
            (directory / 'experience_analysis.json').write_text(json.dumps({**analyzed, **changes}))
            result = report.expert_stages(self.root, task)
            self.assertEqual(result['analysis']['status'], 'unknown')

    def queue_fixture(self):
        task = {'task_name': 'putting_shoes_on_rack', 'task_index': 22}
        directory = self.root / task['task_name'] / 'one_expert_retry_v1'
        timing = directory / 'supervisor_timing'
        timing.mkdir(parents=True)
        queued_ns, started_ns = 355123913639519, 368213944441843
        receipt = {'queued_perf_counter_ns': queued_ns, 'queued_at_utc': self.origin.isoformat(),
            'expected_trial': str(directory), 'command': ['python', 'scripts/run_behavior_experiment.py',
                '--phase', 'one_expert_retry', '--task', task['task_name'], '--attempt', 'v1',
                '--protocol', str(self.root / 'protocol.json'), '--wait-for-runner']}
        started = {'event': 'ledger_started', 'perf_counter_ns': started_ns,
            'timestamp_utc': (self.origin + timedelta(seconds=(started_ns-queued_ns)/1e9)).isoformat()}
        receipt_path, events_path = self.root / 'shoes_retry_v1_queue.json', timing / 'timing_events.jsonl'
        receipt_path.write_text(json.dumps(receipt))
        events_path.write_text(json.dumps(started) + '\n')
        return task, directory, receipt_path, events_path, receipt, started

    def test_queue_receipt_duration_is_monotonic_and_separate_from_trial_wall(self):
        task, directory, path, events_path, receipt, started = self.queue_fixture()
        for target_key in ('expected_trial', 'fresh_target'):
            with self.subTest(target_key=target_key):
                payload = dict(receipt)
                payload[target_key] = payload.pop('expected_trial')
                path.write_text(json.dumps(payload))
                measured = report.queue_to_ledger_time(directory, self.root, task, 'one_expert_retry')
                self.assertEqual(measured['status'], 'measured')
                self.assertEqual(measured['duration_ns'], 13090030802324)
                self.assertEqual(measured['wall_seconds'], 13090.030802324)
                self.assertFalse(measured['included_in_trial_wall'])
        # The original shoes command omitted --protocol and used the CLI default.
        payload = deepcopy(receipt)
        del payload['command'][8:10]
        path.write_text(json.dumps(payload))
        with patch.object(report, 'DEFAULT_OUTPUT', self.root):
            measured = report.queue_to_ledger_time(directory, self.root, task, 'one_expert_retry')
        self.assertEqual(measured['status'], 'measured')
        path.write_text(json.dumps(receipt))
        (directory / 'supervisor_timing/wall_timing.json').write_text('{"total_seconds":11}')
        (directory / 'trial_started.json').write_text(json.dumps({
            'task': task['task_name'], 'phase': 'one_expert_retry', 'attempt': 'v1'}))
        (self.root / 'protocol.json').write_text(json.dumps({'tasks': [task]}))
        result = report.build_report(self.root)
        trial = result['tasks'][0]['attempts'][0]
        self.assertEqual(trial['timing']['supervisor_total_seconds'], 11)
        self.assertEqual(result['runner_queue_time']['status'], 'per_trial_records')
        self.assertIsNone(result['runner_queue_time']['wall_seconds'])
        self.assertIn('不是纯锁等待，未加入上述运行墙钟', report.html_report(result))

    def test_queue_missing_mismatched_or_inconsistent_clock_stays_unknown(self):
        task, directory, path, events_path, receipt, started = self.queue_fixture()
        for label, modify in (
            ('missing_perf', lambda r, e: r.pop('queued_perf_counter_ns')),
            ('wrong_target', lambda r, e: r.update(expected_trial=str(self.root / 'another_trial'))),
            ('wrong_task', lambda r, e: r['command'].__setitem__(5, 'turning_on_radio')),
            ('wrong_phase', lambda r, e: r['command'].__setitem__(3, 'zero_shot')),
            ('wrong_attempt', lambda r, e: r['command'].__setitem__(7, 'v2')),
            ('wrong_protocol_parent', lambda r, e: r['command'].__setitem__(9, str(self.root / 'other/protocol.json'))),
            ('invalid_output_flag', lambda r, e: r['command'].extend(['--output', str(self.root)])),
            ('invalid_output_equals', lambda r, e: r['command'].append('--output=' + str(self.root))),
            ('reversed_perf', lambda r, e: r.update(queued_perf_counter_ns=e['perf_counter_ns'] + 1)),
            ('epoch_mismatch', lambda r, e: r.update(queued_at_utc=(self.origin-timedelta(hours=1)).isoformat())),
            ('missing_utc', lambda r, e: r.pop('queued_at_utc')),
            ('wrong_start_event', lambda r, e: e.update(event='span_started')),
        ):
            with self.subTest(label=label):
                payload, event = deepcopy(receipt), deepcopy(started)
                modify(payload, event)
                path.write_text(json.dumps(payload))
                events_path.write_text(json.dumps(event) + '\n')
                result = report.queue_to_ledger_time(directory, self.root, task, 'one_expert_retry')
                self.assertEqual(result['status'], 'unknown')
                self.assertIsNone(result['wall_seconds'])
        path.unlink()
        result = report.queue_to_ledger_time(directory, self.root, task, 'one_expert_retry')
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['wall_seconds'])


if __name__ == '__main__':
    unittest.main()
