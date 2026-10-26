"""CPU-only conditional-stage reporting; no expert contents or simulation."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts import report_behavior_experiment as report


class BehaviorReportRetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = {'task_name': 'turning_on_radio', 'task_index': 0,
                     'fixed_expert': {'episode_index': 0, 'raw_episode_id': 10, 'frames': 1956}}
        (self.root / 'protocol.json').write_text(json.dumps({'tasks': [self.task]}))

    def attempt(self, phase='zero_shot', version=1, *, status='task_success',
                valid=True, success=True, sealed=True):
        directory = self.root / self.task['task_name'] / f'{phase}_v{version}'
        directory.mkdir(parents=True)
        value = {'task': self.task['task_name'], 'phase': phase, 'attempt': f'v{version}',
                 'started_at_utc': f'2026-01-01T00:00:0{version}+00:00',
                 'status': status, 'evaluation_valid': valid, 'official_success': success,
                 'steps': 0, 'agent_exit_code': 0, 'worker_exit_code': 0}
        filename = 'trial_record.json' if sealed else 'trial_started.json'
        (directory / filename).write_text(json.dumps(value))
        if sealed:
            (directory / 'episode_closed.json').write_text(json.dumps({
                'status': 'closed', 'official_success': success, 'steps': 0}))
        return directory

    def build(self):
        result = report.build_report(self.root)
        return result, result['tasks'][0]

    def test_latest_valid_success_without_retry_is_not_required(self):
        self.attempt(status='infrastructure_error', valid=False, success=False)
        self.attempt(version=2)
        result, task = self.build()
        retry = task['phases']['one_expert_retry']
        self.assertEqual(retry['status'], 'not_required')
        self.assertEqual(retry['attempts'], 0)
        self.assertEqual(retry['basis_directory'], 'turning_on_radio/zero_shot_v2')
        self.assertEqual(result['not_run_task_phases'], 0)
        self.assertEqual(result['not_required_task_phases'], 1)
        for key in ('export', 'analysis'):
            self.assertEqual(task['expert_stages'][key]['status'], 'not_required')
        self.assertIn('零样本成功，无需专家重试', report.html_report(result))

    def test_no_zero_or_valid_failure_still_needs_retry(self):
        result, task = self.build()
        self.assertEqual(result['not_run_task_phases'], 2)
        self.attempt(status='task_failure', success=False)
        result, task = self.build()
        self.assertEqual(task['phases']['one_expert_retry']['status'], 'not_run')
        self.assertEqual(task['expert_stages']['export']['status'], 'pending')

    def test_newer_unsealed_infrastructure_or_invalid_success_blocks_skip(self):
        self.attempt()
        cases = ({'sealed': False}, {'status': 'infrastructure_error', 'valid': False},
                 {'valid': False}, {'success': False})
        for index, changes in enumerate(cases, 2):
            with self.subTest(changes=changes):
                self.attempt(version=index, **changes)
                result, task = self.build()
                self.assertEqual(task['phases']['one_expert_retry']['status'], 'not_run')
                self.assertEqual(task['expert_stages']['analysis']['status'], 'pending')
                self.assertEqual(result['not_required_task_phases'], 0)

    def test_existing_retry_is_preserved_even_after_zero_success(self):
        self.attempt()
        self.attempt(phase='one_expert_retry', status='infrastructure_error', valid=False)
        _, task = self.build()
        self.assertEqual(task['phases']['one_expert_retry']['status'], 'infrastructure_error')

    def test_missing_or_conflicting_closed_success_does_not_skip_retry(self):
        directory = self.attempt()
        path = directory / 'episode_closed.json'
        path.unlink()
        _, task = self.build()
        self.assertEqual(task['phases']['one_expert_retry']['status'], 'not_run')
        for changes in ({'official_success': False}, {'steps': 1}, {'status': 'closing'}):
            with self.subTest(changes=changes):
                path.write_text(json.dumps({'status': 'closed', 'official_success': True, 'steps': 0, **changes}))
                _, task = self.build()
                self.assertEqual(task['phases']['one_expert_retry']['status'], 'not_run')

    def test_queue_time_unknown_and_unsealed_video_not_claimed_complete(self):
        self.attempt(sealed=False)
        result, task = self.build()
        self.assertEqual(result['runner_queue_time']['status'], 'not_recorded')
        self.assertIsNone(result['runner_queue_time']['wall_seconds'])
        page = report.html_report(result)
        self.assertIn('等待 runner.lock', page)
        self.assertIn('不能记作零', page)
        self.assertIn('尚未封存', page)
        self.assertIn('执行录像（尚未完整验收）', page)
        self.assertNotIn('机器人相机全过程', page)

    def test_partial_or_invalid_expert_evidence_is_not_replaced(self):
        self.attempt()
        directory = self.root / 'turning_on_radio/expert_episode_0'
        directory.mkdir()
        (directory / 'export_error.json').write_text('{"status":"failed"}')
        _, task = self.build()
        self.assertEqual(task['phases']['one_expert_retry']['status'], 'not_required')
        self.assertEqual(task['expert_stages']['export']['status'], 'pending')
        (directory / 'export_complete.json').write_text('broken JSON')
        (directory / 'experience_analysis.json').write_text('{"status":"failed"}')
        _, task = self.build()
        self.assertEqual(task['expert_stages']['export']['status'], 'unknown')
        self.assertEqual(task['expert_stages']['analysis']['status'], 'unknown')
        self.assertTrue(task['expert_stages']['export']['reasons'])
        self.assertTrue(task['expert_stages']['analysis']['reasons'])

    def test_live_snapshot_links_are_labeled_as_unsealed_measurement(self):
        directory = self.attempt(sealed=False)
        path = directory / 'live_time_breakdown_20260101T010000Z.json'
        path.write_text(json.dumps({'status': 'live_snapshot_unsealed',
            'cutoff_at_utc': '2026-01-01T01:00:00Z', 'cutoff_Asia_Shanghai': '2026-01-01T09:00:00+08:00'}))
        path.with_suffix('.md').write_text('CPU timing fixture.')
        result, task = self.build()
        self.assertEqual(task['phases']['zero_shot']['status'], 'unsealed')
        live = task['attempts'][0]['latest_live_time_snapshot']
        self.assertTrue(live['not_final_measurement'])
        self.assertEqual(live['json'], path.relative_to(self.root).as_posix())
        self.assertIn('不是最终计时', report.html_report(result))

    def test_model_profile_uses_session_metadata_and_does_not_fill_old_defaults(self):
        directory = self.attempt()
        (directory / 'agent').mkdir()
        (directory / 'agent_config.json').write_text('{"agent_effort":"xhigh"}')
        prepared = directory / 'agent/behavior-agent-prepared.json'
        prepared.write_text('{"model_selection":"gpt-6-astra"}')
        result, task = self.build()
        profile = task['attempts'][0]['model_profile']
        self.assertIsNone(profile['model_reasoning_effort'])
        self.assertIsNone(profile['model_service_tier_requested'])
        self.assertFalse(profile['server_service_tier_verified'])
        self.assertIn('未显式指定', report.html_report(result))
        prepared.write_text(json.dumps({'model_selection': 'gpt-6-astra', 'model_reasoning_effort': 'low',
            'model_service_tier_requested': 'fast', 'server_service_tier_verified': False}))
        result, task = self.build()
        self.assertEqual(task['attempts'][0]['model_profile']['model_reasoning_effort'], 'low')
        self.assertEqual(task['attempts'][0]['model_profile']['model_service_tier_requested'], 'fast')
        self.assertIn('服务端服务层未证实', report.html_report(result))

    def test_started_phase_profiles_disclose_differences_unknowns_and_unsealed_results(self):
        zero = self.attempt(status='task_failure', success=False)
        retry = self.attempt(phase='one_expert_retry', sealed=False)
        common = {'model_selection': 'gpt-6-astra', 'model_reasoning_effort': 'low',
                  'model_service_tier_requested': 'fast', 'server_service_tier_verified': True}
        for directory in (zero, retry):
            (directory / 'agent').mkdir()
            (directory / 'processes.json').write_text('{"agent_pid":123}')
        (retry / 'agent/behavior-agent-prepared.json').write_text(json.dumps(common))
        cases = ((dict(common, model_reasoning_effort='high'), 'different'),
                 ({'model_selection': 'gpt-6-astra'}, 'unknown'),
                 (dict(common), 'matching_recorded_config'))
        for profile, expected in cases:
            with self.subTest(expected=expected):
                (zero / 'agent/behavior-agent-prepared.json').write_text(json.dumps(profile))
                result, task = self.build()
                comparison = task['model_profile_comparison']
                self.assertEqual(comparison['status'], expected)
                self.assertIn('尚未封存', comparison['message'])
                self.assertIn('两阶段可比性', report.html_report(result))
                if expected == 'different':
                    self.assertEqual(comparison['differing_fields'], ['model_reasoning_effort'])
                    self.assertIn('不能把结果或耗时差异单独归因于专家经验', comparison['message'])
                elif expected == 'unknown':
                    self.assertIn('model_reasoning_effort', comparison['unknown_fields'])
                    self.assertIn('server_service_tier', comparison['unknown_fields'])
                    self.assertIn('不能视为一致', comparison['message'])
        # Equal requested settings still do not verify the effective server tier.
        (zero / 'agent/behavior-agent-prepared.json').write_text(json.dumps(
            dict(common, server_service_tier_verified=False)))
        _, task = self.build()
        self.assertEqual(task['model_profile_comparison']['status'], 'unknown')

    def test_missing_or_prepared_only_retry_does_not_claim_configuration_has_run(self):
        zero = self.attempt(status='task_failure', success=False)
        (zero / 'processes.json').write_text('{"agent_pid":123}')
        _, task = self.build()
        self.assertEqual(task['model_profile_comparison']['status'], 'pending')
        retry = self.attempt(phase='one_expert_retry', sealed=False)
        (retry / 'agent').mkdir()
        (retry / 'agent/behavior-agent-prepared.json').write_text(json.dumps({
            'model_selection': 'gpt-6-astra', 'model_reasoning_effort': 'low',
            'model_service_tier_requested': 'fast'}))
        result, task = self.build()
        comparison = task['model_profile_comparison']
        self.assertEqual(comparison['status'], 'pending')
        self.assertIn('尚无单专家经验重试的实际会话启动证据', comparison['message'])
        self.assertIn('不将当前默认配置或准备记录视为已经切换运行', report.html_report(result))


if __name__ == '__main__':
    unittest.main()
