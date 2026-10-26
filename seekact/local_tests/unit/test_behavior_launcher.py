import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness.codex_cli import behavior
from services.backends import BACKENDS
from services.controller.config import Configuration


class BehaviorLauncherTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.dict('os.environ', {'RAX_ARTIFACT_ROOT': str(self.temp)}))
        self.stack.enter_context(patch.dict(BACKENDS, {'behavior_ap': 'services.mock_backend:MockBackend'}))
        self.stack.enter_context(patch.object(behavior, 'inherited_settings',
            return_value=(['unused', 'name.with.dots'], 'configured-default-model')))
        config = Configuration(root=self.temp / 'episode', task='example_task',
                               task_description='Complete the current simulated task and verify official success.',
                               robot='behavior_ap', agent_provider='codex', observation_profile='rgb-only')
        config.root.mkdir()
        path = config.root / 'operator-input.json'
        path.write_text(json.dumps(config.to_dict()))
        self.session = config, path

    def test_prepare_preserves_login_and_disables_unrelated_tools(self):
        config, path = self.session
        plan = behavior.prepare(config, path, '/python', '/codex')
        command = plan['command']
        assert '--json' in command and '--ephemeral' in command
        assert '--sandbox' in command and 'read-only' in command
        assert '--ignore-user-config' in command
        assert command[command.index('--model') + 1] == 'gpt-6-astra'
        assert 'model_reasoning_effort="low"' in command
        assert 'service_tier="fast"' in command and 'features.fast_mode=true' in command
        assert 'features.memories=false' in command
        assert 'features.skip_host_skill_discovery=true' in command
        assert 'features.hooks=false' in command and 'features.shell_tool=false' in command
        assert 'features.plugins=false' in command and 'web_search="disabled"' in command
        assert not any('unused' in value or 'name.with.dots' in value for value in command)
        assert 'mcp_servers.rax_robot.tool_timeout_sec=180' in command
        assert any('services.controller.frontend' in value for value in command)
        allowed = next(value.split('=', 1)[1] for value in command
                       if value.startswith('mcp_servers.rax_robot.enabled_tools='))
        assert json.loads(allowed) == list(behavior.TOOLS)
        assert not any('CODEX_HOME=' in value for value in command)
        assert not any('g1_vla' in value or 'start_g1' in value for value in command)
        assert not (config.root / 'skill-library').exists()
        assert plan['phase'] == 'zero_shot' and plan['experience'] is None
        assert plan['model_selection'] == 'gpt-6-astra'
        assert plan['model_reasoning_effort'] == 'low'
        assert plan['model_service_tier_requested'] == 'fast'
        assert plan['server_service_tier_verified'] is False
        instructions = (config.root / 'agent-workspace/AGENTS.md').read_text()
        assert 'zero-shot attempt' in instructions and 'without asking for ARM' in instructions
        assert '<single_expert_experience>' not in instructions


    def test_prepare_only_can_be_consumed_once_without_rewriting(self):
        config, path = self.session
        first = behavior.prepare(config, path, '/python', '/codex')
        assert behavior.prepare(config, path, '/python', '/codex') == first
        (config.root / 'agent_events.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'execution evidence'):
            behavior.prepare(config, path, '/python', '/codex')


    def test_foreign_or_modified_workspace_is_refused(self):
        config, path = self.session
        behavior.prepare(config, path, '/python', '/codex')
        (config.root / 'agent-workspace/extra_history.md').write_text('not allowed')
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            behavior.prepare(config, path, '/python', '/codex')


    def test_only_explicit_single_experience_is_injected(self):
        config, path = self.session
        source = self.temp / 'one_expert.md'
        source.write_text('An observed useful effect, with applicable conditions and uncertainty.')
        plan = behavior.prepare(config, path, '/python', '/codex', source)
        assert plan['phase'] == 'one_expert_retry'
        assert plan['experience']['sources_permitted'] == 1
        assert len(plan['experience']['sha256']) == 64
        instructions = (config.root / 'agent-workspace/AGENTS.md').read_text()
        assert source.read_text() in instructions
        assert instructions.count('<single_expert_experience>') == 1
        assert not (config.root / 'agent-workspace/skill-library').exists()


    def test_invalid_experience_and_config_rejected(self):
        config, path = self.session
        for name, payload in [('empty.txt', ''), ('large.md', 'a' * (128 * 1024 + 1)), ('video.mp4', 'not text')]:
            source = self.temp / name
            source.write_text(payload)
            with self.assertRaises(ValueError):
                behavior.experience_text(source)
        path.write_text(json.dumps({**config.to_dict(), 'task_description': 'Changed task'}))
        with self.assertRaisesRegex(ValueError, 'differ'):
            behavior.prepare(config, path, '/python', '/codex')
        assert not (config.root / 'agent-workspace').exists()


    def test_other_robot_cannot_invoke_this_entry(self):
        config, path = self.session
        wrong = SimpleNamespace(**{**vars(config), 'robot': 'g1_sonic'})
        with self.assertRaisesRegex(ValueError, 'behavior_ap'):
            behavior.prepare(wrong, path, '/python', '/codex')


    def test_run_records_each_line_timestamps_and_total_walltime_without_claiming_success(self):
        config, path = self.session
        plan = behavior.prepare(config, path, '/python', '/codex')
        rows = [
            {'type': 'error', 'message': 'Reconnecting... waiting for network'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'Observed a change.'}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 10}},
        ]
        code = 'import json,time; rows=' + repr(rows) + ';\nfor r in rows:\n print(json.dumps(r),flush=True); time.sleep(.02)\n'
        plan['command'] = [sys.executable, '-u', '-c', code]
        result = behavior.run_agent(config, plan)
        events = [json.loads(line) for line in (config.root / 'agent_events.jsonl').read_text().splitlines()]
        assert [row['event'] for row in events] == rows
        assert all(row['arrived_at_utc'].endswith('+00:00') for row in events)
        assert all(a['arrived_perf_counter_ns'] <= b['arrived_perf_counter_ns'] for a, b in zip(events, events[1:]))
        assert result['status'] == 'agent_completed' and result['agent_wall_seconds'] >= .06
        assert result['official_task_success'] is None
        assert result['codex_turns_completed'] == 1
        assert result['non_reconnect_errors'] == 0
        assert result['model_reasoning_effort'] == 'low'
        assert result['model_service_tier_requested'] == 'fast'
        assert result['server_service_tier_verified'] is False
        assert 'not pure inference latency' in result['timing_scope']
        assert 'Observed a change.' in (config.root / 'agent_messages.md').read_text()
        with self.assertRaisesRegex(ValueError, 'execution evidence'):
            behavior.prepare(config, path, '/python', '/codex')


    def test_unlimited_session_has_no_implicit_deadline(self):
        config, path = self.session
        from dataclasses import replace
        unlimited = replace(config, session_seconds=None)
        path.write_text(json.dumps(unlimited.to_dict()))
        plan = behavior.prepare(unlimited, path, '/python', '/codex')
        plan['command'] = [sys.executable, '-c', 'print(\'{"type":"turn.completed"}\')']
        result = behavior.run_agent(unlimited, plan)
        assert result['status'] == 'agent_completed'
        assert result['walltime_budget_seconds'] is None
        assert not result['timed_out']

    def test_zero_exit_without_turn_complete_is_infrastructure_failure(self):
        config, path = self.session
        plan = behavior.prepare(config, path, '/python', '/codex')
        plan['command'] = [sys.executable, '-c', 'print("not a completed model turn")']
        result = behavior.run_agent(config, plan)
        assert result['exit_code'] == 0 and result['status'] == 'infrastructure_error'
        assert result['official_task_success'] is None


    def test_walltime_budget_kills_only_owned_agent_group(self):
        config, path = self.session
        plan = behavior.prepare(config, path, '/python', '/codex')
        plan['command'] = [sys.executable, '-c', 'import time; time.sleep(20)']
        bounded = SimpleNamespace(**{**vars(config), 'session_seconds': .1})
        result = behavior.run_agent(bounded, plan)
        assert result['timed_out'] and result['status'] == 'walltime_budget_exhausted'
        assert .1 <= result['agent_wall_seconds'] < 3
        assert result['official_task_success'] is None
        assert not result['automatic_motion_replay']
        assert not result['simulator_worker_closed_by_launcher']

if __name__ == "__main__":
    unittest.main()
