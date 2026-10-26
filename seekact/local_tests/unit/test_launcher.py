"""Host-run Claude Code launcher: workspace confinement, sandbox, credentials and resume."""
import json
from pathlib import Path
import time

import pytest

from conftest import configuration
from harness.claude_cli import launcher


@pytest.fixture
def deployed(tmp_path):
    key = tmp_path/'anthropic.key'
    key.write_text('sk-ant-api-secret')
    key.chmod(0o600)
    binary = tmp_path/'bin/claude'
    binary.parent.mkdir()
    binary.write_text('')
    config = configuration(tmp_path, provider='anthropic', subagents=0)
    return config, launcher.deploy(config, key_file=key, claude=binary)


def test_deploy_builds_the_workspace_and_private_session_files(deployed):
    config, d = deployed
    workspace = d['workspace']
    assert (workspace/'AGENTS.md').is_file() and (workspace/'CLAUDE.md').read_text() == '@AGENTS.md\n'
    assert (workspace/'.claude/skills').resolve() == (workspace/'.agents/skills').resolve()
    assert 'Stack the red cup' in (workspace/'TASK.md').read_text()
    assert (workspace/'api/runtime.py').is_file()
    for name in ('configuration.json', 'model-relay.json'):
        assert (config.root/name).stat().st_mode & 0o077 == 0
    assert 'sk-ant' not in (config.root/'model-relay.json').read_text()
    assert not list(workspace.rglob('*.key'))
    # the persistent skill library: cloned on the session's own branch, outside the read-only skills
    clone = workspace/'skill-library'
    assert (clone/'.git').is_dir() and (clone/'INDEX.md').is_file()
    manifest = json.loads((config.root/'workspace-manifest.json').read_text())
    assert manifest['skill_library']['branch'] == f'session/{config.root.name}'
    assert (workspace/'.agents/skills/continual-learning/SKILL.md').is_file()
    assert 'skill_library_merge' in (workspace/'MCP_SESSION.md').read_text()


def test_settings_confine_tools_to_the_workspace_and_sandbox_bash(deployed):
    config, d = deployed
    settings = json.loads((d['claude_home']/'settings.json').read_text())
    workspace = str(d['workspace'])
    permissions, sandbox = settings['permissions'], settings['sandbox']
    assert permissions['defaultMode'] == 'dontAsk'
    assert f'Edit(/{workspace}/**)' in permissions['allow'] and 'mcp__robot' in permissions['allow']
    assert 'Agent' not in permissions['allow'] and 'WebFetch' in permissions['allow'] and 'WebSearch' in permissions['deny']
    assert sandbox['enabled'] and sandbox['failIfUnavailable'] and not sandbox['allowUnsandboxedCommands']
    assert sandbox['filesystem']['allowWrite'] == [workspace]
    denied = sandbox['filesystem']['denyRead']
    for protected in (launcher.PROJECT, Path.home()/'.claude', Path.home()/'.ssh', Path.home()/'.config/rax',
                      config.root/'model-relay.json', config.root/'mcp.jsonl'):
        assert str(protected) in denied
        assert f'Read(/{protected})' in permissions['deny']
    assert sandbox['network'] == {'allowAllUnixSockets': True, 'allowLocalBinding': False, 'allowedDomains': ['*'],
                                  'deniedDomains': ['192.168.30.1']}  # the RB-Y1 controller's unauthenticated gRPC
    assert 'WebFetch(domain:192.168.30.1)' in permissions['deny']
    for hidden in ('/run/docker.sock', '/var/run/docker.sock', f'/run/user/{launcher.UID}'):
        assert hidden in denied  # Unix sockets that would bypass the robot service.
    assert settings['env']['RAX_MCP_SOCKET'] == str(d['socket'])
    assert settings['env']['PYTHONPATH'] == f'{workspace}:{workspace}/.pylib'  # the workspace, then the agent's own packages
    assert len(str(d['socket'])) < 100


def test_site_paths_that_reach_the_robot_are_protected(tmp_path, monkeypatch):
    """The operator lists robot-controller checkouts and SDK environments in RAX_PROTECTED_PATHS."""
    stacks = [tmp_path/'controller-checkout', tmp_path/'vendor-sdk-env']
    monkeypatch.setenv('RAX_PROTECTED_PATHS', ':'.join(map(str, stacks)) + ':')
    config = configuration(tmp_path)
    settings = launcher.settings(config, config.root/'agent-workspace', tmp_path/'robot.sock')
    for stack in stacks:
        assert str(stack) in settings['sandbox']['filesystem']['denyRead']
        assert f'Read(/{stack}/**)' in settings['permissions']['deny']
    monkeypatch.setenv('RAX_PROTECTED_PATHS', 'relative/sdk')
    with pytest.raises(ValueError, match='absolute paths'):
        launcher.settings(config, config.root/'agent-workspace', tmp_path/'robot.sock')


def test_unsafe_defaults_are_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('RAX_PROTECTED_PATHS', raising=False)
    monkeypatch.delenv('RAX_AGENT_PYTHON', raising=False)
    launcher.warn_unsafe_defaults(configuration(tmp_path))
    warnings = capsys.readouterr().err
    assert 'RAX_PROTECTED_PATHS is empty' in warnings and 'RAX_AGENT_PYTHON is set' in warnings
    monkeypatch.setenv('RAX_PROTECTED_PATHS', str(tmp_path/'sdk'))
    launcher.warn_unsafe_defaults(configuration(tmp_path, agent_python='/opt/agent/bin/python'))
    assert capsys.readouterr().err == ''


def test_the_configured_controller_address_is_denied_too(tmp_path):
    """The factory controller address is always refused; a configured `address` adds its host."""
    config = configuration(tmp_path)
    assert launcher.denied_hosts(config) == ['192.168.30.1']
    moved = configuration(tmp_path, backend_options={'address': '10.0.0.7:50051'})
    settings = launcher.settings(moved, moved.root/'agent-workspace', tmp_path/'robot.sock')
    assert settings['sandbox']['network']['deniedDomains'] == ['192.168.30.1', '10.0.0.7']
    assert 'WebFetch(domain:10.0.0.7)' in settings['permissions']['deny']
    assert launcher.denied_hosts(configuration(tmp_path, backend_options={'address': '192.168.30.1:50051'})) \
        == ['192.168.30.1']


def test_agent_python_comes_from_the_session_configuration(tmp_path):
    default = configuration(tmp_path/'a')
    assert launcher.agent_python(default) == Path(launcher.sys.executable).parent
    chosen = configuration(tmp_path/'b', agent_python='/opt/agent-venv/bin/python')
    assert launcher.agent_python(chosen) == Path('/opt/agent-venv/bin')
    settings = launcher.settings(chosen, tmp_path/'ws', tmp_path/'s.sock')
    assert settings['env']['PATH'].startswith('/opt/agent-venv/bin:')
    with pytest.raises(ValueError, match='agent_python'):
        configuration(tmp_path/'c', agent_python='relative/python')


def test_mcp_config_bridges_to_the_robot_service_socket(deployed):
    _, d = deployed
    server = json.loads((d['claude_home']/'mcp.json').read_text())['mcpServers']['robot']
    assert server['type'] == 'stdio' and server['args'] == [
        str(launcher.PROJECT/'harness/claude_cli/mcp_bridge.py'), str(d['socket'])]


def test_cli_arguments_and_environment_hold_no_credential(deployed):
    _, d = deployed
    args = launcher.claude_arguments(d, prompt='go')
    assert args[:2] == ['-p', '--model'] and args[-2:] == ['--', 'go']
    assert args[args.index('--permission-mode')+1] == 'dontAsk' and 'Agent' in args
    assert args[args.index('--effort')+1] == 'xhigh'
    environment = launcher.agent_environment(d, ('127.0.0.1', 4242))
    assert environment['ANTHROPIC_BASE_URL'] == 'http://127.0.0.1:4242'
    assert environment['HOME'] == environment['CLAUDE_CONFIG_DIR'] == str(d['claude_home'])
    assert 'sk-ant' not in json.dumps(environment) + ' '.join(args)


def test_a_used_session_root_is_refused(deployed, tmp_path):
    config, _ = deployed
    with pytest.raises(FileExistsError):
        launcher.deploy(config, key_file=tmp_path/'anthropic.key', claude=tmp_path/'bin/claude')


def test_resume_picks_the_newest_top_level_conversation(tmp_path):
    project = tmp_path/'projects/-ws'
    (project/'old/subagents').mkdir(parents=True)
    (project/'old.jsonl').write_text('{}')
    time.sleep(.01)
    (project/'new.jsonl').write_text('{}')
    (project/'old/subagents/agent-x.jsonl').write_text('{}')
    assert launcher.latest_claude_session(tmp_path) == 'new'


def test_resume_budget_excludes_usage_waits(tmp_path):
    record = tmp_path/'agent-lifecycle.json'
    record.write_text(json.dumps({'started_at': 100.0, 'finished_at': 400.0, 'usage_wait_seconds': 100.0,
                                  'attempts': [{'started_at': 0.0, 'finished_at': 50.0}]}))
    assert launcher.attempt_seconds(launcher.resumable_lifecycle(record)) == 250.0


def test_agent_python_prefers_the_session_then_the_operator_environment(tmp_path, monkeypatch):
    config = configuration(tmp_path, provider='anthropic')
    monkeypatch.setenv('RAX_AGENT_PYTHON', '/opt/agent/bin/python')
    assert launcher.agent_python(config) == Path('/opt/agent/bin')
    from dataclasses import replace
    assert launcher.agent_python(replace(config, agent_python='/srv/venv/bin/python')) == Path('/srv/venv/bin')


def test_robot_model_folder_is_linked_read_only_when_installed(tmp_path, monkeypatch):
    from services.workspace.setup import ROBOT_FACTS
    model = tmp_path/'robot-models'/'mock'
    model.mkdir(parents=True)
    (model/'model.urdf').write_text('<robot name="m"/>')
    monkeypatch.setitem(ROBOT_FACTS['mock'], 'model_dir', 'robot-models/mock')
    key = tmp_path/'anthropic.key'
    key.write_text('sk-ant-api-secret')
    key.chmod(0o600)
    binary = tmp_path/'bin/claude'
    binary.parent.mkdir()
    binary.write_text('')
    config = configuration(tmp_path, provider='anthropic', subagents=0)
    d = launcher.deploy(config, key_file=key, claude=binary)
    assert (d['workspace']/'robot-model'/'model.urdf').is_file() and (d['workspace']/'.pylib').is_dir()
    settings = json.loads((d['claude_home']/'settings.json').read_text())
    assert f'Read(/{model}/**)' in settings['permissions']['allow']


def _deploy(tmp_path, **overrides):
    key = tmp_path/'anthropic.key'
    key.write_text('sk-ant-api-secret')
    key.chmod(0o600)
    binary = tmp_path/'bin/claude'
    binary.parent.mkdir(exist_ok=True)
    binary.write_text('')
    config = configuration(tmp_path, provider='anthropic', **overrides)
    return config, launcher.deploy(config, key_file=key, claude=binary)


def test_native_subagents_every_model_and_pinned_aliases(tmp_path):
    config, d = _deploy(tmp_path, subagents=2)
    allow = json.loads((d['claude_home']/'settings.json').read_text())['permissions']['allow']
    assert 'Agent' in allow and not (d['claude_home']/'agents').exists()
    assert json.loads((config.root/'model-relay.json').read_text())['models'] == ['*']
    env = launcher.agent_environment(d, ('127.0.0.1', 1))
    assert env['ANTHROPIC_DEFAULT_HAIKU_MODEL'] == 'claude-haiku-5-5'
    assert env['ANTHROPIC_DEFAULT_SONNET_MODEL'] == 'claude-sonnet-5-5'
    assert env['ANTHROPIC_DEFAULT_OPUS_MODEL'] == 'claude-opus-5-5'
    assert 'haiku' in (d['workspace']/'AGENTS.md').read_text()


def test_resume_refreshes_the_harness_and_keeps_the_agents_work(tmp_path):
    config, d = _deploy(tmp_path, subagents=2)
    workspace, home = d['workspace'], d['claude_home']
    (workspace/'AGENTS.md').write_text('old instructions')
    (workspace/'.agents/skills/old-skill').mkdir()
    (workspace/'memory'/'PROGRESS.md').write_text('my progress')
    (workspace/'code'/'mine.py').write_text('x = 1')
    (home/'agents').mkdir()
    (home/'agents'/'perception-check.md').write_text('stale')
    relay = json.loads((config.root/'model-relay.json').read_text())
    relay['models'] = ['claude-opus-5-5']
    (config.root/'model-relay.json').write_text(json.dumps(relay))
    (config.root/'model-relay.json').chmod(0o600)
    (home/'projects'/'ws').mkdir(parents=True)
    (home/'projects'/'ws'/'conversation-1.jsonl').write_text('{}\n')  # the conversation being resumed
    redeployed = launcher.redeploy(config, claude=d['claude'])
    assert redeployed['resume_session'] == 'conversation-1'
    assert 'AGENTS.md' in redeployed['refreshed']
    assert (workspace/'AGENTS.md').read_text() != 'old instructions' and not (workspace/'.agents/skills/old-skill').exists()
    assert (workspace/'.agents/skills/continual-learning/SKILL.md').is_file()
    assert (workspace/'memory'/'PROGRESS.md').read_text() == 'my progress' and (workspace/'code'/'mine.py').is_file()
    assert not (home/'agents'/'perception-check.md').exists()
    assert json.loads((config.root/'model-relay.json').read_text())['models'] == ['*']
    assert 'Agent' in json.loads((home/'settings.json').read_text())['permissions']['allow']
