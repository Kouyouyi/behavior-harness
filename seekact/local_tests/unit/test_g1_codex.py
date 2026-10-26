import importlib.util
import json
from pathlib import Path
import time

import pytest

from harness.codex_cli.launcher import prepare, settings, pause_active
from services.controller.config import Configuration


def config(tmp_path,**kwargs):
    return Configuration(root=tmp_path/'session',task='vla',robot='g1_sonic',agent_provider='codex',subagents=0,**kwargs)


def test_codex_uses_stdio_owner_and_does_not_rewrite_login(tmp_path):
    c=config(tmp_path);path=tmp_path/'operator-input.json'
    path.write_text(json.dumps(c.to_dict()))
    command=prepare(c,path,'/python','/codex')
    assert '--no-alt-screen' in command and '--sandbox' in command
    assert 'read-only' in command and not any('CODEX_HOME=' in v for v in command)
    assert any('services.controller.frontend' in v for v in command)
    assert any('g1_vla' in v for v in command)
    assert any('g1_move' in v for v in command)
    assert all(any(name in v for v in command) for name in ('g1_hand','g1_hand_result','g1_operator_messages'))
    text=(c.root/'agent-workspace/AGENTS.md').read_text()
    assert 'gain 1.5' in text and '10-second' in text
    assert 'wait for the operator' in text
    assert not (c.root/'state.json').exists()


def test_only_robot_mcp_and_no_shell_in_session(tmp_path):
    values=settings(config(tmp_path),'/config','/python',['remote','name.with.dots','rax_robot'])
    assert 'mcp_servers."remote".enabled=false' in values
    assert 'mcp_servers."name.with.dots".enabled=false' in values
    assert 'features.shell_tool=false' in values
    assert 'features.plugins=false' in values
    assert 'features.multi_agent=false' in values
    assert 'features.code_mode_host=true' in values
    assert 'features.code_mode_host=false' not in values
    assert 'mcp_servers.rax_robot.enabled=true' in values


def test_exit_without_owner_does_not_fabricate_stop(tmp_path):
    outcome=pause_active(config(tmp_path))
    assert outcome['status']=='pause_not_confirmed'
    assert not (tmp_path/'vla_requests').exists()


def test_reject_parallel_agents(tmp_path):
    c=Configuration(root=tmp_path/'session',task='vla',robot='g1_sonic',agent_provider='codex',subagents=1)
    with pytest.raises(ValueError,match='subagents'):prepare(c,tmp_path/'c','/python','/codex')


def test_watch_handles_partial_audit_line_and_stale_owner(tmp_path):
    file=Path(__file__).resolve().parents[2]/'scripts/g1_watch_agent.py'
    spec=importlib.util.spec_from_file_location('watch',file);watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)
    audit=tmp_path/'mcp.jsonl'
    row=json.dumps(dict(time=time.time(),event='mcp_request',tool='g1_vla',arguments=dict(evidence='Observed target')))
    audit.write_text(row);tail=watch.AuditTail(audit);tail.poll();assert not tail.lines
    with audit.open('a') as f:f.write('\n')
    tail.poll();assert len(tail.lines)==1 and 'Observed target' in tail.lines[0]
    tail.poll();assert len(tail.lines)==1
    assert '过期/未启动' in watch.snapshot(tmp_path,tmp_path/'config',tail)
    assert '\x1b' not in watch.short('bad\x1b[2J')


def test_codex_mcp_stdio_protocol_lists_vla_without_motion(tmp_path):
    import subprocess,sys
    c=config(tmp_path);path=tmp_path/'operator.json';path.write_text(json.dumps(c.to_dict()))
    requests=[dict(jsonrpc='2.0',id=1,method='initialize',params={}),
              dict(jsonrpc='2.0',id=2,method='tools/list',params={})]
    result=subprocess.run([sys.executable,'-m','services.controller.frontend','--config',str(path)],
        input=''.join(json.dumps(r)+'\n' for r in requests),capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    replies=[json.loads(line) for line in result.stdout.splitlines()]
    assert replies[0]['result']['serverInfo']['name']=='robot'
    assert 'g1_vla' in {t['name'] for t in replies[1]['result']['tools']}
    assert 'g1_move' in {t['name'] for t in replies[1]['result']['tools']}
    assert not (tmp_path/'vla_requests').exists()


def test_codex_exit_requests_stop_for_active_movement_without_vla(tmp_path):
    from services.g1_sonic.locomotion import LocomotionJobs
    owner=tmp_path/'owner';owner.mkdir();jobs=LocomotionJobs(owner,'session')
    (owner/'ankle_status.json').write_text(json.dumps(dict(session_id='session',phase='ready',
        updated_monotonic=time.monotonic(),locomotion=jobs.snapshot()|dict(active_job={'id':'a'*32}))))
    response=pause_active(config(tmp_path,backend_options=dict(control_directory=str(owner))))
    assert response['status']=='movement_stop_requested'
    assert json.loads((owner/'move_requests/stop.json').read_text())['action']=='stop'
