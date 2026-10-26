"""The robot service MCP: tool listing, dispatch, audit, uncertainty and the socket transport."""
import json
from pathlib import Path
import socket
import threading

import pytest

from conftest import configuration, frontend


def text(result):
    return json.loads(result['content'][0]['text'])


def test_tools_are_status_plus_backend_tools(tmp_path):
    f = frontend(tmp_path)
    try:
        names = [t['name'] for t in f.tools()]
        assert names == ['status', 'request_operator', 'mock_observe', 'mock_move']  # No Gemini without a key.
        status = text(f.call('status', {}))
        assert status['task'] == 'stack_cups' and status['robot_status']['robot'] == 'mock'
        assert 'control_uncertain' not in status
    finally:
        f.close()


def test_validation_failure_moves_nothing_and_reports_the_whole_reason(tmp_path):
    f = frontend(tmp_path)
    try:
        reply = f.call('mock_move', {'position': [5, 0, 0]})
        error = text(reply)['error']
        assert reply['isError'] and error['stage'] == 'validation' and error['reason']
        assert set(error) == {'type', 'operation', 'stage', 'reason'}  # no labels, no generic hints
        assert 'Traceback' not in error['reason']  # an expected refusal needs no traceback
        assert f.backend.position == [0.0, 0.0, 0.0]
        assert text(f.call('mock_move', {'position': [0.1, 0.2, 0.3]})) == {'position': [0.1, 0.2, 0.3]}
    finally:
        f.close()


def test_an_unexpected_error_reaches_the_agent_whole_with_its_traceback(tmp_path):
    f = frontend(tmp_path)
    original = f.backend.call
    def broken(name, arguments):
        raise RuntimeError('This stream is expired')
    try:
        f.backend.call = broken
        error = text(f.call('mock_move', {'position': [0.1, 0, 0]}))['error']
        assert error['type'] == 'RuntimeError' and error['reason'].startswith('This stream is expired')
        assert 'Traceback (innermost last)' in error['reason'] and 'in broken' in error['reason']
        assert str(Path(__file__).parent) not in error['reason'] and 'control_uncertain' not in error  # no host dirs
        f.backend.call = original  # nothing is locked afterwards
        assert 'control_uncertain' not in text(f.call('mock_observe', {}))
        assert text(f.call('mock_move', {'position': [0, 0, 0]})) == {'position': [0, 0, 0]}
    finally:
        f.close()


def test_unknown_tools_and_bad_arguments_are_refused(tmp_path):
    f = frontend(tmp_path)
    try:
        assert f.call('robodojo_step', {})['isError']
        assert f.call('status', {'x': 1})['isError']
        assert f.call('mock_move', [1, 2])['isError']
        assert f.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'resources/list'})['error']
    finally:
        f.close()


def test_every_call_is_audited(tmp_path):
    f = frontend(tmp_path)
    try:
        f.call('mock_move', {'position': [0.1, 0, 0]})
        f.call('mock_move', {'position': [9, 0, 0]})
    finally:
        f.close()
    events = [json.loads(line) for line in (configuration(tmp_path).root/'mcp.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events] == ['mcp_request', 'mcp_result', 'mcp_request', 'mcp_error']


def test_backend_tools_may_not_shadow_harness_tools(tmp_path):
    from services.controller.frontend import Frontend
    from services.controller.session import Session

    class Shadow:
        def tools(self):
            return [{'name': 'status', 'description': '', 'inputSchema': {'type': 'object'}}]
    config = configuration(tmp_path)
    session = Session(config)
    try:
        with pytest.raises(ValueError, match='harness tool names'):
            Frontend(session, Shadow())
    finally:
        session.close()


def test_socket_serves_the_agent_cli_bridge_and_python_scripts(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'auto_research_agent'))
    from api.runtime import Context
    f = frontend(tmp_path)
    path = tmp_path/'mcp.sock'
    stop = threading.Event()
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(path))
        listener.listen(16)
        server = threading.Thread(target=f.serve_socket, args=(listener,), kwargs={'stop': stop}, daemon=True)
        server.start()
        try:
            monkeypatch.setenv('RAX_MCP_SOCKET', str(path))
            with Context() as first, Context() as second:
                first.call('mock_move', position=[0.2, 0, 0])
                assert second.value('mock_observe') == {'position': [0.2, 0.0, 0.0]}
                with pytest.raises(RuntimeError, match='MCP tool failed'):
                    second.call('mock_move', position=[3, 0, 0])
                tools = first.request('tools/list', {})['tools']
                assert [t['name'] for t in tools][0] == 'status'
        finally:
            stop.set()
            server.join(timeout=5)
            f.close()


def test_request_operator_waits_for_the_reply(tmp_path):
    from services.controller import operator
    f = frontend(tmp_path)
    root = f.s.config.root

    def answer():
        for _ in range(100):
            waiting = operator.pending(root)
            if waiting:
                operator.reply(root, waiting[0]['id'], 'cubes are back in place')
                return
            threading.Event().wait(0.05)
    helper = threading.Thread(target=answer)
    helper.start()
    try:
        value = text(f.call('request_operator', {'kind': 'reset', 'message': 'red cube fell off', 'timeout_s': 30}))
        helper.join()
        assert value['status'] == 'answered' and value['reply'] == 'cubes are back in place'
        assert value['robot_status']['robot'] == 'mock' and operator.pending(root) == []
        error = text(f.call('request_operator', {'kind': 'coffee', 'message': 'x'}))['error']
        assert error['stage'] == 'validation'
    finally:
        f.close()


def test_a_handled_stop_reports_its_reason_and_the_measured_state(tmp_path):
    f = frontend(tmp_path)

    def refuse(name, arguments):
        exc = RuntimeError('refused; the arm holds')
        exc.handled_stop = True
        exc.robot_state = {'tcp_position': [0.4, 0.0, 0.3]}
        raise exc
    f.backend.call = refuse
    try:
        error = text(f.call('mock_move', {'position': [0.1, 0.2, 0.3]}))['error']
        assert error['reason'] == 'refused; the arm holds' and error['robot_state']['tcp_position'] == [0.4, 0.0, 0.3]
        assert 'recovery' not in error and 'control_uncertain' not in error
    finally:
        f.close()


def test_skill_library_merge_is_listed_with_a_clone_and_merges_through_mcp(tmp_path):
    import subprocess
    from services import skill_library
    from services.controller.frontend import Frontend
    from services.controller.session import Session
    from services.mock_backend import MockBackend
    assert 'skill_library_merge' not in {t['name'] for t in frontend(tmp_path/'a').tools()}  # no clone: not listed
    config = configuration(tmp_path/'b')
    workspace = config.root/'agent-workspace'
    workspace.mkdir(parents=True)
    skill_library.checkout(config, workspace)
    f = Frontend(Session(config), MockBackend(config))
    assert 'skill_library_merge' in {t['name'] for t in f.tools()}
    reply = json.loads(f.call('skill_library_merge', {'message': 'nothing yet'})['content'][0]['text'])
    assert reply['merged'] is False and 'no committed changes' in reply['reason']
    folder = workspace/'skill-library'/'skills'/'wave'
    (folder/'code').mkdir(parents=True)
    (folder/'evidence'/'images').mkdir(parents=True)
    (folder/'evidence'/'images'/'start.jpg').write_bytes(b'\xff\xd8' + b'0' * 100)
    (folder/'code'/'run.py').write_text('def main(ctx):\n    return {"ok": True, "measured": {}, "why": ""}\n')
    sections = '\n\n'.join(f'## {s}\ntext' for s in skill_library.SECTIONS) + '\n\n### Perception checkpoints\nNone'
    (folder/'SKILL.md').write_text('---\nname: wave\ndescription: waves\nstatus: draft\nrobot: mock\n'
                                   f'entry: code/run.py\n---\n\n# Wave\n\n{sections}\n')
    index = workspace/'skill-library'/'INDEX.md'
    index.write_text(index.read_text() + '| [wave](skills/wave/SKILL.md) | draft | waves | greeting |\n')
    for args in (['add', '-A'], ['commit', '-q', '-m', 'wave']):
        subprocess.run(['git', *args], cwd=workspace/'skill-library', check=True)
    reply = json.loads(f.call('skill_library_merge', {'message': 'add wave'})['content'][0]['text'])
    assert reply['merged'] is True and reply['skills'] == ['wave']
