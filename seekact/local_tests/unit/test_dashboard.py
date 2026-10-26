"""Operator dashboard against a temporary artifact root; telemetry and launches are mocked."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from services.dashboard import sessions as sessions_module
from services.dashboard.server import COOKIE, make_server
from services.dashboard.sessions import Sessions
from services import arming

PROJECT = Path(__file__).resolve().parents[2]


class FakeTelemetry:
    def __init__(self):
        self.value = {'kind': 'rby1', 'ok': True, 'address': '192.168.30.1:50051', 'emo_pressed': [],
                      'power': True, 'manager': 'Enabled', 'control': 'Idle', 'stale': [],
                      'joints': {'head': [0.0, 0.0]}, 'battery': {'level_percent': 90}}

    def snapshot(self):
        return dict(self.value)


@pytest.fixture
def telemetry():
    return FakeTelemetry()


@pytest.fixture
def dashboard(artifact_root, telemetry):
    # A launcher stand-in that just sleeps, so launches never start a model or a robot service.
    sleeper = [sys.executable, '-c', 'import sys, time; time.sleep(60)']
    sessions = Sessions(project=PROJECT, launcher=sleeper, robot='rby1', telemetry=telemetry)
    server, token = make_server('127.0.0.1', 0, sessions=sessions, token='test-token')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    yield base, token, sessions
    server.shutdown()
    server.server_close()
    for root in sessions.roots():
        record = sessions_module.read_json(root/'dashboard-launch.json')
        if record:
            try:
                os.killpg(record['pgid'], 9)
            except ProcessLookupError:
                pass


def call(base, path, body=None, token='test-token', headers=None):
    data = None if body is None else json.dumps(body).encode()
    all_headers = {'Content-Type': 'application/json', **({'X-Operator-Token': token} if token else {}),
                   **(headers or {})}
    request = Request(base + path, data=data, method='POST' if body is not None else 'GET', headers=all_headers)
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b'{}')
    except HTTPError as exc:
        return exc.code, json.loads(exc.read() or b'{}')


def configure(base, task='red_on_blue'):
    status, value = call(base, '/api/configure', {'task': task, 'task_text': 'Put the red cube on the blue cube.'})
    assert status == 200, value
    return value


def test_auth_required(dashboard):
    base, token, _ = dashboard
    assert call(base, '/api/sessions', token=None)[0] == 401
    assert call(base, '/api/sessions', token='wrong')[0] == 401
    assert call(base, '/api/disarm-all', {}, token=None)[0] == 401
    with pytest.raises(HTTPError) as error:
        urlopen(base + '/', timeout=5)
    assert error.value.code == 401


def test_token_url_sets_cookie(dashboard):
    base, token, _ = dashboard

    class NoRedirect(__import__('urllib.request').request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    opener = __import__('urllib.request').request.build_opener(NoRedirect)
    with pytest.raises(HTTPError) as error:
        opener.open(base + f'/?token={token}', timeout=5)
    assert error.value.code == 303
    cookie = error.value.headers['Set-Cookie']
    assert cookie.startswith(f'{COOKIE}={token}') and 'HttpOnly' in cookie and 'SameSite=Strict' in cookie
    with urlopen(Request(base + '/', headers={'Cookie': f'{COOKIE}={token}'}), timeout=5) as response:
        assert b'Robot Operator' in response.read()


def test_post_requires_json_content_type(dashboard):
    base, token, _ = dashboard
    status, value = call(base, '/api/disarm-all', {}, headers={'Content-Type': 'application/x-www-form-urlencoded'})
    assert status == 400 and 'application/json' in value['error']


def test_configure_and_list(dashboard, artifact_root):
    base, token, _ = dashboard
    value = configure(base)
    config = json.loads(Path(value['config_path']).read_text())
    assert config['agent_effort'] == 'max' and config['agent_provider'] == 'claude-login'
    assert config['task'] == 'red_on_blue' and config['observation_profile'] == 'rgbd'
    assert value['commands']['interactive'].endswith(f"--config {value['config_path']}")
    assert '--non-interactive' in value['commands']['headless']
    status, listing = call(base, '/api/sessions')
    assert status == 200 and [s['id'] for s in listing['sessions']] == [value['id']]
    session = listing['sessions'][0]
    assert session['status'] == 'configured' and session['running'] is False
    assert session['motion']['armed'] is False and session['usage']['calls'] == 0
    assert call(base, '/api/configure', {'task': '../evil'})[0] == 400


def test_arm_requires_confirmation_and_disarm_holds(dashboard, artifact_root):
    base, token, sessions = dashboard
    session_id = configure(base)['id']
    status, value = call(base, f'/api/session/{session_id}/arm', {})
    assert status == 403 and 'emergency stop' in value['error']
    assert call(base, f'/api/session/{session_id}/arm', {'estop_confirmed': 'yes'})[0] == 403
    status, value = call(base, f'/api/session/{session_id}/arm', {'estop_confirmed': True, 'minutes': 30})
    assert status == 200 and value['armed'] is True and value['until'] is not None
    status, value = call(base, f'/api/session/{session_id}/disarm', {})
    assert status == 200 and value['motion']['armed'] is False and value['hold'].startswith('ok')


def test_disarm_all(dashboard):
    base, token, sessions = dashboard
    first, second = configure(base, 'a')['id'], configure(base, 'b')['id']
    for session_id in (first, second):
        call(base, f'/api/session/{session_id}/arm', {'estop_confirmed': True})
    status, value = call(base, '/api/disarm-all', {})
    assert status == 200 and sorted(value['disarmed']) == sorted([first, second])
    assert all(not arming.read(root)['armed'] for root in sessions.roots())


def test_launch_refused_while_another_runs(dashboard, monkeypatch):
    base, token, sessions = dashboard
    monkeypatch.setattr(sessions_module, 'is_launcher', lambda pid: True)  # the stand-in is not the launcher
    first, second = configure(base, 'a')['id'], configure(base, 'b')['id']
    status, value = call(base, f'/api/session/{first}/launch', {})
    assert status == 200 and value['pid'] > 0
    status, listing = call(base, '/api/sessions')
    assert listing['running'] == [first]
    status, value = call(base, f'/api/session/{second}/launch', {})
    assert status == 409 and 'one robot' in value['error']
    status, value = call(base, f'/api/session/{first}/stop', {})
    assert status == 200
    record = json.loads((sessions.root(first)/'dashboard-launch.json').read_text())
    for _ in range(100):
        if not sessions_module.pid_alive(record['pid']):
            break
        __import__('time').sleep(0.05)
    assert not sessions_module.pid_alive(record['pid'])


def test_resume_needs_a_previous_launch(dashboard):
    base, token, _ = dashboard
    session_id = configure(base)['id']
    status, value = call(base, f'/api/session/{session_id}/resume', {})
    assert status == 409 and 'never launched' in value['error']


def test_robot_panel_shows_registered_telemetry(dashboard, telemetry):
    base, token, sessions = dashboard
    status, value = call(base, '/api/robot')
    assert status == 200 and value['kind'] == 'rby1' and value['ok'] is True

    def broken():
        raise RuntimeError('grpc down')
    telemetry.snapshot = broken
    status, value = call(base, '/api/robot')
    assert status == 200 and value['ok'] is False and 'grpc down' in value['error']


def test_robot_without_telemetry_reports_it():
    sessions = Sessions(project=PROJECT, robot='mock')
    value = sessions.robot()
    assert value['ok'] is False and 'no telemetry' in value['error']


def test_frames_and_activity(dashboard):
    base, token, sessions = dashboard
    session_id = configure(base)['id']
    root = sessions.root(session_id)
    frame = root/'agent-workspace/runtime/frames/20261003120000abcdef'
    frame.mkdir(parents=True)
    (frame/'left.jpg').write_bytes(b'\xff\xd8jpeg')
    (frame/'meta.json').write_text(json.dumps({'time': 1.0, 'tcp_position': [0.4, 0, 0.3]}))
    status, value = call(base, f'/api/session/{session_id}/frame')
    assert value['frame']['frame_id'] == frame.name and value['frame']['cameras'] == ['left']
    with urlopen(Request(base + f'/api/session/{session_id}/frame/{frame.name}/left.jpg',
                         headers={'X-Operator-Token': token}), timeout=5) as response:
        assert response.read() == b'\xff\xd8jpeg'
    assert call(base, f'/api/session/{session_id}/frame/..%2F..%2Fx/left.jpg')[0] in (400, 404)
    transcript = root/'claude-home/projects/-ws/abc.jsonl'
    transcript.parent.mkdir(parents=True)
    transcript.write_text('\n'.join(json.dumps(e) for e in [
        {'type': 'assistant', 'timestamp': 't1', 'message': {'content': [
            {'type': 'text', 'text': 'Looking at the scene.'},
            {'type': 'tool_use', 'name': 'mcp__robot__rby1_observe', 'input': {}}]}},
        {'type': 'user', 'timestamp': 't2', 'message': {'content': [
            {'type': 'tool_result', 'content': [{'type': 'text', 'text': '{"frame_id": "x"}'}, {'type': 'image'}]}]}},
    ]) + '\n')
    (root/'mcp.jsonl').write_text(json.dumps({'time': 1.0, 'event': 'mcp_request', 'tool': 'rby1_observe',
                                              'call_id': 'c', 'arguments': {}}) + '\n')
    status, value = call(base, f'/api/session/{session_id}/activity')
    kinds = [e['kind'] for e in value['transcript']]
    assert kinds == ['assistant', 'tool_call', 'tool_result']
    assert value['transcript'][1]['name'] == 'mcp__robot__rby1_observe'
    assert '[image]' in value['transcript'][2]['text']
    assert value['mcp'][0]['tool'] == 'rby1_observe'
    assert call(base, '/api/session/nope/activity')[0] == 404


def test_rby1_telemetry_snapshot_flags_emo_stale_state_and_faults():
    from datetime import timedelta
    from types import SimpleNamespace as NS
    from services.rby1.telemetry import Telemetry

    def enum(name):
        return NS(name=name)
    ages = [0.01] * 22 + [2.0, 2.0]  # the head stopped reporting
    state = NS(joint_states=[NS(time_since_last_update=timedelta(seconds=a)) for a in ages], position=[0.0] * 24,
               emo_states=[NS(state=enum('Released')), NS(state=enum('Pressed'))],
               battery_state=NS(level_percent=81.5, voltage=50.2),
               ft_sensor_right=NS(force=[0.0, 0.0, 1.0]), ft_sensor_left=NS(force=[0.0, 0.0, 2.0]))
    commanded = []

    class Robot:
        def get_state(self):
            return state

        def get_control_manager_state(self):
            return NS(state=enum('Enabled'), control_state=enum('Idle'))

        def is_power_on(self, pattern):
            return True

        def __getattr__(self, name):  # any other call (a command) fails the test
            commanded.append(name)
            raise AssertionError(f'telemetry called {name}')

    class Kin:
        def frames(self, names, q):
            import numpy as np
            return {n: np.eye(4) for n in names}
    value = Telemetry(robot=Robot(), kinematics=Kin()).snapshot()
    assert value['kind'] == 'rby1' and value['ok'] is False and commanded == []
    assert value['stale'] == ['head'] and value['emo_pressed'] == [1] and value['battery']['level_percent'] == 81.5
    assert value['end_effectors'] == {'right': [0.0, 0.0, 0.0], 'left': [0.0, 0.0, 0.0]}
