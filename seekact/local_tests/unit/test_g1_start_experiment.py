"""Launcher orchestration with fake processes only; never connects to a robot."""
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from scripts import g1_start_experiment as start
from scripts.g1_manage_terminals import Process


def proc(pid, *argv):
    return Process(pid, 1, '1', tuple(argv))


def exited(run):
    run.mkdir(parents=True, exist_ok=True)
    start.write(run/'native-exit.json', dict(reason='operator_keyboard_O', exit_code=0))


def test_live_native_blocks_all_retirement(tmp_path):
    owner = proc(2, 'python', '-m', 'g1_sonic.ankle_executor', '--run', str(tmp_path/'old'))
    exited(tmp_path/'old')
    with pytest.raises(RuntimeError, match='原生控制仍运行'):
        start.retirement_plan({2: owner, 3: proc(3, '/build/g1_field')}, tmp_path)


@pytest.mark.parametrize('reason,code', [('automatic_O', 0), ('operator_keyboard_O', 1), (None, 0)])
def test_owner_without_successful_manual_o_cannot_be_retired(tmp_path, reason, code):
    run = tmp_path/'old'
    exited(run)
    start.write(run/'native-exit.json', dict(reason=reason, exit_code=code))
    p = proc(2, 'python', '-m', 'g1_sonic.ankle_executor', '--run', str(run))
    with pytest.raises(RuntimeError, match='不自动关闭'):
        start.retirement_plan({2: p}, tmp_path)


def test_only_matching_finished_session_processes_are_retired(tmp_path):
    run = tmp_path/'old'
    exited(run)
    owner = proc(2, 'python', '-m', 'g1_sonic.ankle_executor', '--run', str(run))
    ui = proc(3, 'python', str(run/'console.py'))
    unrelated = proc(4, 'python', 'unrelated.py')
    start.write(run/'background-services.json', dict(
        ui=dict(pid=3, command=list(ui.argv)), frontend=dict(pid=4, command=['different'])))
    assert {p.pid for p in start.retirement_plan({2: owner, 3: ui, 4: unrelated}, tmp_path)} == {2, 3}
    os.utime(run/'native-exit.json', (0, 0))
    assert not start.manual_exit(run, owner)


def test_active_ui_task_is_not_cancelled_by_retirement(tmp_path):
    run = tmp_path/'old'
    exited(run)
    p = proc(2, 'python', '-m', 'harness.codex_cli.ui_agent', '--config', str(run/'operator-input.json'))
    start.write(run/'background-services.json', dict(ui_agent=dict(pid=2, command=list(p.argv))))
    start.write(run/'ui-agent-status.json', dict(phase='running'))
    with pytest.raises(RuntimeError, match='仍在处理任务'):
        start.retirement_plan({2: p}, tmp_path)


def test_native_race_never_sends_signal(tmp_path, monkeypatch):
    p = proc(2, 'python', '-m', 'g1_sonic.ankle_executor')
    monkeypatch.setattr(start.terminals, 'snapshot', lambda: {2: p, 3: proc(3, 'g1_field')})
    monkeypatch.setattr(start.os, 'pidfd_open', lambda _: pytest.fail('signal handle opened'))
    with pytest.raises(RuntimeError, match='取消旧服务清理'):
        start.retire([p])


def test_sensor_reuse_and_identity_requirements(tmp_path):
    p = proc(2, 'python', '-m', 'services.g1_sonic.camera_service', '--output', str(tmp_path))
    assert start.existing_source({2: p}, 'services.g1_sonic.camera_service', '--output') == tmp_path
    with pytest.raises(RuntimeError, match='重复服务'):
        start.existing_source({2: p, 3: p}, 'services.g1_sonic.camera_service', '--output')
    frame = 'a'*32
    (tmp_path/frame).mkdir()
    (tmp_path/frame/'rgb.png').touch()
    row = dict(frame_id=frame, received_monotonic=time.monotonic(),
               source='live_camera', host_boot_id=start.BOOT_ID)
    start.write(tmp_path/'latest.json', row)
    assert start.camera_ready(tmp_path)
    start.write(tmp_path/'latest.json', row | dict(host_boot_id='old-boot'))
    assert not start.camera_ready(tmp_path)
    start.write(tmp_path/'latest.json', row | dict(received_monotonic=time.monotonic()-3))
    assert not start.camera_ready(tmp_path)


@pytest.mark.parametrize('change', ['none', 'old_boot', 'stale', 'fault', 'hot', 'incomplete'])
def test_telemetry_requires_current_healthy_29_motors(tmp_path, change):
    row = dict(host_boot_id=start.BOOT_ID, monotonic=time.monotonic(),
               motors=[dict(motorstate=0, temperature_case_winding_c=[40, 60]) for _ in range(29)])
    if change == 'old_boot':
        row['host_boot_id'] = 'old'
    elif change == 'stale':
        row['monotonic'] -= 2
    elif change == 'fault':
        row['motors'][0]['motorstate'] = 512
    elif change == 'hot':
        row['motors'][0]['temperature_case_winding_c'] = [40, 130]
    elif change == 'incomplete':
        row['motors'].pop()
    path = tmp_path/'telemetry.json'
    start.write(path, row)
    assert start.telemetry_ready(path) == (change == 'none')


def test_existing_model_is_verified_without_starting_duplicate(tmp_path, monkeypatch):
    launcher = start.Startup(SimpleNamespace(run=tmp_path), '/tmp/unused.sock')
    monkeypatch.setattr(start, 'port_open', lambda _: True)
    monkeypatch.setattr(launcher, 'start', lambda *a: pytest.fail('duplicate model'))
    calls = []
    def verify(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"ready":true,"rtc_config":{"rtc_delay_steps":8}}\n', stderr='')
    monkeypatch.setattr(start.subprocess, 'run', verify)
    launcher.ensure_model()
    assert len(calls) == 1 and calls[0][1]['timeout'] == 45
    assert 'get_rax_identity' in calls[0][0][-1]
    assert start.read(tmp_path/'model-ready.json')['ready']


def test_failed_start_releases_only_owned_ui_before_control(tmp_path):
    launcher = start.Startup(SimpleNamespace(run=tmp_path), '/tmp/unused.sock')
    stopped = []
    def child(name):
        return SimpleNamespace(poll=lambda: None, terminate=lambda: stopped.append(name), wait=lambda **kw: 0)
    launcher.children = {name: child(name) for name in ['model', 'camera', 'telemetry', 'frontend', 'ui']}
    launcher.release_failed_ui()
    assert stopped == ['ui', 'frontend']


def fake_launch(tmp_path, monkeypatch, fault=False):
    run = tmp_path/'runtime/g1/new'
    run.mkdir(parents=True)
    (run/'experiment.env').write_text('export EXAMPLE=1\n')
    monkeypatch.setattr(start, 'ROOT', tmp_path)
    monkeypatch.setattr(start, 'port_open', lambda _: False)
    monkeypatch.setattr(start.terminals, 'snapshot', lambda: {})
    launcher = start.Startup(SimpleNamespace(run=run, ui_port=8782), '/tmp/test-unused.sock')
    calls = []
    monkeypatch.setattr(launcher, 'ensure_model', lambda: calls.append('model'))
    monkeypatch.setattr(launcher, 'sensors', lambda: calls.append('sensors'))
    monkeypatch.setattr(launcher, 'mcp_ready', lambda: True)
    def ui_ready():
        launcher.url = 'http://127.0.0.1:8782/?token=test-only'
        return True
    monkeypatch.setattr(launcher, 'ui_ready', ui_ready)
    def owner_ready(initialized=False):
        calls.append('owner-initialized' if initialized else 'standing-ready')
        if fault and not initialized:
            raise RuntimeError('owner fault')
        return True
    monkeypatch.setattr(launcher, 'owner_ready', owner_ready)
    def spawn(name, command):
        calls.append(name)
        if name == 'ui_agent':
            start.write(run/'ui-agent-status.json', dict(phase='idle'))
    monkeypatch.setattr(launcher, 'start', spawn)
    monkeypatch.setattr(start.subprocess, 'run', lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(start.subprocess, 'Popen', lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(start.signal, 'pidfd_send_signal', lambda *a: pytest.fail('signal during startup'))
    return launcher, calls


def test_two_windows_agent_only_after_standing_no_automatic_vla(tmp_path, monkeypatch):
    launcher, calls = fake_launch(tmp_path, monkeypatch)
    launcher.launch()
    windows = [c for c in calls if isinstance(c, list) and c[0] == 'gnome-terminal']
    assert len(windows) == 2
    assert windows[0][-1].endswith('window-owner.sh')
    assert windows[1][-1].endswith('window-native.sh')
    assert calls.index('owner-initialized') < calls.index(windows[1])
    assert calls.index('standing-ready') < calls.index('ui_agent')
    ready = start.read(launcher.run/'startup-ready.json')
    assert ready['ready'] and not ready['automatic_vla_prepare'] and not ready['automatic_native_stop']
    assert (tmp_path/'runtime/g1/current.env').read_text() == 'export EXAMPLE=1\n'


def test_owner_fault_does_not_launch_agent_or_stop_control(tmp_path, monkeypatch):
    launcher, calls = fake_launch(tmp_path, monkeypatch, fault=True)
    with pytest.raises(RuntimeError, match='owner fault'):
        launcher.launch()
    assert 'ui_agent' not in calls
    assert not (launcher.run/'startup-ready.json').exists()
    assert not (tmp_path/'runtime/g1/current.env').exists()
    launcher.children = {'frontend': SimpleNamespace(poll=lambda: None,
                         terminate=lambda: pytest.fail('cleanup after control launch'))}
    launcher.release_failed_ui()


def test_read_only_check_cannot_prepare_or_write_error(tmp_path, monkeypatch):
    monkeypatch.setattr(start, 'ROOT', tmp_path)
    monkeypatch.setattr(start, 'preflight', lambda _: None)
    monkeypatch.setattr(start.terminals, 'snapshot', lambda: {1: proc(1, 'g1_field')})
    monkeypatch.setattr(start, 'prepare_run', lambda _: pytest.fail('prepared'))
    run = tmp_path/'runtime/g1/existing'
    run.mkdir(parents=True)
    assert start.main(['--check', '--run', str(run)]) == 1
    assert list(run.iterdir()) == []


def test_prepare_only_never_starts_services_or_retires_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(start, 'ROOT', tmp_path)
    monkeypatch.setattr(start, 'preflight', lambda _: None)
    called = []
    monkeypatch.setattr(start, 'prepare_run', lambda args: called.append(args.run))
    monkeypatch.setattr(start, 'retirement_plan', lambda *a: pytest.fail('retirement'))
    monkeypatch.setattr(start.Startup, 'launch', lambda _: pytest.fail('startup'))
    assert start.main(['--prepare-only']) == 0
    assert len(called) == 1
