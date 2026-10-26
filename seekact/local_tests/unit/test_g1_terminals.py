from pathlib import Path

from scripts import g1_manage_terminals as terminals

ROOT = Path('/repo/runtime/g1')
RUN = ROOT/'E107'


def proc(pid, parent, *argv):
    return terminals.Process(pid, parent, str(pid), argv)


def desktop(role='temperature'):
    return {1: proc(1, 0, '/usr/libexec/gnome-terminal-server'),
            2: proc(2, 1, 'bash', str(RUN/f'window-{role}.sh'))}


def test_only_retired_known_readonly_trees_close():
    processes = desktop() | {3: proc(3, 2, 'script', '-c', 'watch'),
                            4: proc(4, 3, '/venv/bin/python', '-m', 'g1_sonic.temperature_watch', '--run', str(RUN))}
    assert not terminals.candidates(processes, ROOT)[0][3]
    assert terminals.candidates(processes, ROOT, RUN)[0][3] == 'current run'
    processes[5] = proc(5, 4, 'unexpected-service')
    assert terminals.candidates(processes, ROOT)[0][3]
    assert not terminals.candidates(processes, Path('/different'))


def test_never_close_live_control_agent_or_shared_services():
    for argv in [('g1_field', 'eno1'), ('python', '-m', 'g1_sonic.ankle_executor')]:
        assert terminals.candidates(desktop() | {7: proc(7, 0, *argv)}, ROOT)[0][3]
    assert terminals.candidates(desktop('codex') | {3: proc(3, 2, 'codex')}, ROOT)[0][3]
    for role in ('camera', 'model'):
        assert not terminals.candidates(desktop(role), ROOT)[0][3]
        assert terminals.candidates(desktop(role) | {3: proc(3, 2, 'python', 'service.py')}, ROOT)[0][3]
    for role in ('owner', 'native', 'codex'):
        assert not terminals.candidates(desktop(role), ROOT)[0][3]


def test_apply_rechecks_process_tree_and_dry_run_never_signals(monkeypatch):
    initial = desktop('native')
    monkeypatch.setattr(terminals, 'snapshot', lambda: initial)
    monkeypatch.setattr(terminals.os, 'pidfd_open', lambda _: (_ for _ in ()).throw(AssertionError('signal attempted')))
    assert terminals.clean(ROOT)[0]['action'] == 'would_close'
    snapshots = iter([initial, initial | {3: proc(3, 2, 'g1_field')}])
    monkeypatch.setattr(terminals, 'snapshot', lambda: next(snapshots))
    assert terminals.clean(ROOT, apply=True)[0]['reason'] == 'process tree changed'
