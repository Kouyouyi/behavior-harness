#!/usr/bin/env python3
"""Run fresh SeekAct/BEHAVIOR episodes; never resume or replay physical actions.

The unprivileged supervisor starts an isolated privileged Isaac worker, then a
Codex owner through the existing MCP frontend. Only the supervisor may _close
that worker. No expert file is opened until a sealed valid zero-shot failure is
verified. This module does not import Isaac.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from services.behavior_ap.backend import request
from services.behavior_ap.timing import WallClockLedger
from services.controller.config import Configuration

REPO_ROOT = ROOT.parent
DEFAULT_PROTOCOL = REPO_ROOT / 'configs/seekact/three_tasks.json'
SIM_ROOT = Path(os.environ.get('SEEKACT_SIM_ROOT', str(REPO_ROOT.parent / 'behavior-ap'))).expanduser().resolve()
SIM_PYTHON = Path(os.environ.get('SEEKACT_SIM_PYTHON', sys.executable))
AGENT_PYTHON = Path(os.environ.get('SEEKACT_AGENT_PYTHON', sys.executable))
SUDO_HELPER = Path(os.environ['SEEKACT_SUDO_HELPER']) if os.environ.get('SEEKACT_SUDO_HELPER') else None
GPU_LOCK = Path(os.environ.get('SEEKACT_GPU_LOCK', '/tmp/seekact-gpu.lock'))
ASSET_ROOT = Path(os.environ.get('BEHAVIOR_ASSET_ROOT', str(Path.home() / 'datasets/BEHAVIOR-1K')))
PHASES = ('zero_shot', 'one_expert_retry')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_optional(path):
    return read_json(path) if Path(path).is_file() else None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_tasks(path, selected=None):
    protocol = read_json(path)
    for key, expected in (('render_profile', 'fast'), ('variant', 'original'),
                          ('split', 'public_test'), ('instance_id', 301)):
        if protocol.get(key) != expected:
            raise ValueError(f'Protocol {key} must be {expected!r}')
    tasks = protocol.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Protocol needs its fixed task list')
    names = []
    for row in tasks:
        name = row.get('task_name', '')
        if not re.fullmatch(r'[a-z][a-z0-9_]*', name) or not str(row.get('prompt', '')).strip():
            raise ValueError('Every protocol task needs a canonical name and full prompt')
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError('Duplicate protocol task')
    if selected is not None:
        matches = [row for row in tasks if selected == row['task_name'] or selected == str(row['task_index'])]
        if len(matches) != 1:
            raise ValueError(f'Task {selected!r} is not uniquely present in this protocol')
        tasks = matches
    return protocol, tasks


def classify_episode(agent, closed, *, agent_exit, worker_exit, agent_started,
                     supervisor_error=None, worker_error=None, finalization_error=None):
    """Keep raw official success separate from whether an evaluation is valid."""
    problems = []
    if supervisor_error:
        problems.append('supervisor_error')
    if worker_exit != 0:
        problems.append('worker_nonzero_exit_or_still_running')
    if worker_error is not None or finalization_error is not None:
        problems.append('worker_or_video_finalization_error')
    if not isinstance(closed, dict) or closed.get('status') != 'closed':
        problems.append('missing_closed_episode')
        closed = closed if isinstance(closed, dict) else {}
    steps = closed.get('steps')
    limit=closed.get('max_steps',10000)
    if (type(steps) is not int or steps < 0 or (limit is not None and steps > limit)
            or closed.get('execution_integrity') is not True
            or type(closed.get('submitted')) is not int or type(closed.get('acknowledged')) is not int
            or closed.get('submitted') != steps or closed.get('acknowledged') != steps
            or closed.get('uncertain_action')):
        problems.append('action_ack_integrity_not_verified')
    if closed.get('errors'):
        problems.append('simulator_errors')
    vla = closed.get('vla_evidence') or {}
    if vla.get('infrastructure_errors'):
        problems.append('vla_model_tool_infrastructure_error')
    if vla.get('last_job') and not vla['last_job'].get('done'):
        problems.append('unfinished_vla_job')
    if vla and not any(j.get('executed_steps', 0) > 0 for j in vla.get('completed_jobs', [])):
        problems.append('requested_vla_tools_not_exercised')
    if vla.get('vla_only'):
        jobs=vla.get('completed_jobs', [])
        if sum(j.get('executed_steps',0) for j in jobs) != steps:
            problems.append('vla_only_action_attribution_incomplete')
        if any(j.get('kind')=='champion' and j.get('executed_steps',0)>0 for j in jobs) and not vla.get('handoffs'):
            problems.append('champion_executed_without_visual_handoff')
    if closed.get('render_profile') != 'fast' or type(closed.get('official_success')) is not bool:
        problems.append('invalid_environment_or_success_evidence')
    videos = closed.get('videos') or {}
    if not isinstance(videos, dict):
        problems.append('invalid_video_evidence')
        videos = {}
    if videos.get('passed') is not True or videos.get('expected_frames') != steps:
        problems.append('video_integrity_not_verified')
    elif steps:
        if videos.get('status') != 'verified' or videos.get('playable_video_available') is not True:
            problems.append('no_verified_playable_video')
        for role in ('robot_cameras', 'third_person'):
            entries = videos.get('videos', {})
            meta = entries.get(role, {}) if isinstance(entries, dict) else {}
            if (not isinstance(meta, dict) or meta.get('full_decode_passed') is not True
                    or type(meta.get('actual_frames')) is not int or meta.get('actual_frames') != steps):
                problems.append(f'{role}_full_decode_not_verified')
    elif videos.get('status') != 'no_frames':
        problems.append('zero_step_video_status_missing')
    agent_ok = (agent_started and agent_exit == 0 and isinstance(agent, dict)
                and agent.get('status') == 'agent_completed' and agent.get('exit_code') == 0)
    expected_wall_timeout=(closed.get('termination_reason')=='wall_time_limit' and closed.get('wall_limit_seconds') is not None and isinstance(agent,dict) and agent.get('timed_out') is True)
    agent_ok=agent_ok or (agent_started and expected_wall_timeout)
    if not agent_ok:
        problems.append('agent_did_not_complete_normally' if agent_started else 'agent_not_started')
    valid = not problems
    if valid:
        status = 'task_success' if closed['official_success'] else 'task_timeout' if closed.get('termination_reason')=='wall_time_limit' else 'task_failure'
    elif agent_started and not agent_ok:
        status = 'agent_infrastructure_error'
    else:
        status = 'infrastructure_error'
    return {'status': status, 'evaluation_valid': valid, 'reasons': problems,
            'official_success': closed.get('official_success'), 'steps': steps,
            'task_success': valid and closed.get('official_success') is True,
            'agent_status': agent.get('status') if isinstance(agent, dict) else None,
            'videos': videos}


def valid_failed_zero_shot(output, task, protocol_path=None):
    """Use the newest sealed zero-shot, refusing an unsealed newer attempt."""
    base = Path(output) / task
    attempts = sorted(base.glob('zero_shot_*'), key=lambda p: p.stat().st_mtime_ns)
    if not attempts:
        raise ValueError('Experience retry requires an existing sealed zero-shot failure')
    records = []
    for directory in attempts:
        path = directory / 'trial_record.json'
        if not path.is_file():
            raise ValueError(f'Unsealed zero-shot evidence must be resolved first: {directory}')
        record = read_json(path)
        if record.get('task') != task or record.get('phase') != 'zero_shot':
            raise ValueError('Zero-shot provenance does not match this task')
        records.append((record.get('started_at_utc', ''), path, record))
    _, path, record = max(records, key=lambda row: row[0])
    if protocol_path is not None and record.get('protocol_sha256') != sha256(protocol_path):
        raise ValueError('Zero-shot protocol differs from the requested experience retry')
    directory = path.parent
    checked = classify_episode(read_optional(directory / 'agent/agent_timing.json'),
        read_optional(directory / 'episode_closed.json'),
        agent_exit=record.get('agent_exit_code'), worker_exit=record.get('worker_exit_code'),
        agent_started=record.get('agent_started', False), supervisor_error=record.get('supervisor_error'),
        worker_error=read_optional(directory / 'worker_error.json'),
        finalization_error=read_optional(directory / 'finalization_error.json'))
    if record.get('status') != 'task_failure' or checked['status'] != 'task_failure':
        raise ValueError('Experience is allowed only after the latest valid zero-shot task failure')
    return {'record': str(path), 'record_sha256': sha256(path),
            'agent_timing_sha256': sha256(directory / 'agent/agent_timing.json'),
            'episode_closed_sha256': sha256(directory / 'episode_closed.json')}


def prepare_experience(path, destination, zero_shot):
    # Caller must obtain zero_shot through valid_failed_zero_shot BEFORE this read.
    if zero_shot is None:
        raise ValueError('Zero-shot failure provenance is required before reading experience')
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in ('.md', '.txt') or not path.is_file():
        raise ValueError('Experience must be one UTF-8 .md/.txt expert summary')
    raw = path.read_bytes()
    raw.decode('utf-8')
    if not raw.strip() or len(raw) > 128 * 1024:
        raise ValueError('Experience must be nonempty and at most 128 KiB')
    destination.write_bytes(raw)
    return {'source': str(path), 'snapshot': str(destination), 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'sources_permitted': 1, 'zero_shot_failure': zero_shot}


def worker_command(config_path):
    assignments = {
        'PYTHONPATH': os.pathsep.join((str(ROOT), str(SIM_ROOT))),
        'OMNI_KIT_ACCEPT_EULA': 'YES', 'OMNIGIBSON_HEADLESS': '1',
        'OMNIGIBSON_DATA_PATH': str(ASSET_ROOT), 'BEHAVIOR_GPU_LOCK_PATH': str(GPU_LOCK),
        'PYTHONUNBUFFERED': '1',
    }
    if os.environ.get('BEHAVIOR_ALLOW_TODESK_DISPLAY')=='1':
        assignments['BEHAVIOR_ALLOW_TODESK_DISPLAY']='1'
    prefix = ['bash', str(SUDO_HELPER)] if SUDO_HELPER else []
    return prefix + ['env', *[f'{k}={v}' for k, v in assignments.items()],
            str(SIM_PYTHON), '-m', 'services.behavior_ap.worker', '--config', str(config_path)]


def wait_for_ready(process, trial, socket_path, task, *, poll_seconds=2):
    """Unbounded initialization/GPU wait; never terminate another GPU job."""
    while True:
        if process.poll() is not None:
            raise RuntimeError(f'Worker exited before readiness (exit={process.returncode})')
        ready = read_optional(trial / 'worker_ready.json')
        if ready is not None and socket_path.exists():
            expected = {'status': 'ready', 'task': task, 'render_profile': 'fast',
                        'split': 'public_test', 'instance_id': 301, 'step': 0}
            if any(ready.get(k) != v for k, v in expected.items()):
                raise RuntimeError('Worker readiness differs from the fixed original-scene protocol')
            try:
                state, _ = request(socket_path, 'behavior_state', timeout=30)
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(poll_seconds)
                continue
            if state.get('task') != task or state.get('step') != 0 or state.get('official_success') is not False:
                raise RuntimeError('Initial state is not this fresh unfinished task')
            return ready
        time.sleep(poll_seconds)


def finalize_worker(process, socket_path, ledger, *, poll_seconds=2):
    """Queue close behind any in-flight action; no process signal or motion replay."""
    close_id = uuid.uuid4().hex
    accepted = False
    while process.poll() is None:
        if not accepted and socket_path.exists():
            try:
                value, _ = request(socket_path, '_close', request_id=close_id, timeout=300)
                accepted = value.get('closing') is True
                ledger.emit('close_ack', request_id=close_id, value=value)
            except Exception as exc:
                # _close is a supervisor-only non-motion operation. Same ID is retained.
                ledger.emit('close_transport_error', request_id=close_id,
                            error=f'{type(exc).__name__}: {exc}', physical_replay=False)
        time.sleep(poll_seconds)
    return process.wait()


def run_trial(output, protocol_path, protocol, task, phase, attempt, *, experience=None,
              cancelled=None, session_seconds=14400):
    if phase not in PHASES or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', attempt):
        raise ValueError('Invalid experiment phase or fresh-attempt label')
    cancelled = cancelled if cancelled is not None else {'requested': False}
    output = Path(output).resolve()
    trial = output / task['task_name'] / f'{phase}_{attempt}'
    if trial.exists():
        raise FileExistsError(f'Existing attempt; never resume or overwrite: {trial}')
    zero = valid_failed_zero_shot(output, task['task_name'], protocol_path) if phase == 'one_expert_retry' else None
    if (phase == 'one_expert_retry') != (experience is not None):
        raise ValueError('Only one_expert_retry requires and accepts --experience')
    trial.parent.mkdir(parents=True, exist_ok=True)
    trial.mkdir()  # Atomic reservation; no exist_ok.
    ledger = WallClockLedger(trial / 'supervisor_timing', expected_categories=(
        'worker_startup', 'agent_session', 'worker_finalize'))
    worker = agent = None
    agent_started = False
    agent_exit = worker_exit = None
    error = None
    socket_path = Path('/tmp') / ('symb-bhv-' + uuid.uuid4().hex[:20] + '.sock')
    wall_limit=protocol.get('wall_limit_seconds')
    deadline_monotonic=None if wall_limit is None else time.monotonic()+wall_limit
    max_steps=protocol.get('max_steps',10000)
    if max_steps is None and wall_limit is None:
        if not protocol.get('unlimited_duration_authorized'):raise ValueError('Unlimited execution requires explicit operator authorization')
        session_seconds=None
    if wall_limit is not None:session_seconds=wall_limit
    record = {'schema_version': 1, 'task': task['task_name'], 'task_index': task['task_index'],
        'phase': phase, 'attempt': attempt, 'trial': str(trial), 'started_at_utc': utc_now(),
        'protocol_path': str(protocol_path), 'protocol_sha256': sha256(protocol_path),
        'render_profile': 'fast', 'variant': 'original', 'split': 'public_test', 'instance_id': 301,
        'max_steps': max_steps, 'wall_limit_seconds':wall_limit,'session_seconds': session_seconds, 'socket_path': str(socket_path),
        'automatic_motion_replay': False, 'fresh_initial_scene': True, 'experience': None,
        'requested_seed': protocol.get('seed')}
    save(trial / 'trial_started.json', record)
    (trial / 'protocol_snapshot.json').write_bytes(Path(protocol_path).read_bytes())
    record['runtime_revisions'] = {}
    for label, repository in (('agent', ROOT), ('simulator', SIM_ROOT)):
        record['runtime_revisions'][label] = {
            'head': subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip(),
            'working_tree_status': subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain'], text=True),
        }
    try:
        if experience is not None:
            record['experience'] = prepare_experience(experience, trial / 'single_expert_experience.md', zero)
        worker_config = {'output': str(trial), 'socket_path': str(socket_path),
            'uid': os.getuid(), 'gid': os.getgid(), 'task': task['task_name'], 'prompt': task['prompt'],
            'max_steps': max_steps, 'seed': protocol.get('seed'),
            'deadline_monotonic':deadline_monotonic,'wall_limit_seconds':wall_limit}
        vla_tools = protocol.get('vla_tools', {})
        if vla_tools.get('enabled'):
            vla_tools = dict(vla_tools, champion_skills=task.get('champion_skills', {}),
                champion_cross_task_transfer=task.get('champion_cross_task_transfer', False))
            worker_config['vla_tools'] = vla_tools
            record['control_mode'] = ('vla_only_ap_search_champion_execution' if vla_tools.get('vla_only')
                                      else 'native_tools_plus_ap_and_champion_vla')
            record['vla_tools'] = vla_tools
        save(trial / 'worker_config.json', worker_config)
        agent_root = trial / 'agent'
        agent_root.mkdir()
        config = Configuration(root=agent_root, task=task['task_name'], task_description=task['prompt'],
            robot='behavior_ap', backend_options={'socket_path': str(socket_path), 'max_steps': max_steps,'deadline_monotonic':deadline_monotonic,
                **({'vla_tools': vla_tools} if vla_tools.get('enabled') else {})},
            agent_provider='codex', observation_profile='rgbd', session_seconds=session_seconds,
            subagents=0)
        config_path = trial / 'agent_config.json'
        save(config_path, config.to_dict())
        command = worker_command(trial / 'worker_config.json')
        save(trial / 'launch_commands.json', {'worker': command, 'credentials_copied': False})
        with ledger.span('worker_startup'):
            with (trial / 'worker.log').open('x') as log:
                worker = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(trial / 'processes.json', {'worker_supervisor_pid': worker.pid})
            record['worker_ready'] = wait_for_ready(worker, trial, socket_path, task['task_name'])
        if not cancelled['requested']:
            command = [str(AGENT_PYTHON), '-m', 'harness.codex_cli.behavior', '--config', str(config_path)]
            if record['experience']:
                command += ['--experience', record['experience']['snapshot']]
            commands = read_json(trial / 'launch_commands.json')
            save(trial / 'launch_commands.json', dict(commands, agent=command))
            env = dict(os.environ, PYTHONPATH=str(ROOT), RAX_ARTIFACT_ROOT=str(output), PYTHONUNBUFFERED='1')
            with ledger.span('agent_session'):
                with (trial / 'agent_launcher.log').open('x') as log:
                    agent = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                agent_started = True
                save(trial / 'processes.json', {'worker_supervisor_pid': worker.pid, 'agent_pid': agent.pid})
                sent_interrupt = False
                while agent.poll() is None:
                    if cancelled['requested'] and not sent_interrupt:
                        try:
                            agent.send_signal(signal.SIGTERM)  # Only our unprivileged launcher; never worker.
                        except ProcessLookupError:
                            pass
                        sent_interrupt = True
                        ledger.emit('agent_interrupt_requested', pid=agent.pid)
                    time.sleep(1)
                agent_exit = agent.wait()
        else:
            error = {'type': 'CancellationRequested', 'message': 'Cancelled before Agent startup; worker will close normally'}
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        ledger.emit('supervisor_error', **error)
        # An unexpected supervisor error must not orphan an active Agent owner.
        if agent is not None and agent.poll() is None:
            try:
                agent.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            agent_exit = agent.wait()
    finally:
        if worker is not None:
            try:
                with ledger.span('worker_finalize'):
                    worker_exit = finalize_worker(worker, socket_path, ledger)
            except BaseException as exc:
                error = {'type': 'SupervisorFinalizationError', 'message': str(exc), 'previous': error,
                         'worker_was_not_killed': True}
        evidence_errors = []
        def evidence(path):
            try:
                return read_optional(path)
            except (OSError, ValueError) as exc:
                evidence_errors.append({'path': str(path), 'error': f'{type(exc).__name__}: {exc}'})
                return None
        agent_timing = evidence(trial / 'agent/agent_timing.json')
        closed = evidence(trial / 'episode_closed.json')
        worker_error = evidence(trial / 'worker_error.json')
        finalization_error = evidence(trial / 'finalization_error.json')
        if evidence_errors:
            error = {'type': 'EvidenceReadError', 'files': evidence_errors, 'previous': error}
        record.update(ended_at_utc=utc_now(), agent_started=agent_started, agent_exit_code=agent_exit,
                      worker_exit_code=worker_exit, supervisor_error=error,
                      cancellation_requested=cancelled['requested'])
        record.update(classify_episode(agent_timing, closed, agent_exit=agent_exit,
            worker_exit=worker_exit, agent_started=agent_started, supervisor_error=error,
            worker_error=worker_error, finalization_error=finalization_error))
        record['agent_timing'] = agent_timing
        record['episode_closed'] = closed
        record['supervisor_timing'] = ledger.finish({'status': record['status'],
            'worker_times_overlap_do_not_sum': True, 'pure_model_inference_time_measured': False})
        save(trial / 'trial_record.json', record)
    return record


def update_summary(output, protocol_path, *, active=None):
    records = [read_json(p) for p in sorted(Path(output).glob('*/*/trial_record.json'))]
    summary = {'schema_version': 1, 'updated_at_utc': utc_now(), 'protocol': str(protocol_path),
        'active': active, 'attempts': len(records), 'counts': dict(Counter(r['status'] for r in records)),
        'trials': records, 'counting_unit': 'attempt; previous evidence retained, not selected for success',
        'official_success_source': 'worker official BehaviorTask.success / BDDL',
        'timing_note': 'Supervisor, Agent and worker times overlap; never sum them as independent elapsed time'}
    save(Path(output) / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--phase', choices=PHASES, default='zero_shot')
    parser.add_argument('--task', help='Protocol task name or numerical task index; zero-shot defaults to all in order')
    parser.add_argument('--attempt', default='v1')
    parser.add_argument('--experience', type=Path)
    parser.add_argument('--wait-for-runner', action='store_true',
                        help='Wait for runner.lock; recheck fresh attempt and retry eligibility after acquisition')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.attempt):
        parser.error('--attempt must be a simple fresh directory label')
    if args.phase == 'one_expert_retry' and (args.task is None or args.experience is None):
        parser.error('one_expert_retry needs exactly one --task and --experience')
    if args.phase == 'zero_shot' and args.experience is not None:
        parser.error('zero_shot cannot read --experience')
    protocol_path = args.protocol.expanduser().resolve()
    protocol, tasks = protocol_tasks(protocol_path, args.task)
    output = protocol_path.parent
    os.environ['RAX_ARTIFACT_ROOT'] = str(output)
    cancelled = {'requested': False}
    def cancel(*_):
        cancelled['requested'] = True
    old = {sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (output / 'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if args.wait_for_runner else fcntl.LOCK_NB))
            for task in tasks:
                trial = output / task['task_name'] / f'{args.phase}_{args.attempt}'
                if trial.exists():
                    raise FileExistsError(f'Existing attempt; choose a fresh --attempt, never resume: {trial}')
            for task in tasks:
                if cancelled['requested']:
                    break
                update_summary(output, protocol_path, active={'task': task['task_name'], 'phase': args.phase, 'attempt': args.attempt})
                record = run_trial(output, protocol_path, protocol, task, args.phase, args.attempt,
                                   experience=args.experience, cancelled=cancelled)
                update_summary(output, protocol_path, active=None)
                print(json.dumps({k: record[k] for k in ('task', 'phase', 'status', 'steps', 'official_success')}, ensure_ascii=False), flush=True)
                if not record['evaluation_valid']:
                    # Preserve evidence; root can fix and request a NEW attempt. No automatic partial replay.
                    return 2
            update_summary(output, protocol_path, active=None)
        return 130 if cancelled['requested'] else 0
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
