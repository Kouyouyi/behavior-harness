#!/usr/bin/env python3
"""Wait for sealed zero-shot failures, then export the precommitted single experts.

Never starts/stops simulation or an Agent, reads demonstration content, resumes
an export, or retries an exporter. Persistent reservations make restart fail
closed; recovery from a supervisor/export error is a manual decision.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_behavior_experiment import (DEFAULT_PROTOCOL, classify_episode,
    protocol_tasks, read_json, read_optional, save, sha256, utc_now, valid_failed_zero_shot)
from scripts.export_one_behavior_expert import _fixed_task

TASKS = ('putting_away_toys', 'turning_on_radio')


def gate_task(output, task, protocol):
    """The existing failure validator always gates access to export metadata."""
    try:
        return 'eligible', valid_failed_zero_shot(output, task, protocol)
    except (ValueError, OSError) as exc:
        reason = str(exc)
    attempts = list((output / task).glob('zero_shot_*'))
    if not attempts or any(not (p / 'trial_record.json').is_file() for p in attempts):
        return 'waiting', {'reason': reason}
    try:
        records = [(read_json(p / 'trial_record.json'), p) for p in attempts]
        if any(r.get('task') != task or r.get('phase') != 'zero_shot' for r, _ in records):
            return 'waiting', {'reason': 'Zero-shot provenance mismatch'}
        record, directory = max(records, key=lambda pair: pair[0].get('started_at_utc', ''))
        if record.get('protocol_sha256') != sha256(protocol):
            return 'waiting', {'reason': 'Zero-shot protocol mismatch'}
        checked = classify_episode(read_optional(directory / 'agent/agent_timing.json'),
            read_optional(directory / 'episode_closed.json'),
            agent_exit=record.get('agent_exit_code'), worker_exit=record.get('worker_exit_code'),
            agent_started=record.get('agent_started', False),
            supervisor_error=record.get('supervisor_error'),
            worker_error=read_optional(directory / 'worker_error.json'),
            finalization_error=read_optional(directory / 'finalization_error.json'))
        if record.get('status') == 'task_success' and checked['status'] == 'task_success':
            return 'not_required', {'record': str(directory / 'trial_record.json'),
                'record_sha256': sha256(directory / 'trial_record.json')}
    except (ValueError, OSError) as exc:
        reason = str(exc)
    return 'waiting', {'reason': reason}


def existing_export(output, task, episode):
    """Metadata only, called after a valid failure; any existing directory blocks launch."""
    directories = sorted((output / task).glob('expert_episode_*'))
    if not directories:
        return None
    expected = output / task / f'expert_episode_{episode}'
    complete = expected / 'export_complete.json'
    status = 'existing_export_requires_manual_review'
    if directories == [expected] and complete.is_file():
        try:
            meta = read_json(complete)
        except (ValueError, OSError):
            meta = {}
        if (meta.get('status') == 'export_verified' and meta.get('source_count') == 1
                and meta.get('task_name') == task
                and meta.get('fixed_expert', {}).get('episode_index') == episode):
            status = 'existing_complete_no_reexport'
    return {'status': status, 'existing_paths': [str(p) for p in directories],
            'automatic_resume': False}


def relevant_processes():
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            args = [p.decode(errors='replace') for p in (directory / 'cmdline').read_bytes().split(b'\0') if p]
        except OSError:
            continue
        # Exact argv basename prevents matching a shell command describing the script.
        names = {Path(arg).name for arg in args}
        if 'wait_behavior_expert_exports.py' in names:
            found.append({'pid': int(directory.name), 'kind': 'supervisor'})
        elif 'export_one_behavior_expert.py' in names and '--task' in args:
            i = args.index('--task') + 1
            if i < len(args) and args[i] in (*TASKS, '0', '54'):
                found.append({'pid': int(directory.name), 'kind': 'exporter', 'task': args[i]})
    return found


def run(output, poll_seconds):
    protocol = output / 'protocol.json'
    _, rows = protocol_tasks(protocol)
    by_name = {row['task_name']: row for row in rows}
    for task in TASKS:
        fixed = _fixed_task(task)
        row = by_name[task]
        if row['task_index'] != fixed['task_index'] or any(
                row.get('fixed_expert', {}).get(k) != fixed[k]
                for k in ('episode_index', 'raw_episode_id', 'frames')):
            raise ValueError('Protocol differs from the precommitted single-expert mapping')
    work = output / 'expert_export_supervisor'
    work.mkdir(exist_ok=True)
    with (work / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = work / 'status.json'
        if path.exists():
            raise FileExistsError('Supervisor state already exists; no automatic relaunch or re-export')
        duplicates = relevant_processes()
        if duplicates:
            raise RuntimeError(f'Existing supervisor/exporter: {duplicates}')
        started = time.perf_counter_ns()
        state = {'schema_version': 1, 'status': 'running', 'pid': os.getpid(),
            'started_at_utc': utc_now(), 'started_perf_counter_ns': started,
            'protocol': str(protocol), 'protocol_sha256': sha256(protocol),
            'timing_scope': 'Independent export supervision; never part of experiment execution timing. Task waits/exports may overlap.',
            'automatic_export_retries': False, 'reads_expert_content': False,
            'tasks': {task: {'status': 'waiting', 'episode_index': by_name[task]['fixed_expert']['episode_index'],
                'waiting_started_perf_counter_ns': started, 'waiting_started_at_utc': utc_now()}
                for task in TASKS}}
        children = {}
        save(path, state)
        while True:
            for task, entry in state['tasks'].items():
                if entry['status'] == 'waiting':
                    if sha256(protocol) != state['protocol_sha256']:
                        entry.update(status='protocol_changed_requires_manual_review')
                        continue
                    status, proof = gate_task(output, task, protocol)
                    entry['last_gate'] = proof
                    if status == 'waiting':
                        continue
                    now = time.perf_counter_ns()
                    entry.update(waiting_ended_perf_counter_ns=now, waiting_ended_at_utc=utc_now(),
                        waiting_wall_seconds=(now - entry['waiting_started_perf_counter_ns']) / 1e9)
                    if status == 'not_required':
                        entry.update(status='not_required', reason='Valid official zero-shot success; no expert accessed')
                        continue
                    entry['zero_shot_failure'] = proof
                    existing = existing_export(output, task, entry['episode_index'])
                    if existing:
                        entry.update(existing)
                        continue
                    log_path = work / f'{task}.export.log'
                    command = [sys.executable, str(ROOT / 'scripts/export_one_behavior_expert.py'),
                        '--output', str(output), '--task', task]
                    # Reserve before spawning; after a crash, never launch a second exporter.
                    entry.update(status='launch_reserved', command=command, log=str(log_path),
                        export_started_perf_counter_ns=time.perf_counter_ns(), export_started_at_utc=utc_now())
                    save(path, state)
                    try:
                        with log_path.open('x') as log:
                            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
                        children[task] = process
                        entry.update(status='exporting', exporter_pid=process.pid)
                    except Exception as exc:
                        ended = time.perf_counter_ns()
                        entry.update(status='export_failed_requires_manual_review', error=f'{type(exc).__name__}: {exc}',
                            export_ended_perf_counter_ns=ended, export_ended_at_utc=utc_now(),
                            export_wall_seconds=(ended - entry['export_started_perf_counter_ns']) / 1e9)
                if entry['status'] == 'exporting':
                    code = children[task].poll()
                    if code is None:
                        continue
                    now = time.perf_counter_ns()
                    entry.update(export_exit_code=code, export_ended_perf_counter_ns=now,
                        export_ended_at_utc=utc_now(),
                        export_wall_seconds=(now - entry['export_started_perf_counter_ns']) / 1e9)
                    existing = existing_export(output, task, entry['episode_index'])
                    if code == 0 and existing and existing['status'] == 'existing_complete_no_reexport':
                        complete = output / task / f'expert_episode_{entry["episode_index"]}' / 'export_complete.json'
                        entry.update(status='export_complete', export_complete=str(complete),
                            export_complete_sha256=sha256(complete))
                    else:
                        entry.update(status='export_failed_requires_manual_review', reason='Exporter exit/verified marker did not pass; no retry')
            state['updated_at_utc'] = utc_now()
            state['updated_perf_counter_ns'] = time.perf_counter_ns()
            if not any(e['status'] in ('waiting', 'exporting') for e in state['tasks'].values()):
                state['status'] = 'finished'
                state['supervisor_wall_seconds'] = (time.perf_counter_ns() - started) / 1e9
                save(path, state)
                return 0
            save(path, state)
            time.sleep(poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, default=DEFAULT_PROTOCOL.parent)
    parser.add_argument('--poll-seconds', type=float, default=5)
    args = parser.parse_args(argv)
    if not 1 <= args.poll_seconds <= 60:
        parser.error('--poll-seconds must be 1..60')
    return run(args.output.expanduser().resolve(), args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
