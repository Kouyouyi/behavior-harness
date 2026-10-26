#!/usr/bin/env python3
"""Export exactly one fixed expert, only after a sealed valid zero-shot failure.

This entry reads metadata/provenance first. It never selects another episode,
resumes a partial export, analyzes demonstration content, or starts Isaac.
Remote PyArrow and FFmpeg must already be installed; no dependencies are installed.
"""
from __future__ import annotations

import argparse
import fcntl
from fractions import Fraction
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_behavior_experiment import protocol_tasks, save, sha256, utc_now, valid_failed_zero_shot
from services.behavior_ap.timing import WallClockLedger

REMOTE_BASE = '/cs/kyy/behavior_ap_experiment/reports/seekact_exports'
DEFAULT_REMOTE_PYTHON = '/cs/kyy/behavior_ap_experiment/.venv/bin/python'
ROLES = ('head', 'left_wrist', 'right_wrist')
CONTENT_FILES = frozenset({'episode.parquet', 'annotation.json', *(f'videos/{role}.mp4' for role in ROLES)})
REMOTE_FILES = CONTENT_FILES | {'remote_export_started.json', 'remote_export_finished.json'}


def _fixed_task(name):
    # Metadata selection is frozen before outcomes; never search for substitutes.
    rows = {
        'turning_on_radio': {'task_index': 0, 'episode_index': 0, 'raw_episode_id': 10, 'frames': 1956},
        'putting_shoes_on_rack': {'task_index': 22, 'episode_index': 4400, 'raw_episode_id': 220020, 'frames': 7076},
        'putting_away_toys': {'task_index': 54, 'episode_index': 10800, 'raw_episode_id': 540010, 'frames': 13136},
    }
    if name not in rows:
        raise ValueError('Task has no precommitted single-expert selection')
    return dict(rows[name])


def _encode_command(ffmpeg, source, destination, frames):
    return [str(ffmpeg), '-nostdin', '-hide_banner', '-v', 'error', '-xerror', '-n',
            '-threads', '1', '-i', str(source), '-map', '0:v:0',
            '-vf', f'trim=start_frame=0:end_frame={frames},setpts=N/(30*TB)',
            '-frames:v', str(frames), '-r', '30', '-fps_mode', 'cfr',
            '-an', '-sn', '-dn', '-c:v', 'libx264', '-profile:v', 'high',
            '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '20',
            '-movflags', '+faststart', str(destination)]


def _remote_export(plan):
    """Self-contained remote body. Imported locally but invoked only over SSH."""
    from datetime import datetime, timezone
    import hashlib
    import json
    from pathlib import Path
    import shutil
    import subprocess
    import time
    import traceback

    def stamp():
        return datetime.now(timezone.utc).isoformat()

    def write(path, value):
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    def file_meta(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}

    name = plan['task_name']
    fixed = _fixed_task(name)
    if any(type(plan.get(key)) is not int or plan[key] != value for key, value in fixed.items()):
        raise ValueError('Remote request differs from the fixed expert selection')
    parent = Path('/cs/kyy/behavior_ap_experiment/reports/seekact_exports')
    output = Path(plan['remote_directory'])
    if output.parent != parent or not output.name.startswith(f'{name}_episode_{fixed["episode_index"]}_'):
        raise ValueError('Invalid isolated remote export directory')
    suffix = output.name.rsplit('_', 1)[-1]
    if len(suffix) != 32 or any(c not in '0123456789abcdef' for c in suffix):
        raise ValueError('Remote export directory needs a fresh random identity')
    parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Never overwrite, merge with, or resume an existing export.
    started = time.perf_counter_ns()
    durations = {}
    write(output / 'remote_export_started.json', {'status': 'started', 'source_count': 1,
          'started_at_utc': stamp(), 'plan': plan, 'automatic_fallback': False})
    try:
        # These imports and every source read happen only in this post-gate body.
        import pyarrow.parquet as pq
        source_root = Path('/cs/datasets/behavior2026_dual_language')
        chunk = f'chunk-{fixed["task_index"]:03d}'
        parquet_source = source_root / '_source/data' / chunk / 'file-000.parquet'
        annotation_source = (source_root / '_source/annotations' / f'task-{fixed["task_index"]:04d}'
                             / f'episode_{fixed["raw_episode_id"]:08d}.json')
        sources = {'parquet': str(parquet_source), 'annotation': str(annotation_source), 'videos': {}}
        begin = time.perf_counter_ns()
        table = pq.read_table(parquet_source, filters=[('episode_index', '=', fixed['episode_index'])])
        if table.num_rows != fixed['frames']:
            raise ValueError(f'Fixed episode row count mismatch: {table.num_rows}; no substitute is allowed')
        for column in ('episode_index', 'frame_index'):
            if column not in table.column_names:
                raise ValueError(f'Missing required expert column: {column}')
        if set(table.column('episode_index').to_pylist()) != {fixed['episode_index']}:
            raise ValueError('Parquet filter did not isolate exactly the fixed episode')
        table = table.sort_by([('frame_index', 'ascending')])
        if table.column('frame_index').to_pylist() != list(range(fixed['frames'])):
            raise ValueError('Expert frame indices are not exactly 0..frames-1')
        pq.write_table(table, output / 'episode.parquet', compression='zstd')
        durations['parquet_filter_seconds'] = (time.perf_counter_ns() - begin) / 1e9
        begin = time.perf_counter_ns()
        # Copy only the fixed annotation; do not scan or parse neighboring episodes.
        with annotation_source.open('rb') as source, (output / 'annotation.json').open('xb') as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        durations['annotation_copy_seconds'] = (time.perf_counter_ns() - begin) / 1e9
        (output / 'videos').mkdir()
        camera_keys = {'head': 'zed_link_camera_0', 'left_wrist': 'left_realsense_link_camera_0',
                       'right_wrist': 'right_realsense_link_camera_0'}
        for role, camera in camera_keys.items():
            source = source_root / 'videos' / f'observation.rgb.{camera}' / chunk / 'file-000.mp4'
            sources['videos'][role] = str(source)
            command = _encode_command(plan['ffmpeg'], source, output / 'videos' / f'{role}.mp4', fixed['frames'])
            begin = time.perf_counter_ns()
            with (output / f'encode_{role}.stderr.log').open('x') as log:
                result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=log, check=False)
            if result.returncode:
                raise RuntimeError(f'{role}: FFmpeg exited {result.returncode}; see encode_{role}.stderr.log')
            if (output / f'encode_{role}.stderr.log').stat().st_size:
                raise RuntimeError(f'{role}: FFmpeg reported an error; see encode_{role}.stderr.log')
            durations[f'encode_{role}_seconds'] = (time.perf_counter_ns() - begin) / 1e9
        write(output / 'remote_export_finished.json', {'status': 'encoded_for_transfer', 'source_count': 1,
              'ended_at_utc': stamp(), 'wall_clock_not_simulation_time': True,
              'remote_export_seconds': (time.perf_counter_ns() - started) / 1e9, 'phases': durations,
              'local_full_decode_verification_still_required': True})
        names = ['episode.parquet', 'annotation.json', 'remote_export_started.json', 'remote_export_finished.json']
        names += [f'videos/{role}.mp4' for role in camera_keys]
        manifest = {'schema_version': 1, 'source_count': 1, 'task_name': name, **fixed,
                    'source_start_frame': 0, 'source_fps': 30, 'sources': sources,
                    'protocol_sha256': plan['protocol_sha256'],
                    'zero_shot_record_sha256': plan['zero_shot_record_sha256'],
                    'remote_directory': str(output), 'phases': durations,
                    'source_analysis_performed': False, 'automatic_fallback': False,
                    'files': {relative: file_meta(output / relative) for relative in names}}
        write(output / 'remote_manifest.json', manifest)
        return {'manifest': manifest, 'manifest_file': file_meta(output / 'remote_manifest.json')}
    except BaseException as exc:
        write(output / 'remote_export_error.json', {'status': 'failed', 'source_count': 1,
              'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
              'ended_at_utc': stamp(), 'wall_clock_not_simulation_time': True,
              'remote_elapsed_seconds': (time.perf_counter_ns() - started) / 1e9,
              'partial_files_retained': True, 'automatic_fallback': False})
        raise


def remote_command(argv):
    """SSH invokes a shell remotely; quote each argument, never interpolate code."""
    if any(not isinstance(arg, str) or not arg or '\0' in arg for arg in argv):
        raise ValueError('Every remote argv entry must be nonempty text without NUL')
    return ['ssh', '-o', 'BatchMode=yes', 'huoshan', ' '.join(shlex.quote(arg) for arg in argv)]


def make_plan(row, protocol_path, proof, ffmpeg):
    name = row['task_name']
    fixed = _fixed_task(name)
    expert = row.get('fixed_expert', {})
    if type(row.get('task_index')) is not int or row['task_index'] != fixed['task_index']:
        raise ValueError('Protocol task index differs from the frozen expert mapping')
    for key in ('episode_index', 'raw_episode_id', 'frames'):
        if type(expert.get(key)) is not int or expert[key] != fixed[key]:
            raise ValueError(f'Protocol fixed_expert.{key} differs; no fallback is permitted')
    if not isinstance(ffmpeg, str) or not ffmpeg or '\0' in ffmpeg:
        raise ValueError('Invalid remote FFmpeg executable')
    return {'task_name': name, **fixed, 'source_start_frame': 0, 'source_count': 1,
            'remote_directory': f'{REMOTE_BASE}/{name}_episode_{fixed["episode_index"]}_{uuid.uuid4().hex}',
            'protocol_sha256': sha256(protocol_path), 'zero_shot_record_sha256': proof['record_sha256'],
            'ffmpeg': ffmpeg}


def execute_remote_export(plan, destination, remote_python):
    code = '\n\n'.join(inspect.getsource(f) for f in (_fixed_task, _encode_command, _remote_export))
    code += '\nimport json,sys\nprint(json.dumps(_remote_export(json.loads(sys.argv[1])), ensure_ascii=False, allow_nan=False))\n'
    command = remote_command([remote_python, '-c', code, json.dumps(plan, ensure_ascii=False, allow_nan=False)])
    with (destination / 'remote_export.stderr.log').open('x') as log:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=log, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f'Remote expert export failed (exit {result.returncode}); remote evidence is retained at {plan["remote_directory"]}')
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError('Remote export returned no valid manifest; do not retry or select another expert') from exc


def validated_files(envelope, plan):
    if not isinstance(envelope, dict) or not isinstance(envelope.get('manifest'), dict):
        raise ValueError('Remote response has no manifest')
    manifest = envelope['manifest']
    for key in ('task_name', 'task_index', 'episode_index', 'raw_episode_id', 'frames',
                'source_start_frame', 'source_count', 'remote_directory',
                'protocol_sha256', 'zero_shot_record_sha256'):
        if type(manifest.get(key)) is not type(plan[key]) or manifest[key] != plan[key]:
            raise ValueError(f'Remote manifest {key} differs from the gated request')
    if manifest.get('source_fps') != 30 or manifest.get('source_analysis_performed') is not False:
        raise ValueError('Remote source frame rate or analysis boundary differs')
    if manifest.get('automatic_fallback') is not False:
        raise ValueError('Unexpected remote fallback')
    files = manifest.get('files')
    if not isinstance(files, dict) or set(files) != REMOTE_FILES:
        raise ValueError('Remote manifest must contain only the one fixed expert and export metadata')
    files = dict(files, **{'remote_manifest.json': envelope.get('manifest_file')})
    for relative, meta in files.items():
        if (not isinstance(meta, dict) or type(meta.get('bytes')) is not int or meta['bytes'] <= 0
                or not isinstance(meta.get('sha256'), str) or len(meta['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in meta['sha256'])):
            raise ValueError(f'Invalid transfer size/hash for {relative}')
    return files


def extract_checked(stream, destination, expected):
    """Extract only declared regular files; forbid links, traversal and overwrites."""
    destination = Path(destination).resolve()
    seen = set()
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            name = member.name
            pure = PurePosixPath(name)
            if (name not in expected or name in seen or not member.isfile() or pure.is_absolute()
                    or '..' in pure.parts or '\\' in name or str(pure) != name):
                raise ValueError(f'Refusing unsafe, duplicate, or unexpected archive member: {name!r}')
            meta = expected[name]
            if member.size != meta['bytes']:
                raise ValueError(f'Archive size differs from manifest: {name}')
            path = destination.joinpath(*pure.parts)
            parent = destination
            for part in pure.parts[:-1]:
                parent = parent / part
                if parent.is_symlink():
                    raise ValueError('Archive parent must not be a symlink')
                parent.mkdir(exist_ok=True)
            if path.parent.resolve() != destination and destination not in path.parent.resolve().parents:
                raise ValueError('Archive path escapes the export directory')
            digest = hashlib.sha256()
            received = 0
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f'Unreadable regular file: {name}')
            with source, path.open('xb') as target:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    received += len(block)
                    target.write(block)
                    digest.update(block)
            if received != meta['bytes'] or digest.hexdigest() != meta['sha256']:
                raise ValueError(f'Transferred file size/hash mismatch: {name}')
            seen.add(name)
    if seen != set(expected):
        raise ValueError(f'Incomplete expert transfer: missing {sorted(set(expected) - seen)}')
    return {'files': len(seen), 'bytes': sum(meta['bytes'] for meta in expected.values()), 'sha256_verified': True}


def transfer_export(plan, destination, expected):
    command = remote_command(['tar', '-C', plan['remote_directory'], '-cf', '-', '--', *sorted(expected)])
    with (destination / 'transfer.stderr.log').open('x') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log)
        try:
            evidence = extract_checked(process.stdout, destination, expected)
            # Drain only bounded tar padding, without buffering an unbounded stream.
            trailing = 0
            for block in iter(lambda: process.stdout.read(65536), b''):
                trailing += len(block)
                if trailing > 1024 * 1024 or any(block):
                    raise ValueError('Unexpected bytes after the declared expert archive')
            code = process.wait()
            if code:
                raise RuntimeError(f'Expert transfer failed (exit {code}); partial files retained')
            return evidence
        except BaseException:
            if process.poll() is None:
                process.terminate()  # Only this owned non-physical SSH transfer.
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()


def verify_exported_videos(destination, frames, *, ffmpeg='/usr/bin/ffmpeg', ffprobe='/usr/bin/ffprobe'):
    """Metadata + complete CPU decode, never visual/semantic expert analysis."""
    evidence = {'passed': False, 'source_count': 1, 'expected_frames': frames, 'videos': {}}
    for role in ROLES:
        path = Path(destination) / 'videos' / f'{role}.mp4'
        before = path.stat()
        probe = [str(ffprobe), '-v', 'error', '-threads', '1', '-count_frames', '-select_streams', 'v:0',
                 '-show_entries', 'stream=codec_name,profile,codec_tag_string,pix_fmt,nb_read_frames,avg_frame_rate',
                 '-of', 'json', str(path)]
        result = subprocess.run(probe, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
        if result.returncode or result.stderr.strip():
            raise RuntimeError(f'{role}: ffprobe failed: {result.stderr[-4000:]}')
        streams = json.loads(result.stdout).get('streams')
        if not isinstance(streams, list) or len(streams) != 1:
            raise ValueError(f'{role}: expected exactly one selected video stream')
        stream = streams[0]
        codecs = {'codec_name': 'h264', 'profile': 'High', 'codec_tag_string': 'avc1', 'pix_fmt': 'yuv420p'}
        if (any(stream.get(key) != value for key, value in codecs.items())
                or int(stream.get('nb_read_frames', -1)) != frames
                or Fraction(stream.get('avg_frame_rate', '0/1')) != 30):
            raise ValueError(f'{role}: expert codec, exact frame count, or 30fps differs')
        decode = [str(ffmpeg), '-nostdin', '-v', 'error', '-xerror', '-err_detect', 'explode',
                  '-threads', '1', '-i', str(path), '-map', '0:v:0', '-an', '-f', 'null', '-']
        result = subprocess.run(decode, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
        if result.returncode or result.stderr.strip():
            raise RuntimeError(f'{role}: complete FFmpeg decode failed: {result.stderr[-4000:]}')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f'{role}: expert video changed during verification')
        evidence['videos'][role] = {'path': str(path), **codecs, 'actual_frames': frames, 'fps': 30,
                                    'full_decode_passed': True, 'bytes': after.st_size}
    evidence['passed'] = True
    return evidence


def unchanged_gate(output, task, protocol_path, proof, protocol_hash):
    """Fail closed if this task's selected zero-shot or protocol changed."""
    current = valid_failed_zero_shot(output, task, protocol_path)
    if current != proof or sha256(protocol_path) != protocol_hash:
        raise ValueError('Selected-task zero-shot provenance changed during expert export')


def export_one(output, task, *, remote_python=DEFAULT_REMOTE_PYTHON, ffmpeg='ffmpeg',
               local_ffmpeg='/usr/bin/ffmpeg', local_ffprobe='/usr/bin/ffprobe'):
    output = Path(output).expanduser().resolve()
    protocol_path = output / 'protocol.json'
    _, rows = protocol_tasks(protocol_path, task)
    row = rows[0]
    # Check only this task. Healthy zero-shots of other tasks keep running.
    valid_failed_zero_shot(output, row['task_name'], protocol_path)
    parent = output / row['task_name']
    if parent.is_symlink() or parent.resolve().parent != output:
        raise ValueError('Expert task directory must be a direct local directory')
    lock_path = parent / 'expert_export.lock'
    if lock_path.is_symlink():
        raise ValueError('Expert export lock must not be a symlink')
    # Per-task exporter exclusion, plus atomic reservation of the fresh output.
    # Do not acquire runner.lock: it belongs to the ongoing simulation batch.
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof = valid_failed_zero_shot(output, row['task_name'], protocol_path)
        plan = make_plan(row, protocol_path, proof, ffmpeg)
        destination = parent / f'expert_episode_{plan["episode_index"]}'
        destination.mkdir()  # No overwrite or resume; failed evidence remains intact.
        ledger = WallClockLedger(destination / 'export_timing', expected_categories=(
            'expert_export', 'expert_transfer', 'expert_verification'))
        record = {'schema_version': 1, 'status': 'started', 'source_count': 1,
                  'task_name': row['task_name'], 'fixed_expert': _fixed_task(row['task_name']),
                  'started_at_utc': utc_now(), 'destination': str(destination), 'plan': plan,
                  'zero_shot_failure': proof, 'source_analysis_performed': False,
                  'automatic_fallback': False, 'resume_existing_export': False,
                  'lock_scope': 'selected_task_expert_export', 'lock_path': str(lock_path)}
        save(destination / 'export_started.json', record)
        try:
            with ledger.span('expert_export', episode_index=plan['episode_index']):
                unchanged_gate(output, row['task_name'], protocol_path, proof, plan['protocol_sha256'])
                envelope = execute_remote_export(plan, destination, remote_python)
                expected = validated_files(envelope, plan)
            with ledger.span('expert_transfer'):
                unchanged_gate(output, row['task_name'], protocol_path, proof, plan['protocol_sha256'])
                transfer = transfer_export(plan, destination, expected)
            with ledger.span('expert_verification'):
                copied = json.loads((destination / 'remote_manifest.json').read_text())
                if copied != envelope['manifest']:
                    raise ValueError('Transferred manifest differs from the original gated export response')
                videos = verify_exported_videos(destination, plan['frames'],
                                                ffmpeg=local_ffmpeg, ffprobe=local_ffprobe)
                unchanged_gate(output, row['task_name'], protocol_path, proof, plan['protocol_sha256'])
            record.update(status='export_verified', ended_at_utc=utc_now(), transfer=transfer,
                          videos=videos, remote_manifest=copied,
                          wall_timing=ledger.finish({'remote_export_phases_overlap_do_not_sum': copied.get('phases'),
                              'expert_analysis_performed': False, 'status': 'export_verified'}))
            save(destination / 'export_complete.json', record)
            return record
        except BaseException as exc:
            record.update(status='failed', ended_at_utc=utc_now(), error_type=type(exc).__name__, error=str(exc),
                          partial_local_and_remote_files_retained=True)
            try:
                record['wall_timing'] = ledger.finish({'status': 'failed', 'expert_analysis_performed': False})
            except BaseException as timing_error:
                record['timing_error'] = f'{type(timing_error).__name__}: {timing_error}'
            save(destination / 'export_error.json', record)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True, help='Existing experiment root containing protocol.json and sealed zero-shot evidence')
    parser.add_argument('--task', required=True, help='One canonical protocol task name or task index')
    parser.add_argument('--remote-python', default=DEFAULT_REMOTE_PYTHON, help='Remote interpreter; must provide PyArrow')
    parser.add_argument('--ffmpeg', default='ffmpeg', help='Remote FFmpeg encoder executable')
    parser.add_argument('--local-ffmpeg', default='/usr/bin/ffmpeg')
    parser.add_argument('--local-ffprobe', default='/usr/bin/ffprobe')
    args = parser.parse_args(argv)
    result = export_one(args.output, args.task, remote_python=args.remote_python, ffmpeg=args.ffmpeg,
                        local_ffmpeg=args.local_ffmpeg, local_ffprobe=args.local_ffprobe)
    print(json.dumps({key: result[key] for key in ('status', 'source_count', 'task_name', 'destination')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
