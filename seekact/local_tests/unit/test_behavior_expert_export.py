"""CPU/mock export contracts. No SSH, expert data, real video decode, or Isaac."""
import contextlib
import fcntl
import hashlib
import io
import json
from pathlib import Path, PosixPath
import shlex
import subprocess
import tarfile
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from scripts import export_one_behavior_expert as export
from scripts import run_behavior_experiment as run


def meta(content):
    return {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


def archive(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as stream:
        for name, content, kind in entries:
            item = tarfile.TarInfo(name)
            item.type = kind
            if kind == tarfile.REGTYPE:
                item.size = len(content)
                stream.addfile(item, io.BytesIO(content))
            else:
                item.linkname = '/outside-the-export'
                stream.addfile(item)
    buffer.seek(0)
    return buffer


class ExpertExportTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix='behavior-export-test-')))
        self.row = {'task_index': 0, 'task_name': 'turning_on_radio', 'prompt': 'Synthetic test prompt',
                    'fixed_expert': {'episode_index': 0, 'raw_episode_id': 10, 'frames': 1956}}
        self.protocol = {'render_profile': 'fast', 'variant': 'original', 'split': 'public_test',
                         'instance_id': 301, 'seed': 20261020, 'tasks': [self.row]}
        self.path = self.root / 'protocol.json'
        self.path.write_text(json.dumps(self.protocol))

    def seal(self, success=False):
        trial = self.root / self.row['task_name'] / 'zero_shot_v1'
        (trial / 'agent').mkdir(parents=True)
        (trial / 'agent/agent_timing.json').write_text(json.dumps({'status': 'agent_completed', 'exit_code': 0}))
        closed = {'status': 'closed', 'steps': 0, 'submitted': 0, 'acknowledged': 0,
                  'execution_integrity': True, 'official_success': success, 'uncertain_action': None,
                  'errors': [], 'render_profile': 'fast',
                  'videos': {'passed': True, 'status': 'no_frames', 'expected_frames': 0}}
        (trial / 'episode_closed.json').write_text(json.dumps(closed))
        (trial / 'trial_record.json').write_text(json.dumps({
            'task': self.row['task_name'], 'phase': 'zero_shot',
            'status': 'task_success' if success else 'task_failure',
            'agent_started': True, 'agent_exit_code': 0, 'worker_exit_code': 0,
            'started_at_utc': '2026-10-23T00:00:00Z', 'protocol_sha256': run.sha256(self.path)}))
        return trial

    def plan(self):
        return export.make_plan(self.row, self.path, {'record_sha256': 'a' * 64}, 'ffmpeg')

    def response(self, plan):
        files = {name: b'Generated fixture only, not an expert recording.\n' for name in export.REMOTE_FILES}
        manifest = {key: plan[key] for key in ('task_name', 'task_index', 'episode_index', 'raw_episode_id',
                                              'frames', 'source_start_frame', 'source_count', 'remote_directory',
                                              'protocol_sha256', 'zero_shot_record_sha256')}
        manifest.update(schema_version=1, source_fps=30, source_analysis_performed=False,
                        automatic_fallback=False, phases={'encode_head_seconds': 1000},
                        files={name: meta(content) for name, content in files.items()})
        encoded = json.dumps(manifest).encode()
        files['remote_manifest.json'] = encoded
        return {'manifest': manifest, 'manifest_file': meta(encoded)}, files

    def test_gate_blocks_remote_reads_before_valid_failure(self):
        with patch.object(export, 'execute_remote_export') as remote, patch.object(export, 'transfer_export') as transfer:
            with self.assertRaisesRegex(ValueError, 'existing sealed'):
                export.export_one(self.root, 'turning_on_radio')
            self.seal(success=True)
            with self.assertRaisesRegex(ValueError, 'latest valid'):
                export.export_one(self.root, 'turning_on_radio')
            remote.assert_not_called()
            transfer.assert_not_called()
        self.assertFalse((self.root / 'turning_on_radio/expert_episode_0').exists())

    def test_changed_protocol_or_unsealed_attempt_blocks_export(self):
        self.seal()
        self.path.write_text(json.dumps({**self.protocol, 'seed': 2}))
        with patch.object(export, 'execute_remote_export') as remote:
            with self.assertRaisesRegex(ValueError, 'protocol differs'):
                export.export_one(self.root, '0')
            self.path.write_text(json.dumps(self.protocol))
            (self.root / 'turning_on_radio/zero_shot_v2').mkdir()
            with self.assertRaisesRegex(ValueError, 'Unsealed'):
                export.export_one(self.root, '0')
            remote.assert_not_called()

    def test_other_task_runner_and_unsealed_episode_do_not_block_selected_export(self):
        self.seal()
        (self.root / 'putting_away_toys/zero_shot_v1').mkdir(parents=True)
        with (self.root / 'runner.lock').open('a') as batch_lock:
            fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(export, 'execute_remote_export', side_effect=RuntimeError('mock remote admission')) as remote:
                with self.assertRaisesRegex(RuntimeError, 'mock remote admission'):
                    export.export_one(self.root, '0')
                remote.assert_called_once()
        destination = self.root / 'turning_on_radio/expert_episode_0'
        self.assertTrue((destination / 'export_error.json').exists())
        record = json.loads((destination / 'export_error.json').read_text())
        self.assertEqual(record['lock_scope'], 'selected_task_expert_export')

    def test_same_task_export_lock_blocks_duplicate_before_remote_read(self):
        self.seal()
        with (self.root / 'turning_on_radio/expert_export.lock').open('a') as task_lock:
            fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(export, 'execute_remote_export') as remote:
                with self.assertRaises(BlockingIOError):
                    export.export_one(self.root, '0')
                remote.assert_not_called()
        self.assertFalse((self.root / 'turning_on_radio/expert_episode_0').exists())

    def test_new_selected_task_unsealed_attempt_is_rechecked_before_remote_read(self):
        self.seal()
        make_plan = export.make_plan
        def create_unsealed(*args, **kwargs):
            plan = make_plan(*args, **kwargs)
            (self.root / 'turning_on_radio/zero_shot_newer').mkdir()
            return plan
        with patch.object(export, 'make_plan', side_effect=create_unsealed), \
                patch.object(export, 'execute_remote_export') as remote:
            with self.assertRaisesRegex(ValueError, 'Unsealed'):
                export.export_one(self.root, '0')
            remote.assert_not_called()

    def test_new_selected_task_unsealed_attempt_during_export_blocks_transfer(self):
        self.seal()
        def remote(plan, *_):
            (self.root / 'turning_on_radio/zero_shot_newer').mkdir()
            return self.response(plan)[0]
        with patch.object(export, 'execute_remote_export', side_effect=remote), \
                patch.object(export, 'transfer_export') as transfer:
            with self.assertRaisesRegex(ValueError, 'Unsealed'):
                export.export_one(self.root, '0')
            transfer.assert_not_called()
        self.assertFalse((self.root / 'turning_on_radio/expert_episode_0/export_complete.json').exists())

    def test_fixed_metadata_cannot_select_another_expert(self):
        expected = {'turning_on_radio': (0, 0, 10, 1956),
                    'putting_shoes_on_rack': (22, 4400, 220020, 7076),
                    'putting_away_toys': (54, 10800, 540010, 13136)}
        for name, values in expected.items():
            self.assertEqual(tuple(export._fixed_task(name).values()), values)
        self.row['fixed_expert']['episode_index'] = 1
        self.path.write_text(json.dumps(self.protocol))
        self.seal()
        with patch.object(export, 'execute_remote_export') as remote:
            with self.assertRaisesRegex(ValueError, 'no fallback'):
                export.export_one(self.root, '0')
            remote.assert_not_called()
        with self.assertRaises(ValueError):
            export._fixed_task('../another_task')

    def test_complete_export_has_one_source_and_real_local_walltime(self):
        self.seal()
        captured = {}
        def remote(plan, destination, interpreter):
            self.assertEqual(interpreter, export.DEFAULT_REMOTE_PYTHON)
            captured['envelope'], captured['files'] = self.response(plan)
            return captured['envelope']
        def transfer(plan, destination, expected):
            self.assertEqual(set(expected), export.REMOTE_FILES | {'remote_manifest.json'})
            for name, content in captured['files'].items():
                path = destination / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(content)
            return {'sha256_verified': True, 'files': len(expected)}
        with patch.object(export, 'execute_remote_export', side_effect=remote), \
                patch.object(export, 'transfer_export', side_effect=transfer), \
                patch.object(export, 'verify_exported_videos', return_value={'passed': True}) as verifier:
            result = export.export_one(self.root, '0')
        self.assertEqual(result['status'], 'export_verified')
        self.assertEqual(result['source_count'], 1)
        self.assertFalse(result['source_analysis_performed'])
        self.assertEqual(result['fixed_expert']['episode_index'], 0)
        verifier.assert_called_once()
        self.assertEqual(verifier.call_args.args[1], 1956)
        self.assertEqual(set(result['wall_timing']['by_name']), {'expert_export', 'expert_transfer', 'expert_verification'})
        self.assertTrue(result['wall_timing']['wall_clock_not_simulation_time'])
        self.assertTrue(result['wall_timing']['accounting']['exclusive_plus_idle_equals_total'])
        self.assertFalse(result['wall_timing']['accounting']['background_diagnostics_included_in_totals'])
        self.assertLess(result['wall_timing']['total_seconds'], 1000)
        self.assertTrue((Path(result['destination']) / 'export_complete.json').exists())

    def test_failed_transfer_preserves_partial_evidence_without_fallback(self):
        self.seal()
        def transfer(plan, destination, expected):
            (destination / 'partial-transfer.fixture').write_text('generated partial fixture')
            raise OSError('injected transfer failure')
        with patch.object(export, 'execute_remote_export', side_effect=lambda plan, *_: self.response(plan)[0]) as remote, \
                patch.object(export, 'transfer_export', side_effect=transfer) as transport, \
                patch.object(export, 'verify_exported_videos') as verifier:
            with self.assertRaisesRegex(OSError, 'transfer failure'):
                export.export_one(self.root, '0')
            remote.assert_called_once()
            transport.assert_called_once()
            verifier.assert_not_called()
        destination = self.root / 'turning_on_radio/expert_episode_0'
        error = json.loads((destination / 'export_error.json').read_text())
        self.assertTrue(error['partial_local_and_remote_files_retained'])
        self.assertEqual(error['source_count'], 1)
        self.assertFalse((destination / 'export_complete.json').exists())
        self.assertTrue((destination / 'partial-transfer.fixture').is_file())
        with patch.object(export, 'execute_remote_export') as remote:
            with self.assertRaises(FileExistsError):
                export.export_one(self.root, '0')
            remote.assert_not_called()

    def test_manifest_rejects_other_sources_or_extra_files(self):
        plan = self.plan()
        for key, bad in [('episode_index', 1), ('source_count', 2), ('frames', 1957),
                         ('source_start_frame', 1), ('protocol_sha256', 'b' * 64)]:
            envelope, _ = self.response(plan)
            envelope['manifest'][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                export.validated_files(envelope, plan)
        envelope, _ = self.response(plan)
        envelope['manifest']['files']['other_episode.mp4'] = meta(b'extra')
        with self.assertRaisesRegex(ValueError, 'only the one fixed expert'):
            export.validated_files(envelope, plan)

    def test_remote_shell_quoting_preserves_literal_arguments(self):
        argv = ['/a path/python; echo NO', '-c', 'print("literal $(whoami)")\n', 'value with `backticks`']
        command = export.remote_command(argv)
        self.assertEqual(command[:-1], ['ssh', '-o', 'BatchMode=yes', 'huoshan'])
        self.assertEqual(shlex.split(command[-1]), argv)

    def test_remote_program_is_self_contained_and_only_built_not_run(self):
        plan = self.plan()
        envelope, _ = self.response(plan)
        with patch.object(export.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps(envelope))) as run_command:
            self.assertEqual(export.execute_remote_export(plan, self.root, '/remote python'), envelope)
        command = run_command.call_args.args[0]
        parsed = shlex.split(command[-1])
        self.assertEqual(parsed[:2], ['/remote python', '-c'])
        compile(parsed[2], '<remote-export-not-executed>', 'exec')
        self.assertEqual(json.loads(parsed[3]), plan)
        self.assertNotIn('shell', run_command.call_args.kwargs)

    def test_encoder_trims_exact_frames_reencodes_without_chunk_copy(self):
        command = export._encode_command('/ffmpeg path', '/source.mp4', '/output.mp4', 1956)
        self.assertEqual(command[command.index('-vf') + 1], 'trim=start_frame=0:end_frame=1956,setpts=N/(30*TB)')
        self.assertEqual(command[command.index('-frames:v') + 1], '1956')
        self.assertEqual(command[command.index('-c:v') + 1], 'libx264')
        self.assertEqual(command[command.index('-profile:v') + 1], 'high')
        self.assertEqual(command[command.index('-pix_fmt') + 1], 'yuv420p')
        self.assertIn('-n', command)
        self.assertNotIn('copy', command)

    def remote_fixture(self, rows=1956):
        """Redirect every remote path into a generated fixture; mock Arrow/video I/O."""
        generated = self.root / 'generated_source'
        annotation = generated / '_source/annotations/task-0000/episode_00000010.json'
        annotation.parent.mkdir(parents=True)
        annotation.write_text('{"synthetic_fixture": true}')
        table = Mock(num_rows=rows, column_names=['episode_index', 'frame_index'])
        table.column.side_effect = lambda name: Mock(to_pylist=lambda:
            [0] * rows if name == 'episode_index' else list(range(rows)))
        table.sort_by.return_value = table
        arrow = ModuleType('pyarrow'); pq = ModuleType('pyarrow.parquet'); arrow.parquet = pq
        pq.read_table = Mock(return_value=table)
        pq.write_table = Mock(side_effect=lambda data, path, **kwargs: path.write_bytes(b'synthetic filtered parquet fixture'))
        remote_export_base = self.root / 'remote_exports'
        prefixes = [(export.REMOTE_BASE, remote_export_base),
                    ('/cs/datasets/behavior2026_dual_language', generated)]
        def mapped_path(value):
            text = str(value)
            for prefix, replacement in prefixes:
                if text == prefix or text.startswith(prefix + '/'):
                    return replacement / text[len(prefix):].lstrip('/')
            raise AssertionError(f'Unexpected remote source path: {text}')
        def encoder(command, **kwargs):
            self.assertEqual(command[command.index('-frames:v') + 1], '1956')
            PosixPath(command[-1]).write_bytes(b'synthetic encoded video fixture')
            return subprocess.CompletedProcess(command, 0)
        self.stack.enter_context(patch.dict('sys.modules', {'pyarrow': arrow, 'pyarrow.parquet': pq}))
        self.stack.enter_context(patch('pathlib.Path', side_effect=mapped_path))
        encoding = self.stack.enter_context(patch.object(export.subprocess, 'run', side_effect=encoder))
        return pq, table, encoding, remote_export_base

    def test_remote_body_filters_only_selected_episode_with_mock_arrow_and_encoder(self):
        plan = self.plan()
        pq, table, encoding, base = self.remote_fixture()
        result = export._remote_export(plan)
        pq.read_table.assert_called_once()
        self.assertEqual(pq.read_table.call_args.kwargs['filters'], [('episode_index', '=', 0)])
        table.sort_by.assert_called_once_with([('frame_index', 'ascending')])
        pq.write_table.assert_called_once()
        self.assertEqual(encoding.call_count, 3)
        self.assertEqual(result['manifest']['source_count'], 1)
        self.assertEqual(result['manifest']['frames'], 1956)
        self.assertEqual(set(result['manifest']['files']), export.REMOTE_FILES)
        destination = base / plan['remote_directory'].rsplit('/', 1)[-1]
        self.assertEqual((destination / 'annotation.json').read_text(), '{"synthetic_fixture": true}')
        with self.assertRaises(FileExistsError):
            export._remote_export(plan)
        self.assertEqual(pq.read_table.call_count, 1)

    def test_remote_body_wrong_fixed_frame_count_keeps_error_without_fallback(self):
        plan = self.plan()
        pq, table, encoding, base = self.remote_fixture(rows=1955)
        with self.assertRaisesRegex(ValueError, 'row count mismatch'):
            export._remote_export(plan)
        pq.read_table.assert_called_once()
        pq.write_table.assert_not_called()
        encoding.assert_not_called()
        destination = base / plan['remote_directory'].rsplit('/', 1)[-1]
        error = json.loads((destination / 'remote_export_error.json').read_text())
        self.assertEqual(error['status'], 'failed')
        self.assertTrue(error['partial_files_retained'])
        self.assertFalse(error['automatic_fallback'])
        self.assertFalse((destination / 'remote_manifest.json').exists())


class SafeTransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='expert-transfer-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_regular_declared_files_hash_and_size_are_verified(self):
        expected = {'episode.parquet': meta(b'fixture'), 'videos/head.mp4': meta(b'fixture video')}
        data = archive([('episode.parquet', b'fixture', tarfile.REGTYPE),
                        ('videos/head.mp4', b'fixture video', tarfile.REGTYPE)])
        result = export.extract_checked(data, self.root, expected)
        self.assertTrue(result['sha256_verified'])
        self.assertEqual(result['files'], 2)
        self.assertEqual((self.root / 'videos/head.mp4').read_bytes(), b'fixture video')

    def test_traversal_links_extra_files_and_duplicates_are_refused(self):
        cases = [('../outside', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                 ('videos/head.mp4', tarfile.SYMTYPE), ('videos/head.mp4', tarfile.LNKTYPE),
                 ('unexpected.mp4', tarfile.REGTYPE)]
        for index, (name, kind) in enumerate(cases):
            destination = self.root / str(index); destination.mkdir()
            with self.subTest(name=name, kind=kind), self.assertRaises(ValueError):
                export.extract_checked(archive([(name, b'one', kind)]), destination,
                                       {'videos/head.mp4': meta(b'one')})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            export.extract_checked(archive([('one', b'1', tarfile.REGTYPE), ('one', b'1', tarfile.REGTYPE)]),
                                   self.root, {'one': meta(b'1')})

    def test_existing_file_parent_symlink_incomplete_and_bad_hash_are_refused(self):
        (self.root / 'one').write_text('original')
        with self.assertRaises(FileExistsError):
            export.extract_checked(archive([('one', b'1', tarfile.REGTYPE)]), self.root, {'one': meta(b'1')})
        self.assertEqual((self.root / 'one').read_text(), 'original')
        (self.root / 'videos').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            export.extract_checked(archive([('videos/escape', b'1', tarfile.REGTYPE)]),
                                   self.root, {'videos/escape': meta(b'1')})
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            export.extract_checked(archive([]), self.root, {'absent': meta(b'1')})
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            export.extract_checked(archive([('bad', b'2', tarfile.REGTYPE)]), self.root, {'bad': meta(b'1')})
        self.assertTrue((self.root / 'bad').exists())  # Preserve failure evidence.


class VideoMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='expert-video-mock-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'videos').mkdir()
        for role in export.ROLES:
            (self.root / 'videos' / f'{role}.mp4').write_bytes(b'Synthetic placeholder, never decoded')
        self.stream = {'codec_name': 'h264', 'profile': 'High', 'codec_tag_string': 'avc1',
                       'pix_fmt': 'yuv420p', 'nb_read_frames': '1956', 'avg_frame_rate': '30/1'}

    def test_three_views_are_probed_and_fully_decoded(self):
        def mock_run(command, **kwargs):
            text = json.dumps({'streams': [self.stream]}) if command[0] == '/probe' else ''
            return subprocess.CompletedProcess(command, 0, text, '')
        with patch.object(export.subprocess, 'run', side_effect=mock_run) as run_command:
            result = export.verify_exported_videos(self.root, 1956, ffmpeg='/decode', ffprobe='/probe')
        self.assertTrue(result['passed'])
        self.assertEqual(result['source_count'], 1)
        self.assertEqual(set(result['videos']), set(export.ROLES))
        self.assertEqual(run_command.call_count, 6)
        for call in run_command.call_args_list[1::2]:
            self.assertIn('-xerror', call.args[0])
            self.assertEqual(call.args[0][-3:], ['-f', 'null', '-'])

    def test_wrong_frame_count_or_error_stderr_cannot_pass(self):
        for key, value in [('nb_read_frames', '1955'), ('avg_frame_rate', '29/1'), ('pix_fmt', 'yuv444p')]:
            stream = dict(self.stream, **{key: value})
            with self.subTest(key=key), patch.object(export.subprocess, 'run', return_value=
                    subprocess.CompletedProcess([], 0, json.dumps({'streams': [stream]}), '')):
                with self.assertRaises(ValueError):
                    export.verify_exported_videos(self.root, 1956)
        sequence = [subprocess.CompletedProcess([], 0, json.dumps({'streams': [self.stream]}), ''),
                    subprocess.CompletedProcess([], 0, '', 'injected decode error')]
        with patch.object(export.subprocess, 'run', side_effect=sequence):
            with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                export.verify_exported_videos(self.root, 1956)


if __name__ == '__main__':
    unittest.main()
