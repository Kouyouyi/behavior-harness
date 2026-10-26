import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from services.behavior_ap.video import VideoVerificationError, verify_videos


class BehaviorVideoTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.metadata = {}
        for role in ('robot_cameras', 'third_person'):
            path = self.path / (role + '.mp4')
            path.write_bytes(b'placeholder for mocked subprocess')
            self.metadata[role] = {'path': str(path), 'frames': 3, 'finalized': True,
                                   'frames_submitted': 3, 'frames_encoded': 3,
                                   'dropped_frames': 0, 'queue_pending_frames': 0,
                                   'frames_unencoded': 0, 'encoder_error': None}
        self.result = {'steps': 3, 'videos': self.metadata}

    @staticmethod
    def successful_run(command, **kwargs):
        if 'ffprobe' in str(command[0]):
            stream = {'codec_name': 'h264', 'profile': 'High', 'codec_tag_string': 'avc1',
                      'pix_fmt': 'yuv420p', 'nb_read_frames': '3', 'width': 16, 'height': 16,
                      'avg_frame_rate': '30/1'}
            return subprocess.CompletedProcess(command, 0, json.dumps({'streams': [stream]}), '')
        return subprocess.CompletedProcess(command, 0, '', '')

    def test_both_roles_probe_and_fully_decode(self):
        with patch('services.behavior_ap.video.subprocess.run', side_effect=self.successful_run) as run:
            evidence = verify_videos(self.result, 3)
        assert run.call_count == 4
        assert evidence['passed'] and evidence['playable_video_available']
        assert set(evidence['videos']) == {'robot_cameras', 'third_person'}
        assert all(row['full_decode_passed'] for row in evidence['videos'].values())
        assert all(row['actual_frames'] == 3 for row in evidence['videos'].values())
        json.dumps(evidence, allow_nan=False)

    def test_zero_frames_does_not_fabricate_a_video(self):
        with patch('services.behavior_ap.video.subprocess.run') as run:
            evidence = verify_videos({'steps': 0, 'videos': {}}, 0)
        run.assert_not_called()
        assert evidence['passed'] and evidence['status'] == 'no_frames'
        assert evidence['playable_video_available'] is False
        assert evidence['videos'] == {}

    def test_positive_frames_require_both_roles(self):
        with self.assertRaises(VideoVerificationError) as caught:
            verify_videos({'steps': 3, 'videos': {'robot_cameras': self.metadata['robot_cameras']}}, 3)
        assert caught.exception.evidence['missing_roles'] == ['third_person']

    def test_encoder_drain_error_is_not_overlooked(self):
        self.metadata['robot_cameras']['frames_encoded'] = 2
        with patch('services.behavior_ap.video.subprocess.run') as run:
            with self.assertRaises(VideoVerificationError) as caught:
                verify_videos(self.result, 3)
        run.assert_not_called()
        assert caught.exception.evidence['failed_stage'] == 'metadata'

    def test_actual_frame_count_must_match(self):
        def short_file(command, **kwargs):
            result = self.successful_run(command, **kwargs)
            if 'ffprobe' in str(command[0]):
                data = json.loads(result.stdout)
                data['streams'][0]['nb_read_frames'] = '2'
                result.stdout = json.dumps(data)
            return result
        with patch('services.behavior_ap.video.subprocess.run', side_effect=short_file):
            with self.assertRaises(VideoVerificationError) as caught:
                verify_videos(self.result, 3)
        assert caught.exception.evidence['failed_stage'] == 'probe_validation'
        assert caught.exception.evidence['probe_stream']['nb_read_frames'] == '2'

    def test_codec_must_be_browser_compatible_h264(self):
        def wrong_codec(command, **kwargs):
            result = self.successful_run(command, **kwargs)
            data = json.loads(result.stdout)
            data['streams'][0]['codec_name'] = 'mpeg4'
            result.stdout = json.dumps(data)
            return result
        with patch('services.behavior_ap.video.subprocess.run', side_effect=wrong_codec):
            with self.assertRaises(VideoVerificationError) as caught:
                verify_videos(self.result, 3)
        assert 'codec_name' in caught.exception.evidence['codec_mismatches']

    def test_decode_error_retains_stderr_even_when_exit_zero(self):
        def corrupt_file(command, **kwargs):
            if 'ffmpeg' in str(command[0]):
                return subprocess.CompletedProcess(command, 0, '', 'corrupt decoded frame')
            return self.successful_run(command, **kwargs)
        with patch('services.behavior_ap.video.subprocess.run', side_effect=corrupt_file):
            with self.assertRaises(VideoVerificationError) as caught:
                verify_videos(self.result, 3)
        assert caught.exception.evidence['failed_stage'] == 'full_decode'
        assert caught.exception.evidence['stderr'] == 'corrupt decoded frame'

    def test_timeout_retains_diagnostic(self):
        with patch('services.behavior_ap.video.subprocess.run',
                   side_effect=subprocess.TimeoutExpired('ffprobe', 1, stderr=b'partial decode error')):
            with self.assertRaises(VideoVerificationError) as caught:
                verify_videos(self.result, 3, timeout_seconds=1)
        assert caught.exception.evidence['stderr'] == 'partial decode error'
        assert caught.exception.evidence['failed_stage'] == 'ffprobe'

    def test_bad_probe_json_and_missing_video_fail(self):
        with patch('services.behavior_ap.video.subprocess.run',
                   return_value=subprocess.CompletedProcess('ffprobe', 0, 'not json', '')):
            with self.assertRaises(VideoVerificationError) as caught:
                verify_videos(self.result, 3)
        assert caught.exception.evidence['failed_stage'] == 'ffprobe_json'
        Path(self.metadata['robot_cameras']['path']).unlink()
        with self.assertRaises(VideoVerificationError) as caught:
            verify_videos(self.result, 3)
        assert caught.exception.evidence['failed_stage'] == 'file'

    def test_steps_and_zero_frame_metadata_must_be_consistent(self):
        with self.assertRaises(VideoVerificationError):
            verify_videos(self.result, 2)
        with self.assertRaises(VideoVerificationError):
            verify_videos({'steps': 0, 'videos': self.metadata}, 0)
        for bad in (-1, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                verify_videos(self.result, bad)

    @unittest.skipUnless(Path('/usr/bin/ffmpeg').is_file() and Path('/usr/bin/ffprobe').is_file(),
                         'CPU ffmpeg tools are unavailable')
    def test_real_tiny_generated_video_fully_decodes(self):
        source = self.path / 'generated.mp4'
        subprocess.run(['/usr/bin/ffmpeg', '-nostdin', '-v', 'error', '-y',
                        '-f', 'lavfi', '-i', 'color=c=blue:s=32x32:r=30',
                        '-frames:v', '3', '-an', '-c:v', 'libx264', '-threads', '1',
                        '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-tag:v', 'avc1',
                        '-movflags', '+faststart', str(source)], check=True, timeout=30)
        for metadata in self.metadata.values():
            shutil.copyfile(source, metadata['path'])
        evidence = verify_videos(self.result, 3)
        assert evidence['status'] == 'verified'
        assert evidence['videos']['robot_cameras']['actual_frames'] == 3
        assert evidence['videos']['third_person']['full_decode_passed']


if __name__ == '__main__':
    unittest.main()
