"""Whole-session MP4 archive of the RB-Y1 live stream (ffmpeg or imageio-ffmpeg required; no cameras)."""
import json
import time

import cv2 as cv
import numpy as np
import pytest

from services.archive import Archiver, encode_minute, ffmpeg_path, minute_of
from services.stream import prune

pytestmark = pytest.mark.skipif(ffmpeg_path() is None, reason='no ffmpeg')


def fill_minute(folder, start_ms, n=20, step_ms=100):
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        image = np.zeros((144, 256, 3), np.uint8)
        cv.putText(image, str(k), (20, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv.imwrite(str(folder/f'{start_ms + k * step_ms}.jpg'), image)
    cv.imwrite(str(folder/f'{start_ms}_depth.png'), np.zeros((144, 256), np.uint16))  # never archived


def test_encode_minute_keeps_every_frame_and_its_time(tmp_path):
    minute = tmp_path/'head_camera'/'20261009T1650'
    fill_minute(minute, 1_791_564_600_000)
    out = tmp_path/'archive'/'head_camera'/'20261009T1650.mp4'
    sidecar = encode_minute(minute, out, fps=10, codec='libx264')
    assert int(cv.VideoCapture(str(out)).get(cv.CAP_PROP_FRAME_COUNT)) == 20 == sidecar['frames']
    assert sidecar['frames_ms'][:2] == [1_791_564_600_000, 1_791_564_600_100]
    assert json.loads(out.with_suffix('.json').read_text()) == sidecar


def test_archiver_archives_completed_minutes_and_flushes_the_rest(tmp_path):
    now = time.time()
    done, current = minute_of(now - 120), minute_of(now)
    for name in (done, current):
        fill_minute(tmp_path/'stream'/'head_camera'/name, int(now * 1000), n=5)
    (tmp_path/'stream'/'robot').mkdir()
    (tmp_path/'stream'/'robot'/f'{done}.jsonl').write_text('{"t_ms": 1}\n')
    archiver = Archiver(tmp_path/'stream', tmp_path/'archive', ['head_camera'], fps=10, codec='libx264')
    archiver.running = False  # drive it by hand
    archiver.run_once(minute_of(now))
    assert archiver.archived('head_camera', done) and not archiver.archived('head_camera', current)
    assert (tmp_path/'archive'/'robot'/f'{done}.jsonl').exists()
    archiver.flush()
    assert archiver.archived('head_camera', current)


def test_prune_keeps_unarchived_minutes_until_the_last_resort(tmp_path):
    now = time.time()
    names = {age: minute_of(now - age) for age in (0, 240, 900)}
    for name in names.values():
        (tmp_path/'stream'/'head_camera'/name).mkdir(parents=True)
    archived = set()
    prune(tmp_path/'stream', 60, now, archived=lambda folder, minute: (folder, minute) in archived)
    # 240 s old: outside the 60 s window but not archived -> kept; 900 s: past the last resort -> deleted.
    assert {p.name for p in (tmp_path/'stream'/'head_camera').iterdir()} == {names[0], names[240]}
    archived.add(('head_camera', names[240]))
    prune(tmp_path/'stream', 60, now, archived=lambda folder, minute: (folder, minute) in archived)
    assert {p.name for p in (tmp_path/'stream'/'head_camera').iterdir()} == {names[0]}
