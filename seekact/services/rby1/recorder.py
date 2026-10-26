"""Sliding-window recording of the RB-Y1 head camera and robot state into <session>/stream.

A background thread in the robot service: it only reads (the camera's latest frame and
the robot state through `robot_record`) and never commands anything. Layout and readers:
services/stream.py, auto_research_agent/api/stream.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

import cv2 as cv
import numpy as np

from services.stream import minute_name, prune


def _write_atomic(path, data):
    path = Path(path)
    tmp = path.with_name('.' + path.name)
    tmp.write_bytes(data)
    os.replace(tmp, path)


class Recorder:
    def __init__(self, cameras, robot_record, stream, *, fps=10.0, depth_hz=1.0, robot_hz=10.0,
                 window_s=600.0, jpeg_quality=80, depth=True, archive=None):
        """cameras: {name: object with read() -> (t_s, rgb, depth m | None) | None};
        robot_record(): JSON-safe dict of the robot state now (never commands).
        archive: a directory for the whole-session MP4 archive (services/archive.py), or None."""
        self.cameras, self.robot_record, self.stream = dict(cameras), robot_record, Path(stream)
        self.fps, self.depth_hz, self.robot_hz = float(fps), float(depth_hz), float(robot_hz)
        self.window_s, self.jpeg_quality, self.depth = float(window_s), int(jpeg_quality), depth
        self.stop_event = threading.Event()
        self.threads = []
        self.last_stamp = {}
        self.depth_at = {}
        self.counts = {name: [] for name in self.cameras}
        self.newest = {}
        self.errors = {}
        self.archive, self.archiver = (Path(archive) if archive else None), None

    # ----- lifecycle -------------------------------------------------------------------------
    def start(self):
        self.stream.mkdir(mode=0o755, parents=True, exist_ok=True)
        (self.stream/'robot').mkdir(exist_ok=True)
        for name in self.cameras:
            (self.stream/name).mkdir(exist_ok=True)
        if self.archive is not None:
            from services.archive import Archiver
            self.archiver = Archiver(self.stream, self.archive, list(self.cameras), fps=self.fps)
        for target, label in ((self._camera_loop, 'stream-cameras'), (self._robot_loop, 'stream-robot'),
                              (self._index_loop, 'stream-index')):
            thread = threading.Thread(target=target, name=label, daemon=True)
            thread.start()
            self.threads.append(thread)
        return self

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5)
        if self.archiver is not None:
            self.archiver.flush()  # the minute in progress too: keep the end of the session

    def _archived(self, folder, minute):
        if folder == 'robot':
            return (self.archive/'robot'/f'{minute}.jsonl').exists()
        return self.archiver.archived(folder, minute)

    # ----- cameras ---------------------------------------------------------------------------
    def capture_once(self, now=None):
        """One pass over the cameras: store any new RGB frame (and depth when due)."""
        now = time.time() if now is None else now
        for name, camera in self.cameras.items():
            try:
                latest = camera.read()
            except Exception as exc:
                self.errors[name] = str(exc)[:200]
                continue
            if latest is None:
                continue
            stamp, rgb, depth = latest
            if rgb is None or stamp == self.last_stamp.get(name):
                continue  # no new frame since the last pass
            self.last_stamp[name] = stamp
            t_ms = int(now * 1000)
            folder = self.stream/name/minute_name(now)
            folder.mkdir(exist_ok=True)
            ok, jpeg = cv.imencode('.jpg', cv.cvtColor(np.asarray(rgb), cv.COLOR_RGB2BGR),
                                   [cv.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                _write_atomic(folder/f'{t_ms}.jpg', jpeg.tobytes())
                self.newest[name] = t_ms
                self.counts[name].append(now)
                self.errors.pop(name, None)
            due = self.depth and self.depth_hz > 0 and now - self.depth_at.get(name, 0.0) >= 1.0 / self.depth_hz
            if depth is not None and due:
                mm = np.nan_to_num(np.asarray(depth, dtype=np.float32) * 1000.0, nan=0.0, posinf=0.0,
                                   neginf=0.0).clip(0, 65535).astype(np.uint16)
                ok, png = cv.imencode('.png', mm)
                if ok:
                    _write_atomic(folder/f'{t_ms}_depth.png', png.tobytes())
                    self.depth_at[name] = now

    def _camera_loop(self):
        period = 1.0 / self.fps
        while not self.stop_event.wait(period):
            self.capture_once()

    # ----- robot state -----------------------------------------------------------------------
    def record(self, now=None):
        now = time.time() if now is None else now
        try:
            return {'t_ms': int(now * 1000), **self.robot_record()}
        except Exception as exc:
            return {'t_ms': int(now * 1000), 'error': str(exc)[:200]}

    def _robot_loop(self):
        period = 1.0 / self.robot_hz
        while not self.stop_event.wait(period):
            now = time.time()
            with open(self.stream/'robot'/f'{minute_name(now)}.jsonl', 'a') as log:
                log.write(json.dumps(self.record(now)) + '\n')

    # ----- index and pruning -----------------------------------------------------------------
    def write_index(self, now=None):
        now = time.time() if now is None else now
        cameras = {}
        for name in self.cameras:
            recent = [t for t in self.counts[name] if now - t <= 5.0]
            self.counts[name] = recent
            cameras[name] = {'newest_ms': self.newest.get(name), 'fps': round(len(recent) / 5.0, 1),
                             **({'error': self.errors[name]} if name in self.errors else {})}
        index = {'now_ms': int(now * 1000), 'window_s': self.window_s, 'robot_hz': self.robot_hz,
                 'target_fps': self.fps, 'depth_hz': self.depth_hz if self.depth else 0, 'cameras': cameras}
        _write_atomic(self.stream/'index.json', json.dumps(index, indent=1).encode())

    def _index_loop(self):
        tick = 0
        while not self.stop_event.wait(1.0):
            self.write_index()
            tick += 1
            if tick % 10 == 0:
                prune(self.stream, self.window_s, archived=self._archived if self.archiver else None)
