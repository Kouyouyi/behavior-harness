"""The RB-Y1 head camera (ZED 2i stereo, USB on the robot's UPC) through the ZED SDK (pyzed).

One grab thread keeps the newest left RGB image (and depth) with its capture time; the
backend's observations and the live recording both read that latest frame, so the camera
is opened once per robot service. Only one process can open a ZED: stop any other ZED
publisher on the UPC before a session.
"""
from __future__ import annotations

import threading
import time

import numpy as np

RESOLUTIONS = ('HD2K', 'HD1080', 'HD720', 'VGA')
DEPTH_MODES = ('NEURAL', 'NEURAL_LIGHT', 'NEURAL_PLUS', 'ULTRA', 'QUALITY', 'PERFORMANCE')
BAD_FRAMES_BEFORE_REOPEN = 15   # ~1 s of unusable frames
REOPEN_EVERY_S = 30.0


def unusable(rgb):
    """A frame that shows nothing: one flat colour, or the solid green (R = B = 0) the ZED SDK delivered
    after its own "successful" recovery from a USB connection failure (2026-10-09, push_chair session)."""
    sample = np.asarray(rgb)[::16, ::16].reshape(-1, 3).astype(np.float32)
    mean, spread = sample.mean(0), sample.std(0)
    # A dark scene (lens covered, lights off) is real: only a perfectly flat bright frame counts.
    return bool((mean[0] < 3.0 and mean[2] < 3.0 and mean[1] > 60.0) or (spread.max() < 1.0 and mean.max() > 30.0))


class ZedCamera:
    def __init__(self, serial, *, resolution='HD720', fps=15, depth=True, depth_mode='NEURAL',
                 open_timeout_s=30.0):
        import pyzed.sl as sl
        if resolution not in RESOLUTIONS:
            raise ValueError(f'unknown ZED resolution {resolution!r}; choose from {RESOLUTIONS}')
        if depth and depth_mode not in DEPTH_MODES:
            raise ValueError(f'unknown ZED depth mode {depth_mode!r}; choose from {DEPTH_MODES}')
        self.sl, self.depth = sl, bool(depth)
        self.camera = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init.camera_fps = int(fps)
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode) if self.depth else sl.DEPTH_MODE.NONE
        init.open_timeout_sec = float(open_timeout_s)
        if serial:
            init.set_from_serial_number(int(serial))
        self.init = init
        status = self.camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f'cannot open the head ZED (serial {serial}): {status}; is another process '
                               'using it?')
        info = self.camera.get_camera_information()
        left = info.camera_configuration.calibration_parameters.left_cam
        self.K = [[float(left.fx), 0.0, float(left.cx)], [0.0, float(left.fy), float(left.cy)], [0.0, 0.0, 1.0]]
        self.serial = int(info.serial_number)
        self.lock = threading.Lock()
        self.latest = None   # (capture time s, rgb uint8 HxWx3, depth float32 m or None)
        self.error = None
        self.bad_frames, self.reopens, self.reopened_at = 0, 0, 0.0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name='zed-grab', daemon=True)
        self.thread.start()

    def _loop(self):
        sl = self.sl
        image, depth = sl.Mat(), sl.Mat()
        runtime = sl.RuntimeParameters()
        while not self.stop_event.is_set():
            status = self.camera.grab(runtime)
            if status != sl.ERROR_CODE.SUCCESS:
                self.error = str(status)
                time.sleep(0.05)
                continue
            self.camera.retrieve_image(image, sl.VIEW.LEFT)
            rgb = np.ascontiguousarray(image.get_data()[:, :, 2::-1])  # BGRA -> RGB
            if unusable(rgb):
                self._bad_frame()
                continue
            self.bad_frames = 0
            d = None
            if self.depth:
                self.camera.retrieve_measure(depth, sl.MEASURE.DEPTH)
                d = np.array(depth.get_data(), dtype=np.float32, copy=True)
            with self.lock:
                self.latest = (time.time(), rgb, d)
            self.error = None

    def _bad_frame(self):
        """Never serve an empty frame; reopen the camera when they persist."""
        self.bad_frames += 1
        self.error = (f'the ZED delivers solid-colour frames ({self.bad_frames} in a row); '
                      f'reopened {self.reopens} time(s)')
        if self.bad_frames < BAD_FRAMES_BEFORE_REOPEN or time.monotonic() - self.reopened_at < REOPEN_EVERY_S:
            return
        self.reopened_at, self.reopens = time.monotonic(), self.reopens + 1
        self.camera.close()
        status = self.camera.open(self.init)
        if status != self.sl.ERROR_CODE.SUCCESS:
            self.error = f'the ZED delivered solid-colour frames and reopening it failed: {status}'
            time.sleep(1.0)

    def read(self):
        """(capture time s, rgb, depth m or None) of the newest frame, or None before the first."""
        with self.lock:
            return self.latest

    def intrinsics(self):
        return self.K

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.camera.close()
