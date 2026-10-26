"""The RB-Y1 two-finger grippers: two Dynamixels on the UPC's local USB bus, via rby1_sdk.

`rby1_sdk.upc.GripperDeviceName` (/dev/rby1_gripper), protocol 2.0 at 2 Mbaud, ID 0 the
right gripper and ID 1 the left, powered by the arms' tool flanges at 12 V. This follows
the SDK's teleoperation example (examples/python/17_teleoperation_with_joint_mapping.py).

Nothing moves until `home()`: it powers the flanges, then sweeps both grippers to their
two ends under a small current to find the encoder range, which MOVES the fingers. After
that a background loop holds both at their targets in current-based position mode, where
the current limit is the grip force. Only one process can own the bus.
"""
from __future__ import annotations

import threading
import time

import numpy as np

IDS = {'right': 0, 'left': 1}
FULL_WIDTH_M = 0.1   # opening at the encoder's open end (calibrated on the characterized unit)
STILL_RAD = 0.01     # a finger counts as stopped below this change per tick (encoders jitter by 1 count, 0.0015 rad)
MIN_RANGE_RAD = 1.0  # a homing sweep that finds less travel than this was blocked


class Gripper:
    def __init__(self, robot, *, homing_current_a=0.5, hold_period_s=0.1, open_at_min=True):
        self.robot = robot
        self.homing_current_a = float(homing_current_a)
        self.hold_period_s = float(hold_period_s)
        self.open_at_min = bool(open_at_min)  # the characterized unit's grippers open towards the encoder minimum
        self.bus = None
        self.min_q = self.max_q = None
        self.target = {}           # side -> encoder target (rad)
        self.current = {side: 0.5 for side in IDS}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    @property
    def homed(self):
        return self.min_q is not None

    def _bus(self):
        import rby1_sdk as rby
        if self.bus is None:
            bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
            if not bus.open_port():
                raise RuntimeError(f'cannot open {rby.upc.GripperDeviceName}; is another process using it?')
            bus.set_baud_rate(2_000_000)
            bus.set_torque_constant([1, 1])
            self.bus = bus
        return self.bus

    def _mode(self, mode):
        bus = self.bus
        bus.group_sync_write_torque_enable([(i, 0) for i in IDS.values()])
        bus.group_sync_write_operating_mode([(i, mode) for i in IDS.values()])
        bus.group_sync_write_torque_enable([(i, 1) for i in IDS.values()])

    def home(self, guard=None, timeout_s=30.0):
        """Power the flanges, find both encoder ranges by a slow sweep (moves the fingers), then hold.

        guard() is called every tick and may raise to abort; the fingers then go limp.
        """
        import rby1_sdk as rby
        if self.homed:
            return
        for side in IDS:
            if not self.robot.set_tool_flange_output_voltage(side, 12):
                raise RuntimeError(f'could not set the {side} tool flange to 12 V')
        time.sleep(1.0)
        bus = self._bus()
        missing = [side for side, i in IDS.items() if not bus.ping(i)]
        if missing:
            raise RuntimeError(f'gripper Dynamixel not answering: {", ".join(missing)} (IDs {IDS}); check the '
                               'flange cable and power')
        self._mode(rby.DynamixelBus.CurrentControlMode)
        low, high = np.full(2, np.inf), np.full(2, -np.inf)
        q = prev = np.zeros(2)
        started = time.monotonic()
        try:
            for direction in (1.0, -1.0):
                still = 0
                while still < 15:
                    if guard is not None:
                        guard()
                    if time.monotonic() - started > timeout_s:
                        raise RuntimeError(f'gripper homing did not finish within {timeout_s:.0f} s')
                    bus.group_sync_write_send_torque([(i, direction * self.homing_current_a) for i in IDS.values()])
                    readings = bus.group_fast_sync_read_encoder(list(IDS.values()))
                    if readings is not None:
                        q = q.copy()
                        for i, value in readings:
                            q[i] = value
                    low, high = np.minimum(low, q), np.maximum(high, q)
                    still = still + 1 if np.allclose(q, prev, atol=STILL_RAD) else 0
                    prev = q
                    time.sleep(0.1)
            short = [side for side, i in IDS.items() if high[i] - low[i] < MIN_RANGE_RAD]
            if short:
                raise RuntimeError(f'gripper homing found too little travel for {", ".join(short)} '
                                   f'({np.round(high - low, 3).tolist()} rad); is something between the fingers?')
        except Exception:
            bus.group_sync_write_torque_enable([(i, 0) for i in IDS.values()])
            raise
        self.min_q, self.max_q = low, high
        self._mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        with self.lock:
            self.target = {side: self._encoder(side, self._measured_fraction(side, q)) for side in IDS}
        self.thread = threading.Thread(target=self._loop, name='gripper-hold', daemon=True)
        self.thread.start()

    def _fraction(self, side, encoder):
        i = IDS[side]
        span = self.max_q[i] - self.min_q[i]
        f = float(np.clip((encoder - self.min_q[i]) / span, 0.0, 1.0)) if span > 0 else 0.0
        return 1.0 - f if self.open_at_min else f  # 1 = open, 0 = closed

    def _measured_fraction(self, side, q):
        return self._fraction(side, q[IDS[side]])

    def _encoder(self, side, opening):
        i = IDS[side]
        f = 1.0 - opening if self.open_at_min else opening
        return float(self.min_q[i] + f * (self.max_q[i] - self.min_q[i]))

    def _loop(self):
        while not self.stop_event.wait(self.hold_period_s):
            with self.lock:
                targets = dict(self.target)
                currents = dict(self.current)
            try:
                self.bus.group_sync_write_send_torque([(IDS[s], currents[s]) for s in IDS])
                self.bus.group_sync_write_send_position([(IDS[s], q) for s, q in targets.items()])
            except Exception:
                pass  # the next tick retries; widths() reports a dead bus

    def command(self, side, width_m, current_a):
        """Set a target opening (m, 0 = closed) and current limit (A) for one gripper."""
        if not self.homed:
            raise RuntimeError('the grippers are not homed')
        opening = float(np.clip(width_m / FULL_WIDTH_M, 0.0, 1.0))
        with self.lock:
            self.target[side] = self._encoder(side, opening)
            self.current[side] = float(current_a)

    def hold(self):
        """Keep each gripper where it is now (target = measured)."""
        if not self.homed:
            return
        widths = self.widths()
        for side, width in widths.items():
            if width is not None:
                with self.lock:
                    self.target[side] = self._encoder(side, width / FULL_WIDTH_M)

    def widths(self):
        """{side: measured opening in m, or None}."""
        if not self.homed:
            return {side: None for side in IDS}
        readings = self.bus.group_fast_sync_read_encoder(list(IDS.values())) if self.bus else None
        if readings is None:
            return {side: None for side in IDS}
        q = dict(readings)
        return {side: round(self._fraction(side, q[i]) * FULL_WIDTH_M, 4) if i in q else None
                for side, i in IDS.items()}

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.bus is not None:
            try:
                self.bus.group_sync_write_torque_enable([(i, 0) for i in IDS.values()])
            except Exception:
                pass
            self.bus = None
