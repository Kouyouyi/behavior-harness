"""Read-only RB-Y1 telemetry for the operator dashboard.

Its own gRPC connection to the robot's controller that only reads (`get_state`,
`get_control_manager_state`, `is_power_on`): never a command, never a power, servo or
parameter call, so it is safe next to a running session's robot service.
"""
from __future__ import annotations

import threading
import time

FRESH_S = 0.5
GROUPS = {'torso': (2, 8), 'right_arm': (8, 15), 'left_arm': (15, 22), 'head': (22, 24)}


class Telemetry:
    def __init__(self, address='192.168.30.1:50051', robot=None, kinematics=None):
        self.address = address
        self.robot = robot
        self._kinematics = kinematics
        self.lock = threading.Lock()

    def _connect(self):
        import rby1_sdk as rby
        robot = rby.create_robot_a(self.address)
        if not robot.connect(max_retries=1, timeout_ms=1000):
            raise RuntimeError(f'cannot reach the robot controller at {self.address}')
        return robot

    def snapshot(self):
        with self.lock:
            try:
                if self.robot is None:
                    self.robot = self._connect()
                state = self.robot.get_state()
                cms = self.robot.get_control_manager_state()
                power = bool(self.robot.is_power_on('.*'))
            except Exception as exc:
                self.robot = None
                return {'kind': 'rby1', 'ok': False, 'error': str(exc)[:300], 'address': self.address}
        ages = [j.time_since_last_update for j in state.joint_states]
        ages = [a.total_seconds() if hasattr(a, 'total_seconds') else float(a) for a in ages]
        position = [float(v) for v in state.position]
        joints = {name: [round(v, 4) for v in position[a:b]] for name, (a, b) in GROUPS.items()}
        stale = [name for name, (a, b) in GROUPS.items() if max(ages[a:b]) > FRESH_S]
        pressed = [k for k, e in enumerate(state.emo_states) if getattr(e.state, 'name', str(e.state)) == 'Pressed']
        battery = state.battery_state
        value = {'kind': 'rby1', 'address': self.address, 'time': time.time(), 'joints': joints, 'stale': stale,
                 'emo_pressed': pressed, 'power': power, 'manager': cms.state.name, 'control': cms.control_state.name,
                 'battery': {'level_percent': round(float(battery.level_percent), 1),
                             'voltage': round(float(battery.voltage), 2)},
                 'wrist_force_n': {side: [round(float(v), 1) for v in getattr(state, f'ft_sensor_{side}').force]
                                   for side in ('right', 'left')}}
        try:
            value['end_effectors'] = self._end_effectors(position)
        except Exception as exc:
            value['end_effectors'] = {'error': str(exc)[:200]}
        value['ok'] = not stale and not pressed and cms.state.name not in ('MinorFault', 'MajorFault')
        return value

    def _end_effectors(self, position):
        from .kinematics import EE_FRAME, OFFLINE_URDF, Kinematics
        if self._kinematics is None:
            self._kinematics = Kinematics(urdf_path=OFFLINE_URDF)
        frames = self._kinematics.frames(list(EE_FRAME.values()), position)
        return {side: [round(float(v), 3) for v in frames[frame][:3, 3]] for side, frame in EE_FRAME.items()}
