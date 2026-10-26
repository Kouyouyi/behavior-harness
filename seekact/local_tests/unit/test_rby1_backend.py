"""RB-Y1 backend logic against a simulated robot (the SdkRobot surface): nothing here touches hardware."""
import json
import threading
import time

import cv2 as cv
import numpy as np
import pytest

pytest.importorskip('rby1_sdk')

from services import arming  # noqa: E402
from services.controller.config import Configuration  # noqa: E402
from services.rby1.backend import ARM_REACH_M, PRESETS, MotionStopped, Rby1Backend  # noqa: E402
from services.rby1.kinematics import (COMPONENTS, EE_FRAME, HEAD_CAMERA, OFFLINE_URDF, Kinematics,  # noqa: E402
                                      component, with_component)
from services.rby1.controller_modes import CONTROLLER_TOOLS  # noqa: E402
from services.rby1.kinematics import JOINT_NAMES as JOINTS, component as component_of  # noqa: E402
from services.rby1.robot import Snapshot  # noqa: E402

READY = np.zeros(24)
for _name in ('torso', 'right_arm', 'left_arm'):
    READY = with_component(READY, _name, PRESETS[_name]['ready'])


class FakeHandle:
    """A joint command on the robot's controller: the joints move a step towards the goal per poll."""

    def __init__(self, robot, goals, steps):
        self.robot, self.goals, self.steps = robot, goals, steps
        self.start = robot.q.copy()
        self.polls, self.cancelled, self.code = 0, False, 'Ok'

    def is_done(self):
        if self.cancelled or self.polls >= self.steps:
            return True
        if self.robot.frozen:
            return False
        self.polls += 1
        fraction = self.polls / self.steps
        for name, goal in self.goals.items():
            start = component(self.start, name)
            self.robot.q = with_component(self.robot.q, name, start + fraction * (np.asarray(goal) - start))
        if self.robot.on_poll:
            self.robot.on_poll(self)
        return self.polls >= self.steps

    def finish(self):
        return 'Canceled' if self.cancelled else self.code

    def wait(self, timeout_ms):
        while not self.is_done():
            pass
        return True

    def cancel(self):
        self.cancelled = True


class FakeBaseStream:
    def __init__(self, robot):
        self.robot, self.sends, self.closed = robot, [], False

    def send(self, vx, wz, minimum_time):
        self.sends.append((vx, wz))
        if self.robot.base_stuck:
            return
        x, y, yaw = self.robot.odometry
        dt = minimum_time
        self.robot.odometry = (x + vx * np.cos(yaw) * dt, y + vx * np.sin(yaw) * dt, yaw + wz * dt)

    def close(self):
        self.closed = True


class FakeRobot:
    """The SdkRobot surface the backend uses, with an in-memory robot."""

    def __init__(self, q=READY):
        self.q = np.array(q, dtype=np.float64)
        self.age = np.full(24, 0.01)
        self.emo = False
        self.force = {'right': [0.0, 0.0, 0.0], 'left': [0.0, 0.0, 0.0]}
        self.odometry = (0.0, 0.0, 0.0)
        self.enables, self.commands, self.holds, self.streams, self.flanges = [], [], [], [], []
        self.steps, self.frozen, self.on_poll, self.base_stuck, self.code = 5, False, None, False, 'Ok'
        self.reads = 0
        self.torque = np.zeros(24)
        self.temperature = np.full(24, 30.0)
        self.ready = np.ones(24, bool)
        self.manager = 'Enabled'
        self.resets = 0
        self.controller_commands = []

    def urdf(self):
        return OFFLINE_URDF.read_text()

    def state(self):
        self.reads += 1
        return Snapshot(t=time.time(), q=self.q.copy(), ready=self.ready.copy(), age_s=self.age.copy(),
                        emo_pressed=self.emo,
                        ft={s: {'force': list(f), 'torque': [0.0] * 3, 'age_s': 0.0} for s, f in self.force.items()},
                        odometry=self.odometry, battery={'level_percent': 80.0}, torque=self.torque.copy(),
                        velocity=np.zeros(24), temperature=self.temperature.copy())

    def manager_state(self):
        return self.manager

    def control_state(self):
        return {'power': bool(self.enables), 'servo': bool(self.enables), 'manager': 'Enabled' if self.enables
                else 'Idle', 'control': 'Idle', 'unlimited_mode': False}

    def enable(self, unlimited_mode=False, reset_faults=False):
        self.enables.append(unlimited_mode)
        if self.manager in ('MinorFault', 'MajorFault'):
            if not reset_faults:
                raise RuntimeError(f'the control manager is in {self.manager}')
            self.resets += 1
            self.manager = 'Enabled'
            return ['power_on', 'servo_on', 'reset_fault (MajorFault)', 'enable_control_manager']
        return ['power_on', 'servo_on', 'enable_control_manager']

    def send_joints(self, goals, *, minimum_time, velocity, acceleration, hold_s, priority):
        handle = FakeHandle(self, {c: np.asarray(g) for c, g in goals.items()}, self.steps)
        handle.code = self.code
        self.commands.append({'goals': goals, 'minimum_time': minimum_time, 'velocity': velocity, 'handle': handle})
        return handle

    def stop(self):
        running = [c for c in self.commands if not c['handle'].cancelled and c['handle'].polls < c['handle'].steps]
        for command in self.commands:
            command['handle'].cancel()
        self.holds.append(len(running))
        return len(running)

    def controller_command(self, kind, component, spec, *, hold_s, priority):
        goals = {}
        if kind == 'joint_impedance':
            goals = {component: np.asarray(spec['position'])}
        elif kind == 'joint_group':
            torso = component_of(self.q, 'torso').copy()
            for joint, value in zip(spec['joints'], spec['position']):
                torso[JOINTS.index(joint) - 2] = value
            goals = {'torso': torso}
        elif kind == 'jog':
            index = JOINTS.index(spec['joint'])
            name = next(c for c, (a, b) in COMPONENTS.items() if a <= index < b)
            values = component_of(self.q, name).copy()
            values[index - COMPONENTS[name][0]] = spec['value'] if spec['mode'] == 'absolute' else \
                values[index - COMPONENTS[name][0]] + spec['value']
            goals = {name: values}
        handle = FakeHandle(self, goals, self.steps)
        handle.code = self.code
        self.controller_commands.append({'kind': kind, 'component': component, 'spec': spec, 'hold_s': hold_s,
                                         'handle': handle})
        return handle

    def controller_info(self, what, name=None, count=50):
        return {'what': what, 'name': name}

    def base_stream(self, *, priority, accel, hold_s):
        stream = FakeBaseStream(self)
        self.streams.append(stream)
        return stream

    def set_tool_flange_output_voltage(self, side, volts):
        self.flanges.append((side, volts))
        return True

    def close(self):
        pass


class FakeCamera:
    def __init__(self):
        self.rgb = np.full((360, 640, 3), 120, np.uint8)
        self.depth = np.full((360, 640), 1.5, np.float32)
        self.closed = False

    def read(self):
        return time.time(), self.rgb, self.depth

    def intrinsics(self):
        return [[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]]

    def close(self):
        self.closed = True


class FakeGripper:
    def __init__(self):
        self.homed, self.homings, self.commands = False, 0, []
        self.opening = {'right': 0.1, 'left': 0.1}
        self.object_width = None   # the fingers stop on an object this wide

    def home(self, guard=None):
        self.homings += 1
        if guard:
            guard()
        self.homed = True

    def command(self, side, width_m, current_a):
        self.commands.append((side, width_m, current_a))
        self.opening[side] = max(width_m, self.object_width or 0.0)

    def hold(self):
        pass

    def widths(self):
        return dict(self.opening)

    def close(self):
        pass


@pytest.fixture(scope='module')
def kinematics():
    return Kinematics(urdf_path=OFFLINE_URDF)


@pytest.fixture
def make(tmp_path, kinematics):
    made = []

    def build(robot=None, **options):
        config = Configuration(root=tmp_path/'session', task='t', robot='rby1',
                               backend_options={'stream_enabled': False, 'control_hz': 500.0, 'watchdog_hz': 0,
                                                'controller_commands': False, **options})
        backend = Rby1Backend(config, robot=robot or FakeRobot(), kinematics=kinematics, camera=FakeCamera(),
                              gripper=FakeGripper())
        made.append(backend)
        return backend
    yield build
    for backend in made:
        backend.close()


def arm(backend):
    arming.write(backend.config.root, True)


def test_tools_match_the_workspace_facts(make):
    from services.workspace.setup import ROBOT_FACTS
    backend = make()
    names = {tool['name'] for tool in make(controller_commands=True).tools()}  # the production default
    assert names == set(ROBOT_FACTS['rby1']['tools'])
    assert backend.mutating - set(CONTROLLER_TOOLS) <= names  # those are listed only when enabled
    for tool in backend.tools():
        assert tool['inputSchema']['additionalProperties'] is False


def test_connecting_commands_nothing_and_the_first_armed_motion_enables(make):
    robot = FakeRobot()
    backend = make(robot)
    backend.status()
    backend.call('rby1_observe', {})
    assert robot.enables == [] and robot.commands == []
    arm(backend)
    backend.validate('rby1_move_joints', {'component': 'head', 'positions': [0.1, 0.1]})
    assert robot.enables == [False] and robot.commands == []  # unlimited mode stays off
    backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.1, 0.1]})
    backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.0, 0.0]})
    assert set(robot.enables) == {False} and len(robot.commands) == 2


def test_motion_requires_arming_and_a_released_emo(make):
    robot = FakeRobot()
    backend = make(robot)
    arguments = {'component': 'head', 'positions': [0.1, 0.0]}
    with pytest.raises(PermissionError, match='not armed'):
        backend.validate('rby1_move_joints', arguments)
    arm(backend)
    robot.emo = True
    with pytest.raises(PermissionError, match='emergency stop'):
        backend.validate('rby1_move_joints', arguments)
    assert robot.commands == [] and robot.enables == []
    robot.emo = False
    with pytest.raises(PermissionError, match='not armed'):  # the EMO disarmed the session
        backend.validate('rby1_move_joints', arguments)
    arm(backend)
    backend.validate('rby1_move_joints', arguments)


def test_joint_move_is_timed_by_the_speed_cap_and_settles(make):
    robot = FakeRobot()
    backend = make(robot)
    arm(backend)
    goal = component(READY, 'right_arm') + np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
    value, _ = backend.call('rby1_move_joints', {'component': 'right_arm', 'positions': goal.tolist()})
    command = robot.commands[-1]
    assert command['minimum_time'] == pytest.approx(0.3 / 0.2 + 0.2 / 0.4)  # trapezoid: cruise + ramp
    assert command['velocity'] == {'right_arm': 0.2}
    assert value['settled'] and value['max_joint_error_rad'] <= 0.02
    assert np.allclose(component(robot.q, 'right_arm'), goal)
    assert value['right_ee']['position'] and value['finish'] == 'Ok'


def test_limits_step_cap_and_torso_rule_refuse_before_moving(make):
    robot = FakeRobot()
    backend = make(robot)
    arm(backend)
    with pytest.raises(ValueError, match='joint limits'):
        backend.validate('rby1_move_joints', {'component': 'head', 'positions': [0.6, 0.0]})
    with pytest.raises(ValueError, match='per call'):
        backend.validate('rby1_move_joints', {'component': 'right_arm', 'relative': True,
                                              'positions': [1.3, 0, 0, 0, 0, 0, 0]})
    with pytest.raises(ValueError, match='upright rule'):
        backend.validate('rby1_move_joints', {'component': 'torso', 'relative': True,
                                              'positions': [0, 0.1, 0, 0, 0, 0]})
    with pytest.raises(ValueError, match='has 2 joints'):
        backend.validate('rby1_move_joints', {'component': 'head', 'positions': [0.0] * 3})
    with pytest.raises(ValueError, match='unknown torso preset'):
        backend.validate('rby1_move_joints', {'component': 'torso', 'preset': 'crouch'})
    backend.validate('rby1_move_joints', {'component': 'torso', 'preset': 'ready'})
    assert robot.commands == []


def test_disarm_mid_motion_holds_and_is_a_handled_stop(make):
    robot = FakeRobot()
    robot.steps = 50
    backend = make(robot)
    arm(backend)

    def disarm(handle):
        if handle.polls == 3:
            arming.write(backend.config.root, False)
    robot.on_poll = disarm
    with pytest.raises(MotionStopped, match='disarmed') as caught:
        backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.3, 0.0]})
    assert caught.value.handled_stop is True and 'head' in caught.value.robot_state
    assert robot.commands[-1]['handle'].cancelled and robot.holds == [1]
    assert 0 < robot.q[22] < 0.3


def test_emo_mid_motion_cancels_holds_and_disarms_until_the_operator_rearms(make):
    robot = FakeRobot()
    robot.steps = 50
    backend = make(robot)
    arm(backend)
    robot.on_poll = lambda handle: setattr(robot, 'emo', handle.polls >= 2)
    with pytest.raises(MotionStopped, match='emergency stop'):
        backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.3, 0.0]})
    assert robot.holds and arming.read(backend.config.root)['armed'] is False
    robot.on_poll, robot.emo = None, False  # released: still refused until the operator arms again
    with pytest.raises(PermissionError, match='not armed .*emergency stop'):
        backend.validate('rby1_move_joints', {'component': 'head', 'positions': [0.0, 0.0]})
    arm(backend)
    backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.0, 0.0]})
    assert len(robot.enables) == 2  # power / servo / control manager re-checked before every armed motion


def test_stale_joint_state_refuses_and_stops(make):
    robot = FakeRobot()
    backend = make(robot)
    arm(backend)
    robot.age[22] = 5.0
    with pytest.raises(PermissionError, match='no fresh joint state from head'):
        backend.validate('rby1_move_joints', {'component': 'head', 'positions': [0.1, 0.0]})
    robot.age[22] = 0.01
    robot.steps = 50
    robot.on_poll = lambda handle: robot.age.__setitem__(22, 5.0 if handle.polls >= 2 else 0.01)
    with pytest.raises(MotionStopped, match='stale'):
        backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.3, 0.0]})


def test_stalled_motion_is_cancelled_and_held(make):
    robot = FakeRobot()
    robot.frozen = True
    backend = make(robot, motion_stall_s=0.1)
    arm(backend)
    with pytest.raises(MotionStopped, match='no progress'):
        backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.2, 0.0]})
    assert robot.holds


def test_a_controller_finish_code_other_than_ok_is_a_handled_stop(make):
    robot = FakeRobot()
    robot.code = 'Preempted'
    backend = make(robot)
    arm(backend)
    with pytest.raises(MotionStopped, match='Preempted') as caught:
        backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.1, 0.0]})
    assert caught.value.handled_stop is True


def test_wrist_force_rise_stops_an_arm_motion(make):
    robot = FakeRobot()
    robot.steps = 50
    backend = make(robot, ft_guard_n=30.0)
    arm(backend)
    robot.on_poll = lambda handle: robot.force.__setitem__('right', [0.0, 0.0, 45.0 if handle.polls >= 3 else 0.0])
    with pytest.raises(MotionStopped, match='right wrist force rose 45 N'):
        backend.call('rby1_move_joints', {'component': 'right_arm', 'relative': True,
                                          'positions': [0, 0, 0, 0.3, 0, 0, 0]})


def test_move_body_moves_components_in_one_command_finishing_together(make):
    robot = FakeRobot(q=np.zeros(24))
    backend = make(robot)
    arm(backend)
    value, _ = backend.call('rby1_move_body', {'targets': {'torso': {'preset': 'ready'}, 'head': {'positions': [0.2, 0.2]}}})
    command = robot.commands[-1]
    assert set(command['goals']) == {'torso', 'head'}
    # torso_2 moves 1.047 rad at 0.1 rad/s (0.2 rad/s^2 ramps): it sets the pace for both
    assert command['minimum_time'] == pytest.approx(np.radians(60) / 0.1 + 0.1 / 0.2)
    assert value['components'] == ['head', 'torso'] and value['settled']


def test_eef_move_reaches_a_nearby_pose(make, kinematics):
    robot = FakeRobot()
    backend = make(robot)
    arm(backend)
    start = kinematics.frame(EE_FRAME['right'], robot.q)[:3, 3]
    value, _ = backend.call('rby1_move_eef', {'arm': 'right', 'position': [0.03, 0.0, 0.03], 'relative': True})
    assert value['position_error_mm'] < 5
    assert np.allclose(value['right_ee']['position'], start + [0.03, 0.0, 0.03], atol=0.005)


def test_eef_reach_step_and_height_are_refused_before_moving(make, kinematics):
    robot = FakeRobot()
    backend = make(robot)
    arm(backend)
    with pytest.raises(ValueError, match='out of reach'):
        backend.validate('rby1_move_eef', {'arm': 'right', 'position': [1.2, -0.3, 1.2]})
    with pytest.raises(ValueError, match='per call'):
        backend.validate('rby1_move_eef', {'arm': 'right', 'position': [0.0, 0.0, 0.3], 'relative': True})
    assert robot.commands == [] and ARM_REACH_M == 0.69
    high_floor = make(robot, ee_min_z_m=1.25)  # the ready hand is at z = 1.29 m
    arm(high_floor)
    with pytest.raises(ValueError, match='below 1.25 m'):
        high_floor.validate('rby1_move_eef', {'arm': 'right', 'position': [0.0, 0.0, -0.1], 'relative': True})


def test_self_collision_refuses_a_path_into_the_body_but_lets_a_motion_leave_contact(make, kinematics):
    robot = FakeRobot()
    backend = make(robot)
    arm(backend)
    # a right-arm pose whose forearm (link_right_arm_5) sits inside the upper torso
    into_body = with_component(robot.q, 'right_arm', [-0.082, -0.12, 0.632, -2.249, 0.665, 0.106, -0.921])
    assert kinematics.collisions(into_body) == {('link_right_arm_5', 'link_torso_5')}
    with pytest.raises(ValueError, match="self-collision .*link_right_arm_5.*link_torso_5"):
        backend._check_path(robot.q, {'right_arm': component(into_body, 'right_arm')})
    backend._check_path(into_body, {'right_arm': component(robot.q, 'right_arm')})  # leaving contact is allowed


def test_look_at_points_the_head_camera(make, kinematics):
    robot = FakeRobot()
    backend = make(robot)
    arm(backend)
    point = [1.2, 0.3, 0.8]
    value, _ = backend.call('rby1_look_at', {'point': point})
    T = kinematics.frame(HEAD_CAMERA, robot.q)
    ray = (np.asarray(point) - T[:3, 3]) / np.linalg.norm(np.asarray(point) - T[:3, 3])
    assert np.degrees(np.arccos(T[:3, 2] @ ray)) < 2.0 and value['predicted_error_deg'] < 2.0
    with pytest.raises(ValueError, match='cannot point'):
        backend.validate('rby1_look_at', {'point': [-2.0, 0.0, 1.0]})


def test_base_move_drives_then_turns_closed_loop_on_odometry(make):
    robot = FakeRobot()
    backend = make(robot, base_hz=200.0)
    arm(backend)
    value, _ = backend.call('rby1_base_move', {'forward': 0.3, 'turn': 0.5})
    measured = value['measured_by_odometry']
    assert measured['forward'] == pytest.approx(0.3, abs=0.03) and measured['turn'] == pytest.approx(0.5, abs=0.04)
    sends = robot.streams[-1].sends
    assert max(abs(v) for v, _ in sends) <= 0.2 and max(abs(w) for _, w in sends) <= 0.4
    assert robot.streams[-1].closed and [s['segment'] for s in value['segments']] == ['forward', 'turn']
    with pytest.raises(ValueError, match='per call'):
        backend.validate('rby1_base_move', {'forward': 1.5})


def test_short_base_moves_arrive_within_the_tolerance(make):
    robot = FakeRobot()
    backend = make(robot, base_hz=200.0)
    arm(backend)
    for arguments, key, goal in (({'forward': 0.05}, 'forward', 0.05), ({'turn': 0.1}, 'turn', 0.1)):
        value, _ = backend.call('rby1_base_move', {**arguments, 'max_linear': 0.05, 'max_angular': 0.1})
        assert value['measured_by_odometry'][key] == pytest.approx(goal, abs=0.01)


def test_base_that_does_not_move_is_stopped(make):
    robot = FakeRobot()
    robot.base_stuck = True
    backend = make(robot, base_hz=200.0, base_stall_s=0.1)
    arm(backend)
    with pytest.raises(MotionStopped, match='did not move') as caught:
        backend.call('rby1_base_move', {'forward': 0.2})
    assert caught.value.handled_stop is True and robot.streams[-1].closed


def test_gripper_homes_on_first_use_and_reports_an_object(make):
    backend = make()
    arm(backend)
    backend.gripper.object_width = 0.04
    value, _ = backend.call('rby1_gripper', {'side': 'right', 'action': 'close', 'current_a': 0.6})
    assert value['homed_now'] and value['measured_width_m'] == 0.04 and value['blocked_by_object']
    value, _ = backend.call('rby1_gripper', {'side': 'right', 'action': 'open'})
    assert not value['homed_now'] and backend.gripper.homings == 1
    with pytest.raises(ValueError, match='current_a'):
        backend.validate('rby1_gripper', {'side': 'left', 'action': 'close', 'current_a': 3.0})
    with pytest.raises(ValueError, match='needs width_m'):
        backend.validate('rby1_gripper', {'side': 'left', 'action': 'width'})


def test_observe_saves_the_frame_and_pixel_to_position_uses_fk(make, kinematics):
    robot = FakeRobot()
    backend = make(robot)
    value, images = backend.call('rby1_observe', {})
    directory = backend.frames_root/value['frame_id']
    meta = json.loads((directory/'meta.json').read_text())
    entry = meta['cameras']['head_camera']
    assert images and entry['depth'] == 'head_camera_depth.png'
    assert cv.imread(str(directory/'head_camera_depth.png'), cv.IMREAD_UNCHANGED)[0, 0] == 1500
    points, _ = backend.call('rby1_pixel_to_position', {'frame_id': value['frame_id'], 'camera': 'head_camera',
                                                        'pixels': [[320, 180], [900, 10]]})
    T = kinematics.frame(HEAD_CAMERA, robot.q)
    assert np.allclose(points['points'][0]['position'], T[:3, 3] + 1.5 * T[:3, 2], atol=1e-3)
    assert points['points'][1]['error'] == 'outside the image'


def test_status_reports_without_commanding_and_close_is_idempotent(make):
    robot = FakeRobot()
    backend = make(robot)
    value = backend.status()
    assert value['robot'] == 'rainbow_rby1_a' and value['motion']['armed'] is False
    assert value['enabled'] is False and value['control']['manager'] == 'Idle'
    assert set(value['limits']) == {'torso', 'right_arm', 'left_arm', 'head'} and value['stale_joints'] == []
    assert value['grippers'] == {'homed': False, 'openings_m': {'right': 0.1, 'left': 0.1}}
    json.dumps(value)
    backend.close()
    backend.close()
    assert robot.commands == [] and robot.holds == [0]  # close cancels what runs: nothing ever did


def test_hold_is_always_allowed_and_cancels_without_commanding(make):
    robot = FakeRobot()
    backend = make(robot)
    value, _ = backend.call('rby1_hold', {})
    assert value['held'] and robot.holds == [0] and robot.commands == []
    arm(backend)
    backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.1, 0.0]})
    arming.write(backend.config.root, False)
    backend.call('rby1_hold', {})
    assert len(robot.commands) == 1  # hold sends no command of its own


def test_unknown_options_are_rejected(tmp_path, kinematics):
    config = Configuration(root=tmp_path/'s', task='t', robot='rby1', backend_options={'speed': 9})
    with pytest.raises(ValueError, match='unknown rby1 backend options'):
        Rby1Backend(config, robot=FakeRobot(), kinematics=kinematics, camera=FakeCamera(), gripper=FakeGripper())


def test_head_camera_calibration_and_serial_are_operator_options(tmp_path, kinematics):
    """The built-in hand-eye calibration belongs to one unit: `head_camera_extrinsic` replaces it.
    No camera serial is built in: None opens the first ZED found."""
    from services.rby1.backend import DEFAULTS
    from services.rby1.kinematics import HEAD_LINK, X_HEAD_CAMERA
    assert DEFAULTS['camera_serial'] is None and DEFAULTS['head_camera_extrinsic'] is None
    q = FakeRobot().q
    shifted = np.eye(4)
    shifted[:3, 3] = [0.01, -0.02, 0.03]
    config = Configuration(root=tmp_path/'s', task='t', robot='rby1', backend_options={
        'urdf_path': str(OFFLINE_URDF), 'head_camera_extrinsic': shifted.tolist(), 'stream_enabled': False,
        'watchdog_hz': 0})
    backend = Rby1Backend(config, robot=FakeRobot(), camera=FakeCamera(), gripper=FakeGripper())
    try:
        link = backend.kinematics.frame(HEAD_LINK, q)
        assert np.allclose(backend.kinematics.frame(HEAD_CAMERA, q), link @ shifted)
        assert np.allclose(kinematics.frame(HEAD_CAMERA, q), link @ X_HEAD_CAMERA)  # the default
    finally:
        backend.close()
    with pytest.raises(ValueError, match='4x4 homogeneous'):
        Kinematics(urdf_path=OFFLINE_URDF, head_camera=np.zeros((4, 4)))


def test_recorder_writes_frames_depth_robot_log_and_index(tmp_path, make):
    backend = make(stream_enabled=True, stream_fps=50.0, robot_log_hz=50.0)
    time.sleep(1.3)
    stream = backend.stream
    index = json.loads((stream/'index.json').read_text())
    assert index['cameras']['head_camera']['newest_ms'] is not None
    frames = list((stream/'head_camera').rglob('*.jpg'))
    depths = list((stream/'head_camera').rglob('*_depth.png'))
    records = [json.loads(line) for path in (stream/'robot').glob('*.jsonl') for line in path.read_text().splitlines()]
    assert frames and depths and records
    assert {'torso', 'right_arm', 'left_arm', 'head', 'right_ee', 'odometry', 'ft_force'} <= set(records[-1])
    value, images = backend.call('rby1_stream', {'seconds_ago': [0]})
    assert images and value['frames'][0]['images'][0]['camera'] == 'head_camera'


def test_sdk_command_builders_construct_offline_for_every_component():
    import rby1_sdk as rby
    from services.rby1.robot import SdkRobot
    shell = SdkRobot.__new__(SdkRobot)  # no connection: only the builder code runs
    shell.rby = rby
    goals = {name: np.zeros(stop - start) for name, (start, stop) in COMPONENTS.items()}
    command = shell.joint_command(goals, 1.0, {name: 0.3 for name in goals}, 1.0, 1.0)
    assert isinstance(command, rby.RobotCommandBuilder)
    shell.joint_command({'head': np.zeros(2)}, 1.0, {'head': 0.5}, 1.0, 1.0)
    shell.joint_command(goals, 1.0, {name: 0.3 for name in goals}, {name: 0.4 for name in goals}, 1.0)
    with pytest.raises(ValueError, match='unknown components'):
        shell.joint_command({'wheels': np.zeros(2)}, 1.0, {'wheels': 0.1}, 1.0, 1.0)


def test_concurrent_calls_are_serialized(make):
    robot = FakeRobot()
    robot.steps = 20
    backend = make(robot)
    arm(backend)
    errors = []

    def move(target):
        try:
            backend.call('rby1_move_joints', {'component': 'head', 'positions': [target, 0.0]})
        except Exception as exc:
            errors.append(exc)
    threads = [threading.Thread(target=move, args=(t,)) for t in (0.1, 0.2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(robot.commands) == 2


class FakeDynamixel:
    """Two gripper motors under current control: each runs to the end it is pushed to; encoders jitter a count."""

    def __init__(self, ends=((0.3, 9.0), (0.2, 8.7))):
        self.ends, self.q, self.torque, self.ticks = ends, [4.0, 4.0], [0.0, 0.0], 0
        self.positions = []

    def group_sync_write_torque_enable(self, pairs):
        pass

    def group_sync_write_operating_mode(self, pairs):
        pass

    def group_sync_write_send_torque(self, pairs):
        for i, t in pairs:
            self.torque[i] = t

    def group_sync_write_send_position(self, pairs):
        self.positions.append(list(pairs))

    def group_fast_sync_read_encoder(self, ids):
        self.ticks += 1
        jitter = 0.0015 * (self.ticks % 2)
        for i in ids:
            low, high = self.ends[i]
            self.q[i] = min(high, self.q[i] + 1.5) if self.torque[i] > 0 else max(low, self.q[i] - 1.5)
        return [(i, self.q[i] + jitter) for i in ids]

    def ping(self, i):
        return True


def test_gripper_homing_tolerates_encoder_jitter_and_maps_openings(monkeypatch):
    from services.rby1 import gripper as gripper_module
    monkeypatch.setattr(gripper_module.time, 'sleep', lambda s: None)
    robot = FakeRobot()
    gripper = gripper_module.Gripper(robot)
    gripper.bus = FakeDynamixel()
    gripper.home()
    try:
        assert robot.flanges == [('right', 12), ('left', 12)] and gripper.homed
        assert np.allclose(gripper.min_q, [0.3, 0.2], atol=0.002) and np.allclose(gripper.max_q, [9.0, 8.7], atol=0.002)
        gripper.command('right', 0.0, 0.5)   # closed = the encoder maximum (opens towards the minimum)
        assert gripper.target['right'] == pytest.approx(gripper.max_q[0])
        gripper.command('left', 0.1, 0.5)
        assert gripper.target['left'] == pytest.approx(gripper.min_q[1])
    finally:
        gripper.close()


def test_gripper_homing_that_finds_little_travel_is_refused(monkeypatch):
    from services.rby1 import gripper as gripper_module
    monkeypatch.setattr(gripper_module.time, 'sleep', lambda s: None)
    gripper = gripper_module.Gripper(FakeRobot())
    gripper.bus = FakeDynamixel(ends=((4.0, 4.3), (0.2, 8.7)))
    with pytest.raises(RuntimeError, match='too little travel for right'):
        gripper.home()
    assert not gripper.homed


# ----- torque, end stops and faults (after the 2026-10-08 torso_2 MajorFault) --------------------------------
def test_controller_fault_disarms_and_is_reset_only_after_the_operator_arms_again(make):
    robot = FakeRobot()
    backend = make(robot)
    arm(backend)
    robot.manager = 'MajorFault'
    with pytest.raises(PermissionError, match='MajorFault.*operator'):
        backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.1, 0.0]})
    assert robot.resets == 0 and not robot.commands
    assert arming.read(backend.config.root)['armed'] is False and backend.fault['what'].endswith('MajorFault')
    arm(backend)  # the operator inspected the robot and armed again: now the fault is reset
    backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.1, 0.0]})
    assert robot.resets == 1
    robot.manager = 'MajorFault'  # a new fault after that needs its own disarm and re-arm
    with pytest.raises(PermissionError, match='MajorFault'):
        backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.0, 0.0]})
    assert robot.resets == 1


def test_torso_squat_limit_refuses_the_pose_that_faulted(make):
    backend = make(FakeRobot())
    arm(backend)
    with pytest.raises(ValueError, match='squat limit'):
        backend.validate('rby1_move_joints', {'component': 'torso', 'positions': [0, 1.28, -2.56, 1.28, 0, 0]})
    backend.validate('rby1_move_joints', {'component': 'torso', 'positions': [0, 1.0, -2.0, 1.0, 0, 0]})


def test_gravity_load_along_the_path_is_checked(make):
    backend = make(FakeRobot(), static_torque_fraction=0.3)  # below what the ready arms need (about 45 %)
    arm(backend)
    with pytest.raises(ValueError, match='of its rated'):
        backend.validate('rby1_move_joints', {'component': 'right_arm', 'preset': 'ready'})
    backend = make(FakeRobot())
    arm(backend)
    backend.validate('rby1_move_joints', {'component': 'right_arm', 'preset': 'ready'})


def test_torque_near_the_rating_cancels_and_retargets_in_place(make):
    robot = FakeRobot()
    robot.steps = 50
    backend = make(robot)
    arm(backend)

    def push(handle):
        if handle.polls >= 3:
            robot.torque[4] = -697.0  # torso_2 against its end stop
    robot.on_poll = push
    goal = component(READY, 'torso') + np.array([0, 0.1, -0.2, 0.1, 0, 0])
    with pytest.raises(MotionStopped, match='torque over 150% of its rating.*torso_2'):
        backend.call('rby1_move_joints', {'component': 'torso', 'positions': goal.tolist()})
    relief = robot.commands[-1]
    assert len(robot.commands) == 2 and set(relief['goals']) == {'torso', 'right_arm', 'left_arm'}
    assert np.allclose(relief['goals']['torso'], component(robot.q, 'torso'))  # where it stopped: no motion
    assert backend.events[-1]['what'].startswith('re-targeted')


def test_joint_short_of_its_target_is_retargeted_and_reported(make):
    robot = FakeRobot()
    backend = make(robot, exact_timeout_s=0.05)
    arm(backend)
    goal = component(READY, 'right_arm') + np.array([0, 0, 0, -0.2, 0, 0, 0])
    stop = goal[3] + 0.006  # an end stop 6 mrad before the target

    def blocked(handle):
        robot.q[11] = max(robot.q[11], stop)
    robot.on_poll = blocked
    value, _ = backend.call('rby1_move_joints', {'component': 'right_arm', 'positions': goal.tolist()})
    assert value['settled'] is False and list(value['blocked_rad']) == ['right_arm_3'] and 'warning' in value
    relief = robot.commands[-1]
    assert relief['goals']['right_arm'][3] == pytest.approx(stop)


def test_drive_not_ready_mid_motion_stops_and_disarms(make):
    robot = FakeRobot()
    robot.steps = 50
    backend = make(robot)
    arm(backend)
    robot.on_poll = lambda handle: robot.ready.__setitem__(4, handle.polls < 3)
    with pytest.raises(MotionStopped, match='not ready.*torso_2'):
        backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.2, 0.0]})
    assert arming.read(backend.config.root)['armed'] is False


def test_hot_drive_refuses_motion(make):
    robot = FakeRobot()
    robot.temperature[4] = 80.0
    backend = make(robot)
    arm(backend)
    with pytest.raises(PermissionError, match='hotter'):
        backend.validate('rby1_move_joints', {'component': 'head', 'positions': [0.1, 0.0]})


def test_watchdog_disarms_on_a_fault_and_retargets_a_push_between_calls(make):
    robot = FakeRobot()
    backend = make(robot, watchdog_torque_s=0.0)
    arm(backend)
    backend._watch_once()
    assert not robot.commands  # nothing before the session has taken the drives
    backend.call('rby1_move_joints', {'component': 'head', 'positions': [0.1, 0.0]})
    robot.torque[11] = 61.0  # right elbow at 153 % of its 40 N m
    backend._watch_once()
    backend._watch_once()
    assert set(robot.commands[-1]['goals']) == {'torso', 'right_arm', 'left_arm'}
    robot.manager = 'MajorFault'
    backend._watch_once()
    assert arming.read(backend.config.root)['armed'] is False and backend.fault is not None
    assert backend.status()['safety']['fault'] == backend.fault


def test_torque_up_to_150_percent_does_not_stop_and_over_it_stops_at_once(make):
    robot = FakeRobot()
    robot.steps = 50
    backend = make(robot)
    robot.on_poll = lambda handle: robot.torque.__setitem__(13, -14.5)  # right wrist at 145 %: carrying a load
    arm(backend)
    goal = component(READY, 'right_arm') + np.array([0, 0, 0, 0, 0, 0, 0.2])
    backend.call('rby1_move_joints', {'component': 'right_arm', 'positions': goal.tolist()})
    robot.on_poll = lambda handle: robot.torque.__setitem__(13, -15.5 if handle.polls >= 3 else 0.0)
    with pytest.raises(MotionStopped, match='over 150% of its rating.*right_arm_5'):
        backend.call('rby1_move_joints', {'component': 'right_arm', 'positions': component(READY, 'right_arm').tolist()})


def test_every_preset_is_inside_the_limits_and_the_load_check(make):
    backend = make(FakeRobot())
    arm(backend)
    for name, presets in PRESETS.items():
        for preset in presets:
            goal = backend._joint_goal(READY, name, preset=preset)
            q = with_component(READY, name, goal)
            load = np.abs(backend.kinematics.gravity(q)) * backend.options['gravity_model_scale']
            assert np.max(load / backend.kinematics.torque_limit) < backend.options['static_torque_fraction'], (name, preset)


def test_no_driving_while_low_leaning_or_stretched(make):
    robot = FakeRobot()
    backend = make(robot)
    arm(backend)
    backend.validate('rby1_base_move', {'forward': 0.1})  # READY: allowed
    for name, values in (('torso', [0, 1.047, -2.094, 1.047, 0, 0]), ('torso', [0, 0.5236, -1.0472, 0.6736, 0, 0]),
                         ('right_arm', PRESETS['right_arm']['side'])):
        robot.q = with_component(READY, name, values)
        with pytest.raises(ValueError, match='no driving'):
            backend.call('rby1_base_move', {'forward': 0.1})
    assert not robot.streams


# ----- planning queries: no arming, no command -----------------------------------------------------------
def test_fk_reports_the_current_and_a_hypothetical_posture_without_moving(make, kinematics):
    robot = FakeRobot()
    backend = make(robot)  # not armed
    value, _ = backend.call('rby1_fk', {})
    hand = kinematics.frame(EE_FRAME['right'], robot.q)[:3, 3]
    assert np.allclose(value['end_effectors']['right']['position'], hand, atol=1e-3)
    assert value['hypothetical'] is False and not value['outside_limits'] and not value['self_collisions']
    low, _ = backend.call('rby1_fk', {'joints': {'torso': {'preset': 'lowest'}}, 'frames': ['link_head_2']})
    assert low['end_effectors']['right']['position'][2] < value['end_effectors']['right']['position'][2] - 0.3
    assert 'link_head_2' in low['frames'] and abs(low['center_of_mass'][0]) < 0.2
    deep, _ = backend.call('rby1_fk', {'joints': {'torso': {'positions': [0, 1.28, -2.56, 1.28, 0, 0]}}})
    assert 'torso_1' in deep['outside_limits'] and 'torso_2' in deep['outside_limits']
    assert not robot.commands and not robot.enables


def test_check_gives_the_motion_tools_verdict_without_arming_or_moving(make):
    robot = FakeRobot()
    backend = make(robot)
    ok, _ = backend.call('rby1_check', {'tool': 'rby1_move_body', 'arguments': {'targets': {'torso': {'preset': 'mid'}}}})
    assert ok['accepted'] and ok['planned_s'] > 2 and ok['after']['end_effectors']['right']['position'][2] < 1.25
    assert ok['ready_to_move']['armed'] is False
    deep, _ = backend.call('rby1_check', {'tool': 'rby1_move_joints', 'arguments': {
        'component': 'torso', 'positions': [0, 1.28, -2.56, 1.28, 0, 0]}})
    assert deep['accepted'] is False and 'squat' in deep['reason']
    drive, _ = backend.call('rby1_check', {'tool': 'rby1_base_move', 'arguments': {'forward': 0.3}})
    assert drive['accepted'] is True
    with pytest.raises(ValueError, match='unknown'):
        backend.call('rby1_check', {'tool': 'rby1_move_eef', 'arguments': {'arm': 'right', 'position': [0, 0, 0],
                                                                            'speed': 1}})
    assert not robot.commands and not robot.enables and not robot.streams


# ----- the controller's own command modes (wired SDK builders) ----------------------------------------------
def test_controller_modes_are_off_until_enabled_and_never_tune_the_robot(make):
    names = {t['name'] for t in make(FakeRobot()).tools()}
    assert not set(CONTROLLER_TOOLS) & names and 'rby1_ik' not in names and 'rby1_controller_info' in names
    enabled = {t['name'] for t in make(FakeRobot(), controller_commands=True, controller_tools_disabled=[]).tools()}
    assert set(CONTROLLER_TOOLS) <= enabled
    assert not any(word in ' '.join(enabled) for word in ('gravity', 'parameter_set', 'pid_set', 'gain'))
    backend = make(FakeRobot(), controller_commands=True, controller_tools_disabled=[])
    arm(backend)
    for name, arguments in (('rby1_optimal_control', {'component': 'body', 'stop_cost': 1.0}),
                            ('rby1_cartesian_impedance', {'arm': 'right', 'position': [0, 0, 0], 'stiffness': 5}),
                            ('rby1_cartesian', {'component': 'right_arm',
                                                'targets': [{'link': 'ee_right', 'position': [0, 0, 0], 'gain': 1}]})):
        with pytest.raises(ValueError, match='unknown argument'):
            backend.validate(name, arguments)


def test_cartesian_checks_targets_runs_supervised_and_ends_in_a_joint_hold(make):
    robot = FakeRobot()
    backend = make(robot, controller_commands=True)
    with pytest.raises(PermissionError, match='not armed'):
        backend.validate('rby1_cartesian', {'component': 'right_arm',
                                            'targets': [{'link': 'ee_right', 'position': [0.02, 0, 0], 'relative': True}]})
    arm(backend)
    for position, reason in (([0.3, 0, 0], 'per call'), ([0.2, 0, 0], 'out of reach for the arm alone')):
        with pytest.raises(ValueError, match=reason):
            backend.validate('rby1_cartesian', {'component': 'right_arm', 'targets': [
                {'link': 'ee_right', 'position': position, 'relative': True}]})
    with pytest.raises(ValueError, match='outside'):
        backend.validate('rby1_cartesian', {'component': 'body', 'targets': [
            {'link': 'ee_right', 'position': [0.02, 0, 0], 'relative': True}],
            'joint_targets': [{'joint': 'torso_1', 'position': 1.28}]})
    value, _ = backend.call('rby1_cartesian', {'component': 'body', 'linear_velocity': 0.05, 'targets': [
        {'link': 'ee_right', 'position': [0.05, 0, 0], 'relative': True}]})
    sent = robot.controller_commands[-1]
    assert sent['kind'] == 'cartesian' and sent['component'] == 'body'
    assert sent['spec']['targets'][0]['linear_velocity_limit'] == 0.05
    assert {j['joint'] for j in sent['spec']['joint_targets']} == {'torso_0', 'torso_4'}  # upright rule added
    assert sent['spec']['stop_position_tracking_error'] == backend.options['cartesian_stop_position_error_m']
    assert robot.commands and np.allclose(robot.commands[-1]['goals']['torso'], component(robot.q, 'torso'))  # hold
    assert value['mode'] == 'cartesian' and 'position_error_mm' in value['targets'][0]


def test_controller_command_leaving_the_envelope_is_cancelled(make):
    robot = FakeRobot()
    robot.steps = 50
    backend = make(robot, controller_commands=True)
    arm(backend)
    robot.on_poll = lambda handle: robot.q.__setitem__(2, 0.1 if handle.polls > 3 else 0.0)  # torso_0 rolls
    with pytest.raises(MotionStopped, match='upright'):
        backend.call('rby1_optimal_control', {'component': 'body', 'targets': [
            {'link': 'ee_right', 'position': [0.03, 0, 0], 'relative': True}]})
    assert robot.controller_commands[-1]['spec']['joint_targets'][0]['weight'] == 10.0


def test_optimal_control_uses_fixed_solver_settings_and_impedance_fixed_gains(make):
    robot = FakeRobot()
    backend = make(robot, controller_commands=True, controller_tools_disabled=[])
    arm(backend)
    backend.call('rby1_optimal_control', {'component': 'body', 'center_of_mass': {'position': [0.02, 0, 0.5]},
                                          'targets': [{'link': 'ee_left', 'position': [0, 0.02, 0], 'relative': True,
                                                       'translation_weight': 2}]})
    spec = robot.controller_commands[-1]['spec']
    assert spec['stop_cost'] == 2e-3 and spec['patience'] == 10 and spec['velocity_limit_scaling'] == 0.2
    assert spec['targets'][0]['translation_weight'] == 2 and spec['center_of_mass']['weight'] == 1.0
    backend.call('rby1_cartesian_impedance', {'arm': 'right', 'position': [0.02, 0, 0], 'relative': True})
    spec = robot.controller_commands[-1]['spec']
    assert spec['stiffness'] == [100.0] * 7 and spec['torque_limit'][0] == pytest.approx(0.6 * 70)


def test_joint_group_jog_and_link_impedance(make):
    robot = FakeRobot()
    backend = make(robot, controller_commands=True, controller_tools_disabled=[])
    arm(backend)
    with pytest.raises(ValueError, match='upright'):
        backend.validate('rby1_joint_group', {'joints': ['torso_1'], 'positions': [0.6]})
    backend.call('rby1_joint_group', {'joints': ['torso_5'], 'positions': [0.1]})
    assert component(robot.q, 'torso')[5] == pytest.approx(0.1)
    with pytest.raises(ValueError, match='exceeds'):
        backend.validate('rby1_jog', {'joint': 'right_arm_6', 'mode': 'relative', 'value': 0.5})
    before = robot.q[14]
    backend.call('rby1_jog', {'joint': 'right_arm_6', 'mode': 'relative', 'value': 0.1})
    assert robot.q[14] == pytest.approx(before + 0.1)
    assert robot.controller_commands[-1]['spec']['velocity_limit'] == backend.options['joint_speed_radps']
    started = time.monotonic()
    value, _ = backend.call('rby1_link_impedance', {'arm': 'left', 'position': [0, 0, 0.01], 'relative': True,
                                                    'seconds': 0.5})
    assert time.monotonic() - started >= 0.5 and value['finish'] == 'Timed'


def test_hold_stop_command_info_and_move_eef_in_controller_mode(make):
    robot = FakeRobot()
    backend = make(robot, controller_commands=True)
    backend.call('rby1_hold', {'mode': 'stop_command'})
    assert robot.controller_commands[-1]['kind'] == 'stop'
    value, _ = backend.call('rby1_controller_info', {'what': 'pid_gains', 'name': 'torso'})
    assert value['value'] == {'what': 'pid_gains', 'name': 'torso'}
    with pytest.raises(ValueError, match='needs name'):
        backend.call('rby1_controller_info', {'what': 'parameter'})
    arm(backend)
    value, _ = backend.call('rby1_move_eef', {'arm': 'right', 'position': [0.02, 0, 0], 'relative': True})
    assert robot.controller_commands[-1]['kind'] == 'cartesian' and value['arm'] == 'right'
    check, _ = backend.call('rby1_check', {'tool': 'rby1_cartesian', 'arguments': {
        'component': 'right_arm', 'targets': [{'link': 'ee_right', 'position': [0.02, 0, 0], 'relative': True}]}})
    assert check['accepted'] and 'targets only' in check['checked']


def test_every_controller_builder_constructs_offline():
    import rby1_sdk as rby
    from services.rby1.robot import SdkRobot
    shell = SdkRobot.__new__(SdkRobot)  # no connection: only the builder code runs
    shell.rby, shell.handlers, sent = rby, [], []

    class Capture:
        def send_command(self, command, priority):
            sent.append(command)
            return type('Done', (), {'is_done': lambda self: True})()
    shell.robot = Capture()
    T = np.eye(4).tolist()
    stops = {'stop_position_tracking_error': 1e-3, 'stop_orientation_tracking_error': 1e-3,
             'stop_joint_position_tracking_error': 1e-2}
    specs = {
        'cartesian': ('right_arm', {'minimum_time': 2, 'joint_targets': [], **stops, 'targets': [
            {'ref_link': 'base', 'link': 'ee_right', 'T': T, 'linear_velocity_limit': 0.1, 'angular_velocity_limit': 0.3,
             'acceleration_limit_scaling': 0.3}]}),
        'optimal_control': ('body', {'targets': [{'ref_link': 'base', 'link': 'ee_left', 'T': T, 'translation_weight': 1,
                                                  'rotation_weight': 1}],
                                     'joint_targets': [{'joint': 'torso_0', 'position': 0, 'weight': 10}],
                                     'center_of_mass': {'ref_link': 'base', 'position': [0, 0, 0.5], 'weight': 1},
                                     'velocity_limit_scaling': 0.2, 'acceleration_limit_scaling': 0.2,
                                     'error_scaling': 1, 'stop_cost': 2e-3, 'min_delta_cost': 2e-3, 'patience': 10}),
        'cartesian_impedance': ('left_arm', {'minimum_time': 2, **stops, 'joint_limits': [('left_arm_1', 0, 1)],
                                             'stiffness': [100] * 7, 'damping_ratio': 1, 'torque_limit': [10] * 7,
                                             'reset_reference': True, 'targets': [
                                                 {'ref_link': 'base', 'link': 'ee_left', 'T': T,
                                                  'linear_velocity_limit': 0.1, 'angular_velocity_limit': 0.3,
                                                  'linear_acceleration_limit': 0.5, 'angular_acceleration_limit': 1}]}),
        'link_impedance': ('right_arm', {'ref_link': 'base', 'link': 'ee_right', 'T': T, 'damping_ratio': 1,
                                         'translation_weight': [3000] * 3, 'rotation_weight': [50] * 3}),
        'joint_impedance': ('right_arm', {'position': [0] * 7, 'minimum_time': 2, 'velocity_limit': [0.2] * 7,
                                          'acceleration_limit': [0.4] * 7, 'stiffness': [100] * 7, 'damping_ratio': 1,
                                          'torque_limit': [10] * 7}),
        'joint_group': ('torso', {'joints': ['torso_5'], 'position': [0.1], 'minimum_time': 2,
                                  'velocity_limit': [0.1], 'acceleration_limit': [0.2]}),
        'jog': (None, {'joint': 'head_0', 'mode': 'relative', 'value': 0.1, 'velocity_limit': 0.3,
                       'acceleration_limit': 0.8}),
        'stop': (None, {}),
    }
    for kind, (part, spec) in specs.items():
        shell.controller_command(kind, part, spec, hold_s=0.2, priority=10)
    assert len(sent) == len(specs) and all(isinstance(c, rby.RobotCommandBuilder) for c in sent)
    assert not hasattr(shell, 'set_parameter') and not hasattr(shell, 'gravity_compensation')


def test_a_load_already_over_the_guard_never_locks_control(make):
    """The pnp_box session: a squeezed box kept the right wrist over the guard and every move was refused."""
    robot = FakeRobot()
    robot.steps = 20
    backend = make(robot)
    arm(backend)
    robot.torque[13] = -16.0  # right wrist at 160 % (over the 150 % guard) before the motion starts
    backend.call('rby1_move_eef', {'arm': 'right', 'position': [0, -0.002, 0], 'relative': True})  # eases: allowed
    backend.call('rby1_move_joints', {'component': 'left_arm', 'positions': (
        component(READY, 'left_arm') + [0, 0, 0, 0, 0, 0, 0.05]).tolist()})  # other arm: allowed
    robot.on_poll = lambda handle: robot.torque.__setitem__(13, -16.0 - 0.2 * handle.polls)  # keeps rising
    with pytest.raises(MotionStopped, match='150% of its rating.*right_arm_5'):
        backend.call('rby1_move_eef', {'arm': 'right', 'position': [0, -0.01, 0], 'relative': True})


def test_leaning_allows_about_0_7_rad_with_the_arms_out(make):
    backend = make(FakeRobot())
    arm(backend)
    a = 0.5236
    backend.validate('rby1_move_joints', {'component': 'torso', 'positions': [0, a, -2 * a, a + 0.7, 0, 0]})
    with pytest.raises(ValueError, match='of its rated'):
        backend.validate('rby1_move_joints', {'component': 'torso', 'positions': [0, a, -2 * a, a + 0.85, 0, 0]})


def test_production_defaults_enable_the_characterized_controller_modes_only():
    from services.rby1.backend import DEFAULTS
    assert DEFAULTS['controller_commands'] is True
    assert set(DEFAULTS['controller_tools_disabled']) == {'rby1_cartesian_impedance', 'rby1_link_impedance'}


def test_disabled_controller_modes_are_not_listed_and_cannot_be_called(make):
    backend = make(FakeRobot(), controller_commands=True)
    names = {t['name'] for t in backend.tools()}
    assert 'rby1_optimal_control' in names and 'rby1_cartesian' in names
    assert 'rby1_link_impedance' not in names and 'rby1_cartesian_impedance' not in names
    with pytest.raises(ValueError, match='unknown tool'):
        backend.call('rby1_link_impedance', {'arm': 'left', 'position': [0, 0, 0.01], 'relative': True, 'seconds': 1})


def test_an_sdk_error_while_driving_reaches_the_agent_whole_with_the_held_state(make):
    robot = FakeRobot()
    backend = make(robot)
    arm(backend)

    def expired(self, vx, wz, minimum_time):
        raise RuntimeError('This stream is expired')
    original = FakeBaseStream.send
    FakeBaseStream.send = expired
    try:
        with pytest.raises(RuntimeError, match='This stream is expired') as caught:
            backend.call('rby1_base_move', {'forward': 0.1})
    finally:
        FakeBaseStream.send = original
    assert not getattr(caught.value, 'handled_stop', False)  # not a backend stop: the frontend adds its traceback
    assert 'odometry' in caught.value.robot_state and robot.holds  # held, and the measured state attached
    backend.call('rby1_base_move', {'forward': 0.05})  # nothing is locked afterwards


def _fake_streams(done_after_hold=True, expire_first=0):
    calls, made = [], []

    class Stream:
        def __init__(self):
            self.expire = expire_first if not made else 0
            made.append(self)

        def is_done(self):
            return False

        def send_command(self, command, timeout):
            if self.expire:
                self.expire -= 1
                raise RuntimeError('This stream is expired')
            calls.append(('send', len(made)))

        def cancel(self):
            calls.append('cancel')

        def wait_for(self, timeout_ms):
            calls.append(('wait_for', timeout_ms))
            return done_after_hold

    class Robot:
        def create_command_stream(self, priority):
            return Stream()
    return Robot(), calls, made


def test_closing_a_base_stream_lets_the_hold_time_end_it_without_a_cancel():
    """The SDK's cancel cancels the robot's current control and can hit the next base move."""
    from services.rby1.robot import BaseStream
    robot, calls, _ = _fake_streams()
    BaseStream(robot, 10, (0.3, 0.6), 0.2).close()
    assert calls == [('send', 1), ('wait_for', 1200)] and 'cancel' not in calls
    robot, calls, _ = _fake_streams(done_after_hold=False)  # still running after the hold: cancel then
    BaseStream(robot, 10, (0.3, 0.6), 0.2).close()
    assert calls == [('send', 1), ('wait_for', 1200), 'cancel', ('wait_for', 1000)]


def test_an_expired_base_stream_is_renewed_a_bounded_number_of_times():
    from services.rby1.robot import BaseStream
    robot, calls, made = _fake_streams(expire_first=1)
    stream = BaseStream(robot, 10, (0.3, 0.6), 0.2)
    stream.send(0.05, 0.0, 0.05)
    assert len(made) == 2 and calls == [('send', 2)]
    stream.renewals = BaseStream.MAX_RENEWALS
    stream.stream.expire = 1
    with pytest.raises(RuntimeError, match='expired'):
        stream.send(0.05, 0.0, 0.05)


def test_solid_colour_camera_frames_are_recognised():
    from services.rby1.camera import unusable
    green = np.zeros((720, 1280, 3), np.uint8)
    green[..., 1] = 134                      # the ZED's frames after its USB recovery (RGB 0, 134, 1)
    green[..., 2] = 1
    noisy_green = green.copy()
    noisy_green[..., 1] = np.random.default_rng(0).integers(80, 200, (720, 1280))
    scene = np.random.default_rng(1).integers(0, 255, (720, 1280, 3)).astype(np.uint8)
    grey = np.full((720, 1280, 3), 120, np.uint8)
    dark = (np.random.default_rng(2).integers(0, 6, (720, 1280, 3))).astype(np.uint8)  # a dim but real scene
    assert unusable(green) and unusable(noisy_green) and unusable(grey)
    assert not unusable(scene) and not unusable(dark)


def test_a_cartesian_move_from_a_near_singular_arm_is_refused(make):
    """The controller went into MinorFault on a Cartesian move with the left elbow almost straight."""
    faulted = with_component(np.array(READY, dtype=np.float64), 'torso', [0, 0.7854, -1.5708, 0.8854, 0, 0])
    faulted = with_component(faulted, 'left_arm', [-2.0258, 0.4068, -2.0898, -0.3767, 2.487, 1.6201, 0.3427])
    backend = make(FakeRobot(faulted), controller_commands=True)
    arm(backend)
    down = {'component': 'left_arm', 'targets': [{'link': 'ee_left', 'position': [0, 0, -0.02], 'relative': True}]}
    with pytest.raises(ValueError, match='near a singular pose'):
        backend.validate('rby1_cartesian', down)
    value, _ = backend.call('rby1_check', {'tool': 'rby1_cartesian', 'arguments': down})
    assert value['accepted'] is False and 'singular' in value['reason']
    backend = make(FakeRobot(), controller_commands=True)  # READY: well conditioned
    arm(backend)
    backend.validate('rby1_cartesian', {'component': 'left_arm', 'targets': [
        {'link': 'ee_left', 'position': [0, 0, -0.02], 'relative': True}]})
