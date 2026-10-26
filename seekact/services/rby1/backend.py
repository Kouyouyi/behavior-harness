"""MCP backend for the real Rainbow Robotics RB-Y1 (model A) through rby1_sdk.

The robot's controller (192.168.30.1:50051, gRPC) runs the joint and wheel controllers;
this process runs on the robot's UPC (a Jetson) and talks to it with rby1_sdk 0.8.3
(`services.rby1.robot`, the only module that sends robot commands). Body: a
differential-drive base, a 6-joint torso, two 7-DoF arms with two-finger Dynamixel
grippers on the UPC's USB bus, and a 2-joint head carrying a ZED 2i stereo camera.

Safety layering, outermost first:
  1. The physical emergency stop and the operator beside the robot. The backend also
     reads the EMO state and refuses or stops motion while one is pressed.
  2. Operator arming (services.arming): motion tools refuse unless armed, and a running
     motion stops within one control period once the operator disarms.
  3. This backend: joint targets inside the URDF limits minus a margin, per-call joint and
     base displacement caps, slow joint and base speeds with gentle acceleration, an upright
     torso rule and a squat limit, end-effector height, self-collision (vendor
     collision model) and the gravity load on every joint (vendor dynamics) checked along
     the whole joint path, and a wrist-force guard while the arms move. Joint moves are
     timed position commands on the robot's controller, polled at control_hz; disarming,
     the EMO, stale state, a drive that is not ready, no progress, a force spike or a joint
     torque over 150 % of its rating cancels the motion and holds.
     A joint left short of its target (an end stop, a contact) is re-targeted to where it is:
     otherwise its drive winds up to its current limit and faults, and every joint drops
     (that happened on the robot). A watchdog does the same between calls.
     Connecting moves nothing and changes no robot state: power, servo and the control
     manager come up at the first armed motion, and the drives then hold where they are.
     A controller fault disarms the session and is reset only after the operator arms again.
  4. The robot controller's own limits and faults.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import threading
import time
import uuid

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from services import arming

from .controller_modes import CONTROLLER_DEFAULTS, CONTROLLER_TOOLS, ControllerModes
from .kinematics import (COMPONENTS, EE_FRAME, HEAD_CAMERA, JOINT_NAMES, SHOULDER_FRAME, Kinematics, component,
                         matrix, pose, with_component)

WORKSPACE_NAME = 'agent-workspace'
ARM_REACH_M = 0.69  # shoulder (link_*_arm_0) to end effector, max over 20 000 joint samples of the vendor URDF
JOINT_COMPONENTS = ('torso', 'right_arm', 'left_arm', 'head')
ARMS = ('right', 'left')
D2R = np.pi / 180
# Postures by component. `ready` is the vendor demo's (examples/python/09_demo_motion.py), `half` the
# midpoint from the rest pose (all zero, arms hanging) to it, `park` an arms-out-of-the-camera
# pose. The arms have no `zero`: at rest arm joints 1 and 3 sit on their URDF
# limits, inside the limit margin, so the rest pose is not a valid target.
# `mid`, `low`, `lowest` and the stretched arm postures were run on the robot (scripts/rby1_large_motion.py,
# 2026-10-08; hand positions and loads in agent_skills/control/rby1/whole-body-control). `forward_up`
# and `overhead` are more than 1.2 rad from `ready` for shoulder joint 0: go through `forward`.
_RIGHT_STRETCH = {'forward': [-1.57, -0.3, 0.0, -0.25, 0.0, 0.25, 0.0],
                  'forward_up': [-2.36, -0.3, 0.0, -0.25, 0.0, 0.25, 0.0],
                  'overhead': [-2.9, -0.3, 0.0, -0.35, 0.0, 0.3, 0.0],
                  'forward_out': [-1.2, -0.8, 0.0, -0.4, 0.0, 0.3, 0.0],
                  'side': [0.0, -1.05, 0.0, -0.25, 0.0, 0.0, 0.0]}
_MIRROR = np.array([1, -1, -1, 1, -1, 1, -1])  # right-arm angles -> the left arm's mirror image
PRESETS = {
    'torso': {'zero': [0.0] * 6, 'half': (np.array([0, 15, -30, 15, 0, 0]) * D2R).tolist(),
              'ready': (np.array([0, 30, -60, 30, 0, 0]) * D2R).tolist(),
              'mid': (np.array([0, 45, -90, 45, 0, 0]) * D2R).tolist(),
              'low': [0.0, 1.047, -2.094, 1.047, 0.0, 0.0],
              'lowest': [0.0, 1.2, -2.4, 1.2, 0.0, 0.0]},
    'right_arm': {'half': (np.array([-22.5, -15, 0, -45, 0, 22.5, 0]) * D2R).tolist(),
                  'ready': (np.array([-45, -30, 0, -90, 0, 45, 0]) * D2R).tolist(),
                  'park': [0.0, -0.2, -0.6, -1.5708, 0.0, 0.0, 0.0], **_RIGHT_STRETCH},
    'left_arm': {'half': (np.array([-22.5, 15, 0, -45, 0, 22.5, 0]) * D2R).tolist(),
                 'ready': (np.array([-45, 30, 0, -90, 0, 45, 0]) * D2R).tolist(),
                 'park': [0.0, 0.2, 0.6, -1.5708, 0.0, 0.0, 0.0],
                 **{name: (np.array(q) * _MIRROR + 0.0).tolist() for name, q in _RIGHT_STRETCH.items()}},
    'head': {'zero': [0.0, 0.0]},
}
DEFAULTS = {
    'address': '192.168.30.1:50051',     # the robot controller's gRPC endpoint
    'connect_retries': 5,
    'connect_timeout_ms': 2000,
    'urdf_path': None,                   # None: the robot's own model (Robot.get_robot_model())
    'unlimited_mode': False,             # control manager "unlimited mode" (undocumented by the vendor)
    'priority': 10,                      # command priority of every motion and hold
    'control_hz': 50.0,                  # how often a running motion re-checks arming, EMO, state and force
    'state_max_age_s': 0.2,              # a moving joint whose drive reported longer ago is stale
    'joint_speed_radps': 0.2,            # arms
    'torso_speed_radps': 0.1,
    'head_speed_radps': 0.3,
    'joint_accel_radps2': 0.4,           # arms: a 0.5 s ramp to full speed
    'torso_accel_radps2': 0.2,
    'head_accel_radps2': 0.8,
    'control_hold_time_s': 0.2,          # how long the controller keeps a command after its target (1.0 s
                                         # only added 0.8 s per move on the robot, same final error)
    'max_joint_step_rad': 1.2,           # per call, any joint (zero -> the vendor's torso `ready` is 1.05)
    'joint_limit_margin_rad': 0.05,      # head
    'arm_limit_margin_rad': 0.1,
    'torso_limit_margin_rad': 0.1,
    'torso_max_squat_rad': 1.2,          # torso_1 at most this and torso_2 at least -2x this: torso_2 met an end
                                         # stop at -2.554 rad on the robot, inside its URDF limit -2.618; 1.2
                                         # (torso_2 -2.4) was run clean on 2026-10-08, 0.15 rad from that stop
    'joint_tolerance_rad': 0.02,         # settled when every moved joint is this close
    'joint_exact_rad': 0.004,            # an arm or torso joint further than this from its target after settling
                                         # is blocked: it is re-targeted to where it is
    'exact_timeout_s': 1.0,
    'torque_rise_fraction': 0.1,         # a joint already over the guard when a motion starts stops it only after
                                         # rising this fraction of its rating further (so a load never locks control)
    'torque_guard_fraction': 1.5,        # a measured joint torque above this fraction of its rating stops motion at
                                         # once (raised from 1.0 on the operator's request, 2026-10-08: carrying
                                         # loads put the right wrist and elbow at 100-101 %; the fault was 260 %
                                         # held for 1.2 s). Joint limits are unchanged
    'static_torque_fraction': 1.3,       # the modelled gravity load (scaled) along a path stays below this (0.75,
                                         # then 1.125, then 1.3 on 2026-10-08 at the operator's request: the scaled
                                         # model read ~1.3x the measured torso_3 load when leaning, so 130 % is about
                                         # 100 % real, under the 150 % torque stop)
    'gravity_model_scale': 1.5,          # measured holding torques read 1.2-1.55 x the vendor model (elbows most)
    'max_motor_temp_c': 65.0,            # no motion while a drive is hotter
    'watchdog_hz': 10.0,                 # between calls: faults, drives not ready, torque near the rating (0: off)
    'watchdog_torque_s': 0.3,            # torque over the guard this long between calls: re-target in place
    'settle_timeout_s': 3.0,
    'motion_stall_s': 2.0,               # no progress towards the target for this long: cancel and hold
    'motion_progress_rad': 0.002,
    'torso_guard': True,                 # torso_0 = torso_4 = 0 and torso_2 = -2 torso_1 (upright torso)
    'max_eef_translation_m': 0.25,       # per rby1_move_eef call
    'ee_min_z_m': 0.10,                  # end effectors stay above this (base frame) along the path
    'self_collision_check': True,
    'ft_guard_n': 60.0,                  # wrist force rise over the motion's start that stops an arm motion (40 N
                                         # until 2026-10-08, raised with the torque guard)
    'base_hz': 20.0,
    'base_max_linear_mps': 0.15,
    'base_max_angular_radps': 0.3,
    'base_linear_accel': 0.3,
    'base_angular_accel': 0.6,
    'drive_max_squat_rad': 0.8,          # no driving with torso_1 above this (`mid` is 0.785)
    'drive_max_lean_rad': 0.05,          # no driving with the chest leaning (torso_3 - torso_1) more than this
    'drive_max_hand_reach_m': 0.72,      # no driving with a hand further forward or sideways than this
    'base_max_translation_m': 1.0,       # per rby1_base_move call
    'base_max_rotation_rad': 3.1416,
    'base_position_tolerance_m': 0.005,
    'base_angle_tolerance_rad': 0.01,
    'base_stall_s': 2.0,                 # odometry shows no motion while driving for this long: stop
    'base_hold_s': 0.2,                  # a velocity the controller keeps without a new send
    'camera': True,                      # the head ZED 2i (pyzed)
    'camera_serial': None,               # the head ZED's serial number; None: the first ZED found
    'head_camera_extrinsic': None,       # 4x4 link_head_2 -> optical frame from your hand-eye calibration;
                                         # None: kinematics.X_HEAD_CAMERA, measured on another unit
    'camera_resolution': 'HD720',
    'camera_fps': 15,
    'depth_mode': 'NEURAL',
    'camera_max_age_s': 1.0,             # an observation needs a frame at most this old
    'grippers': True,
    'gripper_current_a': 0.5,            # default grip current limit (the grip force)
    'gripper_max_current_a': 1.0,
    'gripper_settle_s': 1.5,
    'agent_image_width': 640,
    # Sliding-window recording of the head camera and robot state (read-only to the agent: runtime/stream).
    'stream_enabled': True,
    'stream_fps': 10.0,
    'stream_depth_hz': 1.0,
    'stream_window_s': 600.0,
    'stream_archive': True,              # whole-session MP4 per camera per minute (<session>/archive)
    'stream_jpeg_quality': 80,
    'robot_log_hz': 10.0,
    **CONTROLLER_DEFAULTS,  # the controller's own command modes (services/rby1/controller_modes.py)
}


def _schema(properties=None, required=()):
    return {'type': 'object', 'properties': properties or {}, 'required': list(required),
            'additionalProperties': False}


NUMBER = {'type': 'number'}
VEC3 = {'type': 'array', 'items': NUMBER, 'minItems': 3, 'maxItems': 3}
QUAT = {'type': 'array', 'items': NUMBER, 'minItems': 4, 'maxItems': 4,
        'description': 'End-effector orientation quaternion [x, y, z, w] in the base frame.'}


class MotionStopped(RuntimeError):
    def __init__(self, message, relieve=False):
        super().__init__(message)
        self.relieve = relieve  # the joints may be pushing on something: re-target them where they are


FAULTS = ('MinorFault', 'MajorFault')
BODY = ('torso', 'right_arm', 'left_arm')


def _wrap(angle):
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class Rby1Backend(ControllerModes):
    mutating = frozenset({'rby1_move_joints', 'rby1_move_body', 'rby1_move_eef', 'rby1_look_at',
                          'rby1_base_move', 'rby1_gripper', *CONTROLLER_TOOLS})

    def __init__(self, config, *, robot=None, kinematics=None, camera=None, gripper=None):
        self.config = config
        unknown = set(config.backend_options) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'unknown rby1 backend options: {sorted(unknown)}')
        self.options = {**DEFAULTS, **config.backend_options}
        if robot is None:
            from .robot import SdkRobot
            robot = SdkRobot(self.options['address'], retries=self.options['connect_retries'],
                             timeout_ms=self.options['connect_timeout_ms'])
        self.robot = robot
        if kinematics is None:
            path = self.options['urdf_path']
            camera = self.options['head_camera_extrinsic']
            kinematics = (Kinematics(urdf_path=path, head_camera=camera) if path
                          else Kinematics(urdf_text=self.robot.urdf(), head_camera=camera))
        self.kinematics = kinematics
        self.workspace = config.root/WORKSPACE_NAME
        self.frames_root = self.workspace/'runtime'/'frames'
        self.depth = config.observation_profile == 'rgbd'
        self.camera = camera
        self.camera_error = None
        if self.camera is None and self.options['camera']:
            try:
                from .camera import ZedCamera
                self.camera = ZedCamera(self.options['camera_serial'], resolution=self.options['camera_resolution'],
                                        fps=self.options['camera_fps'], depth=self.depth,
                                        depth_mode=self.options['depth_mode'])
            except Exception as exc:  # the session still runs; observe reports why there is no image
                self.camera_error = str(exc)[:300]
        self.gripper = gripper
        if self.gripper is None and self.options['grippers']:
            from .gripper import Gripper
            self.gripper = Gripper(self.robot)
        self.cameras = ('head_camera',) if self.camera is not None else ()
        self.enabled = False      # power, servo and control manager brought up (first armed motion)
        self.enable_steps = []
        self.base_stream = None   # the open base velocity stream while driving
        self.last_motion = None
        self.lock = threading.Lock()  # one motion at a time
        self.fault_disarmed_at = None  # when the backend disarmed for a fault / EMO not yet cleared by a re-arm
        self.fault = None         # the last controller fault or drive problem seen
        self.events = []          # safety events (faults, joints re-targeted in place), newest last
        self.over_since = None    # watchdog: when a joint torque first read over the guard
        self.torque_baseline = None  # joint torques when the current motion call started (_require_armed)
        self._watch_stop = threading.Event()
        self.stream = config.root/'stream' if self.options['stream_enabled'] else None
        self.recorder = None
        if self.stream is not None:  # record from the start of the session, not the first look
            from .recorder import Recorder
            self.recorder = Recorder({name: self.camera for name in self.cameras}, self._robot_record, self.stream,
                                     fps=self.options['stream_fps'], depth_hz=self.options['stream_depth_hz'],
                                     robot_hz=self.options['robot_log_hz'], window_s=self.options['stream_window_s'],
                                     jpeg_quality=self.options['stream_jpeg_quality'], depth=self.depth,
                                     archive=config.root/'archive' if self.options['stream_archive'] else None).start()
        self.watchdog = None
        if self.options['watchdog_hz'] > 0:
            self.watchdog = threading.Thread(target=self._watch, name='rby1-watchdog', daemon=True)
            self.watchdog.start()

    # ----- robot state --------------------------------------------------------------------
    def _limits(self, name):
        start, stop = COMPONENTS[name]
        margin = self.options[{'torso': 'torso_limit_margin_rad', 'head': 'joint_limit_margin_rad'}.get(
            name, 'arm_limit_margin_rad')]
        lower, upper = self.kinematics.lower[start:stop] + margin, self.kinematics.upper[start:stop] - margin
        if name == 'torso':  # the squat limit (torso_1 up, torso_2 down)
            squat = self.options['torso_max_squat_rad']
            upper[1] = min(upper[1], squat)
            lower[2] = max(lower[2], -2.0 * squat)
        return lower, upper

    def _snapshot(self):
        return self.robot.state()

    def _fresh(self, snap, components=JOINT_COMPONENTS):
        """Raise MotionStopped unless every joint of the components ('wheels' too) reported recently."""
        stale = []
        for name in components:
            start, stop = (0, 2) if name == 'wheels' else COMPONENTS[name]
            age = float(np.max(snap.age_s[start:stop]))
            if not np.isfinite(age) or age > self.options['state_max_age_s']:
                stale.append(f'{name} ({age:.2f} s)')
        if stale:
            raise MotionStopped(f'no fresh joint state from {", ".join(stale)}; not moving on stale data')

    # ----- tool definitions ---------------------------------------------------------------
    def tools(self):
        o = self.options
        cams = list(self.cameras) or ['head_camera']
        presets = {name: sorted(PRESETS[name]) for name in JOINT_COMPONENTS}
        positions = {'type': 'array', 'items': NUMBER, 'minItems': 2, 'maxItems': 7}
        JOINTS = {'type': 'object', 'additionalProperties': False,
                  'properties': {name: _schema({'positions': positions, 'preset': {'type': 'string'},
                                                'relative': {'type': 'boolean'}}) for name in JOINT_COMPONENTS}}
        tools = [
            {'name': 'rby1_status', 'inputSchema': _schema(),
             'description': 'Robot state without moving: joints of the torso, both arms and the head, end-effector '
                            'poses (base frame), wrist forces, wheel odometry, EMO, battery, power / servo / control '
                            'manager, gripper openings, whether the operator has armed motion, and limits.'},
            {'name': 'rby1_observe', 'description':
                'Capture the head camera (ZED 2i on the 2-joint head) now. Returns a downscaled image plus joint state; '
                'the full-resolution RGB (and depth as uint16 millimetre PNG in rgbd sessions) and a meta.json with '
                'the intrinsics K and the camera pose T_base_cam (base frame) are saved under '
                'runtime/frames/<frame_id>/. Use frame_id with rby1_pixel_to_position.',
             'inputSchema': _schema()},
            {'name': 'rby1_pixel_to_position', 'description':
                'Base-frame 3-D points for pixels of a saved head-camera frame, from its depth. Pixels are in the '
                'full-resolution image, not the preview. Uses the median valid depth in a (2*radius+1)^2 window.',
             'inputSchema': _schema({'frame_id': {'type': 'string'}, 'camera': {'type': 'string', 'enum': cams},
                                     'pixels': {'type': 'array', 'minItems': 1, 'maxItems': 64,
                                                'items': {'type': 'array', 'items': NUMBER, 'minItems': 2,
                                                          'maxItems': 2}},
                                     'radius': {'type': 'integer', 'minimum': 0, 'maximum': 15}},
                                    ['frame_id', 'camera', 'pixels'])},
            {'name': 'rby1_move_joints', 'description':
                "Joint-space move of one component on the robot's controller, and wait until it settles. component: "
                'torso (6 joints), right_arm / left_arm (7), head (2: pan, tilt). positions: absolute radians, or '
                f'preset ({json.dumps(presets)}). relative=true adds positions to the current joints. Speed caps: '
                f"arms {o['joint_speed_radps']} rad/s, torso {o['torso_speed_radps']} rad/s, head "
                f"{o['head_speed_radps']} rad/s. At most {o['max_joint_step_rad']} rad per joint per call; torso squat "
                f"at most torso_1 = {o['torso_max_squat_rad']} rad. The path is checked by FK for end-effector "
                'height, self-collision and the gravity load on every joint; a joint torque over 150 % of its rating stops '
                'the move, and a joint that stops short of its target (end stop, contact) is re-targeted where it '
                'is (result: blocked_rad, warning). Requires operator arming.',
             'inputSchema': _schema({'component': {'type': 'string', 'enum': list(JOINT_COMPONENTS)},
                                     'positions': positions, 'preset': {'type': 'string'},
                                     'relative': {'type': 'boolean'}}, ['component'])},
            {'name': 'rby1_move_body', 'description':
                'Whole-body joint move: several of torso, right_arm, left_arm and head in one command, timed to '
                'finish together (each never faster than its own cap; the slowest sets the pace). targets: '
                '{component: {positions | preset, relative?}} with the same meaning as rby1_move_joints. The path '
                'is checked for end-effector height and self-collision. The base is never moved here. Requires '
                'operator arming.',
             'inputSchema': _schema({'targets': {'type': 'object', 'minProperties': 1, 'additionalProperties': False,
                                                 'properties': {name: _schema({'positions': positions,
                                                                               'preset': {'type': 'string'},
                                                                               'relative': {'type': 'boolean'}})
                                                                for name in JOINT_COMPONENTS}}}, ['targets'])},
            {'name': 'rby1_move_eef', 'description':
                "Move one arm's end effector (ee_right / ee_left) to a base-frame pose: IK on that arm only (torso, "
                'head and the other arm hold), then a joint-space move (the path between is NOT a straight line; it '
                'is checked for height and self-collision). At most '
                f"{o['max_eef_translation_m']} m per call; target within {ARM_REACH_M} m of that shoulder. Omit "
                'quaternion to keep the current orientation; relative=true adds position to the current one. '
                'Requires operator arming. Returns the measured pose and error.',
             'inputSchema': _schema({'arm': {'type': 'string', 'enum': list(ARMS)}, 'position': VEC3,
                                     'quaternion': QUAT, 'relative': {'type': 'boolean'}}, ['arm', 'position'])},
            {'name': 'rby1_look_at', 'description':
                'Point the head camera at a base-frame point (metres) by moving the 2 head joints (pan, tilt). Use it '
                'before rby1_observe so the object is centred, where depth is best. Returns the pointing error. '
                'Requires operator arming.',
             'inputSchema': _schema({'point': VEC3}, ['point'])},
            {'name': 'rby1_base_move', 'description':
                'Drive the differential-drive base relative to where it is now: straight `forward` (metres, + '
                'forward, - backward), then turn in place `turn` (rad, counter-clockwise +). Closed loop on the '
                'wheel odometry (it slips: verify with rby1_observe). Speeds are capped at '
                f"{o['base_max_linear_mps']} m/s and {o['base_max_angular_radps']} rad/s; at most "
                f"{o['base_max_translation_m']} m per call. There is no sideways motion. Keep the arms in and the "
                'torso upright before driving. Requires operator arming.',
             'inputSchema': _schema({'forward': NUMBER, 'turn': NUMBER,
                                     'max_linear': {'type': 'number', 'minimum': 0.02,
                                                    'maximum': o['base_max_linear_mps']},
                                     'max_angular': {'type': 'number', 'minimum': 0.05,
                                                     'maximum': o['base_max_angular_radps']}})},
            {'name': 'rby1_stream', 'description':
                f"Look back in the continuous recording ({', '.join(cams)} at about {o['stream_fps']:g} Hz and the "
                f"robot state at {o['robot_log_hz']:g} Hz, last {o['stream_window_s'] / 60:g} minutes) without moving "
                'anything. seconds_ago: frames from that many seconds ago (0 = newest), labelled with their age, plus '
                'the robot state then. sheet: one contact sheet per camera of count evenly spaced frames over the '
                'last seconds. Use it to see what changed while you were thinking and to review a motion or script '
                'run. Files: runtime/stream/.',
             'inputSchema': _schema({'cameras': {'type': 'array', 'items': {'type': 'string', 'enum': cams}},
                                     'seconds_ago': {'type': 'array', 'items': {'type': 'number', 'minimum': 0},
                                                     'maxItems': 8},
                                     'sheet': _schema({'seconds': {'type': 'number', 'minimum': 1},
                                                       'count': {'type': 'integer', 'minimum': 2, 'maximum': 16}},
                                                      ['seconds'])})},
            {'name': 'rby1_fk', 'description':
                'Forward kinematics without moving: hand (ee_right / ee_left), head-camera and optional link poses '
                '(base frame) for the current joints, or for a hypothetical posture given in joints '
                '({component: {positions | preset, relative?}}, components not given stay as they are now). Also '
                'reports self-collisions, joints outside the backend limits, the gravity load per joint (fraction '
                'of rating, the same estimate the motion tools refuse above), each hand\'s distance from its '
                'shoulder and the whole-body centre of mass. No arming needed; nothing moves.',
             'inputSchema': _schema({'joints': JOINTS,
                                     'frames': {'type': 'array', 'maxItems': 20,
                                                'items': {'type': 'string', 'enum': self._frame_names()}}})},
            {'name': 'rby1_check', 'description':
                'Dry run of a motion tool (rby1_move_joints, rby1_move_body, rby1_move_eef, rby1_look_at, '
                'rby1_base_move) from the current state: runs every planning check the tool runs before moving '
                '(limits, squat cap, per-call caps, reach, IK, end-effector height, self-collision and gravity '
                'load along the path, driving posture) and returns accepted, the refusal reason, the joint goals, '
                'the predicted hand poses and the planned duration. It does not check arming, the EMO or faults '
                '(reported separately in ready_to_move). No arming needed; nothing moves.',
             'inputSchema': _schema({'tool': {'type': 'string', 'enum': ['rby1_move_joints', 'rby1_move_body',
                                                                         'rby1_move_eef', 'rby1_look_at',
                                                                         'rby1_base_move', *(
                                                                             CONTROLLER_TOOLS if
                                                                             o['controller_commands'] else ())]},
                                     'arguments': {'type': 'object'}}, ['tool', 'arguments'])},
            {'name': 'rby1_controller_info', 'description':
                "Read-only queries of the robot's controller: robot_info (model, firmware, SDK), control_manager "
                '(state, unlimited mode, time scale), pid_gains (name: torso / right_arm / left_arm / head), '
                'parameter_list, parameter (name), fault_logs (file names), log (count recent lines). Nothing here '
                'changes the robot; there is no tool that does.',
             'inputSchema': _schema({'what': {'type': 'string', 'enum': ['robot_info', 'control_manager', 'pid_gains',
                                                                         'parameter_list', 'parameter', 'fault_logs',
                                                                         'log']},
                                     'name': {'type': 'string'},
                                     'count': {'type': 'integer', 'minimum': 1, 'maximum': 200}}, ['what'])},
            {'name': 'rby1_hold', 'inputSchema': _schema({'mode': {'type': 'string', 'enum': ['cancel', 'stop_command']}}),
             'description': 'Stop the base and hold every joint (and gripper) at its current position. mode cancel '
                            "(default): cancel the running commands; the controller decelerates and holds. "
                            "stop_command: additionally send the controller's StopCommand. Always allowed."},
            *self._controller_tools(),
        ]
        if self.gripper is not None:
            tools.append({'name': 'rby1_gripper', 'description':
                          'Open, close or set the opening of one two-finger gripper (opening 0 to 0.1 m). close and '
                          'width stop where the fingers meet the object, squeezing with at most current_a amps '
                          f"(default {o['gripper_current_a']}, max {o['gripper_max_current_a']}): that is the grip "
                          'force. The first call of a session homes both grippers (each opens and closes fully '
                          'once, about 10 s). Returns the measured opening. Requires operator arming.',
                          'inputSchema': _schema({'side': {'type': 'string', 'enum': list(ARMS)},
                                                  'action': {'type': 'string', 'enum': ['open', 'close', 'width']},
                                                  'width_m': {'type': 'number', 'minimum': 0, 'maximum': 0.1},
                                                  'current_a': {'type': 'number', 'minimum': 0.05,
                                                                'maximum': o['gripper_max_current_a']}},
                                                 ['side', 'action'])})
        return tools

    # ----- validation (before any motion) -------------------------------------------------
    def _require_armed(self):
        """Arming, a released EMO and no unacknowledged fault, then power / servo / control manager up,
        cool drives and a fresh state."""
        state = arming.read(self.config.root)
        if not state['armed']:
            raise PermissionError(f"motion is not armed ({state['reason']}); the operator arms motion")
        snap = self._snapshot()
        if snap.emo_pressed:
            self._disarm('an emergency stop was pressed', fault='emo')
            raise PermissionError('an emergency stop is pressed; the operator releases it and arms again')
        manager = self.robot.manager_state()
        acknowledged = self.fault_disarmed_at is not None and (state.get('since') or 0) > self.fault_disarmed_at
        if manager in FAULTS and not acknowledged:
            self._disarm(f'the robot controller is in {manager}', fault=manager)
            raise PermissionError(f'the robot controller is in {manager} (a drive tripped; joints may have dropped). '
                                  'The session is disarmed: report it with request_operator and wait; the operator '
                                  'inspects the robot and arms again, and only then is the fault reset')
        self._enable(reset_faults=manager in FAULTS)
        self.fault_disarmed_at = None  # healthy and armed: the next fault needs its own disarm and re-arm
        snap = self._snapshot()
        self.torque_baseline = None if snap.torque is None else snap.torque.copy()
        try:
            self._fresh(snap)
        except MotionStopped as exc:
            raise PermissionError(str(exc)) from None
        if snap.temperature is not None:
            hot = {n: t for n, t in zip(JOINT_NAMES, snap.temperature) if t > self.options['max_motor_temp_c']}
            if hot:
                raise PermissionError(f"drives hotter than {self.options['max_motor_temp_c']:g} C: {hot}; let them cool "
                                      '(hold still) and tell the operator')
        return snap

    def _enable(self, reset_faults=False):
        """Power, servo and control manager on (the vendor's order), checked before every armed motion.

        Idempotent: it acts only on what is off, so after an emergency stop (which drops power
        and faults the control manager) the next motion the operator arms brings them back. A
        fault is reset only when the operator armed after the backend disarmed for it.
        Moves nothing: the drives then hold where they are.
        """
        steps = self.robot.enable(self.options['unlimited_mode'], reset_faults=reset_faults)
        if steps or not self.enabled:
            self.enable_steps = steps
        if any(step.startswith('reset_fault') for step in steps):
            self._event('fault reset after the operator armed again', steps=steps)
        self.enabled = True

    def _event(self, what, **details):
        self.events = (self.events + [{'at': round(time.time(), 2), 'what': what, **details}])[-20:]

    def _disarm(self, reason, fault=None):
        """The backend disarms the session (EMO, controller fault, drive not ready): the operator re-arms on purpose."""
        now = time.time()
        if fault is not None:
            self.fault = {'at': round(now, 2), 'what': reason}
            self.fault_disarmed_at = now
        self._event('disarmed: ' + reason)
        if arming.read(self.config.root)['armed']:
            arming.write(self.config.root, False, by='rby1-backend', reason=reason)

    def _joint_goal(self, q, name, positions=None, preset=None, relative=False):
        if name not in JOINT_COMPONENTS:
            raise ValueError(f'unknown component {name}')
        if (positions is None) == (preset is None):
            raise ValueError('give exactly one of positions or preset')
        current = component(q, name)
        if preset is not None:
            if relative:
                raise ValueError('relative applies to positions only')
            if preset not in PRESETS[name]:
                raise ValueError(f'unknown {name} preset {preset!r}; choose from {sorted(PRESETS[name])}')
            goal = np.asarray(PRESETS[name][preset], dtype=np.float64)
        else:
            goal = np.asarray(positions, dtype=np.float64)
            if goal.shape != current.shape:
                raise ValueError(f'{name} has {current.size} joints, got {goal.size} positions')
            if relative:
                goal = current + goal
        lower, upper = self._limits(name)
        if not np.all(np.isfinite(goal)) or np.any(goal < lower) or np.any(goal > upper):
            squat = (f" and the squat limit (torso_1 <= {self.options['torso_max_squat_rad']}, torso_2 >= "
                     f"{-2 * self.options['torso_max_squat_rad']})" if name == 'torso' else '')
            raise ValueError(f'{name} target outside the joint limits minus a margin{squat}: lower '
                             f'{np.round(lower, 3).tolist()}, upper {np.round(upper, 3).tolist()}')
        if name == 'torso' and self.options['torso_guard']:
            if abs(goal[0]) > 0.01 or abs(goal[4]) > 0.01 or abs(goal[2] + 2 * goal[1]) > 0.02:
                raise ValueError('torso target breaks the upright rule (torso_0 = torso_4 = 0 and torso_2 = '
                                 f'-2 x torso_1; got {np.round(goal, 3).tolist()}): use the presets or keep that '
                                 'relation')
        return goal

    def _check_path(self, q, goals):
        """Per-call step cap, then end-effector height and self-collision along the joint paths moved together."""
        for name, goal in goals.items():
            step = float(np.max(np.abs(goal - component(q, name))))
            if step > self.options['max_joint_step_rad']:
                raise ValueError(f"{name} move of {step:.2f} rad exceeds {self.options['max_joint_step_rad']} rad "
                                 'per call; split it')
        target = np.array(q, dtype=np.float64)
        for name, goal in goals.items():
            target = with_component(target, name, goal)
        samples = [q + fraction * (target - q) for fraction in np.linspace(0.0, 1.0, 21)]
        if {'torso', 'right_arm', 'left_arm'} & set(goals):
            for k, sample in enumerate(samples):
                frames = self.kinematics.frames(list(EE_FRAME.values()), sample)
                for side, frame in EE_FRAME.items():
                    z = float(frames[frame][2, 3])
                    if z < self.options['ee_min_z_m']:
                        raise ValueError(f'path brings the {side} end effector to z={z:.3f} m at {k / 20:.0%} '
                                         f"(minimum {self.options['ee_min_z_m']} m)")
        limit = self.kinematics.torque_limit
        if {'torso', 'right_arm', 'left_arm'} & set(goals) and np.any(np.isfinite(limit)):
            for k, sample in enumerate(samples[::2]):
                load = np.abs(self.kinematics.gravity(sample)) * self.options['gravity_model_scale'] / limit
                j = int(np.argmax(load))
                if load[j] > self.options['static_torque_fraction']:
                    raise ValueError(f'the posture at {2 * k / 20:.0%} of the path loads {JOINT_NAMES[j]} to about '
                                     f'{load[j]:.0%} of its rated {limit[j]:g} N m just holding against gravity (limit '
                                     f"{self.options['static_torque_fraction']:.0%}); keep the arms closer to the body "
                                     'or the torso more upright')
        if self.options['self_collision_check']:
            hit = self.kinematics.first_new_collision(samples)
            if hit is not None:
                raise ValueError(f'self-collision {hit[1]} at {hit[0] / 20:.0%} of the path; choose another target '
                                 'or an intermediate posture')

    def _eef_goal(self, q, arm, position, quaternion=None, relative=False):
        T_now = self.kinematics.frame(EE_FRAME[arm], q)
        p_now, q_now = pose(T_now)
        target = np.asarray(position, dtype=np.float64)
        if not np.all(np.isfinite(target)):
            raise ValueError('position must be finite')
        if relative:
            target = p_now + target
        quaternion = np.asarray(q_now if quaternion is None else quaternion, dtype=np.float64)
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm < 0.5:
            raise ValueError('quaternion must be a unit quaternion [x, y, z, w]')
        quaternion = quaternion / norm
        shoulder = self.kinematics.frame(SHOULDER_FRAME[arm], q)[:3, 3]
        if np.linalg.norm(target - shoulder) > ARM_REACH_M:
            raise ValueError(f'out of reach: the target is {np.linalg.norm(target - shoulder):.2f} m from the {arm} '
                             f'shoulder (reach {ARM_REACH_M} m); drive closer or change the torso')
        step = float(np.linalg.norm(target - p_now))
        if step > self.options['max_eef_translation_m']:
            raise ValueError(f"move of {step:.3f} m exceeds {self.options['max_eef_translation_m']} m per call; "
                             'split it into shorter moves')
        if target[2] < self.options['ee_min_z_m']:
            raise ValueError(f"target z {target[2]:.3f} m is below {self.options['ee_min_z_m']} m")
        lower, upper = self._limits(f'{arm}_arm')
        solution, position_error, orientation_error = self.kinematics.solve_arm(
            arm, matrix(target, quaternion), q, lower=lower, upper=upper)
        if position_error > 0.01 or orientation_error > 0.05:
            raise ValueError(f'unreachable for this arm with the torso where it is (IK residual '
                             f'{position_error * 1000:.0f} mm, {np.degrees(orientation_error):.1f} deg); move the base '
                             'or torso, or change the orientation')
        return solution, target, quaternion

    def _look_at_goal(self, q, point):
        """Head pan / tilt that point the head camera's optical axis at a base-frame point."""
        from scipy.optimize import least_squares
        target = np.asarray(point, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError('point must be three finite numbers (base frame, metres)')
        q0 = component(q, 'head')
        lower, upper = self._limits('head')

        def camera(head):
            return self.kinematics.frame(HEAD_CAMERA, with_component(q, 'head', head))

        def residual(head):
            T = camera(head)
            ray = target - T[:3, 3]
            return np.concatenate([T[:3, 2] - ray / np.linalg.norm(ray), 0.02 * (head - q0)])
        if np.linalg.norm(target - camera(q0)[:3, 3]) < 0.25:
            raise ValueError('point is too close to the head camera (< 0.25 m)')
        solution = least_squares(residual, np.clip(q0, lower, upper), bounds=(lower, upper)).x
        T = camera(solution)
        ray = (target - T[:3, 3]) / np.linalg.norm(target - T[:3, 3])
        error = float(np.degrees(np.arccos(np.clip(T[:3, 2] @ ray, -1, 1))))
        if error > 5.0:
            raise ValueError(f'the head cannot point at {np.round(target, 3).tolist()} (best {error:.1f} deg off; '
                             'pan within +-0.47 rad, tilt -0.3..1.52 rad): turn the base or change the torso')
        return solution, error

    def _drive_posture(self, q):
        """Refuse driving with the body low, leaning or stretched out (the base can tip or sweep things)."""
        o = self.options
        t = component(q, 'torso')
        if t[1] > o['drive_max_squat_rad'] or abs(t[3] - t[1]) > o['drive_max_lean_rad']:
            raise ValueError(f"no driving with the torso low or leaning (torso_1 {t[1]:.2f} rad, max "
                             f"{o['drive_max_squat_rad']}; lean {t[3] - t[1]:+.2f} rad, max {o['drive_max_lean_rad']}): "
                             'move the torso to `ready` (or `mid`) first')
        frames = self.kinematics.frames(list(EE_FRAME.values()), q)
        for side, frame in EE_FRAME.items():
            x, y = frames[frame][:2, 3]
            if x > o['drive_max_hand_reach_m'] or abs(y) > o['drive_max_hand_reach_m']:
                raise ValueError(f'no driving with the {side} hand stretched out ({x:.2f}, {y:.2f}) m: bring the arms '
                                 'to `ready` or `park` first')

    def _base_plan(self, forward=0.0, turn=0.0, max_linear=None, max_angular=None):
        if not np.all(np.isfinite([forward, turn])):
            raise ValueError('forward and turn must be finite')
        if abs(forward) > self.options['base_max_translation_m']:
            raise ValueError(f"translation of {abs(forward):.2f} m exceeds {self.options['base_max_translation_m']} m "
                             'per call; split it')
        if abs(turn) > self.options['base_max_rotation_rad']:
            raise ValueError(f"rotation of {abs(turn):.2f} rad exceeds {self.options['base_max_rotation_rad']} rad")
        linear = min(float(max_linear or self.options['base_max_linear_mps']), self.options['base_max_linear_mps'])
        angular = min(float(max_angular or self.options['base_max_angular_radps']),
                      self.options['base_max_angular_radps'])
        return float(forward), float(turn), linear, angular

    def _gripper_args(self, side, action, width_m=None, current_a=None):
        if action == 'width' and width_m is None:
            raise ValueError('action width needs width_m')
        if action != 'width' and width_m is not None:
            raise ValueError('width_m applies to action width only')
        current = float(self.options['gripper_current_a'] if current_a is None else current_a)
        if not 0 < current <= self.options['gripper_max_current_a']:
            raise ValueError(f"current_a must be in (0, {self.options['gripper_max_current_a']}]")
        width = {'open': 0.1, 'close': 0.0}.get(action, width_m)
        return side, float(width), current

    def _plan(self, name, arguments, q):
        """The joint goals a motion tool would command from q, after every planning check the tool makes
        before moving ({} for the base). ValueError names the refusal."""
        if name == 'rby1_move_joints':
            goals = {arguments['component']: self._joint_goal(q, arguments['component'], arguments.get('positions'),
                                                              arguments.get('preset'), arguments.get('relative', False))}
        elif name == 'rby1_move_body':
            goals = self._body_goals(q, arguments['targets'])
        elif name == 'rby1_move_eef' and self.options['controller_commands']:
            self._controller_plan('rby1_cartesian', self._eef_as_cartesian(**arguments), q)
            return {}  # the controller solves the IK: targets checked, joints watched while it runs
        elif name == 'rby1_move_eef':
            solution, _, _ = self._eef_goal(q, **arguments)
            goals = {f"{arguments['arm']}_arm": solution}
        elif name in CONTROLLER_TOOLS:
            planned = self._controller_plan(name, arguments, q)[4]
            return planned if isinstance(planned, dict) else {}
        elif name == 'rby1_look_at':
            goals = {'head': self._look_at_goal(q, arguments['point'])[0]}
        elif name == 'rby1_base_move':
            self._base_plan(**arguments)
            self._drive_posture(q)
            return {}
        else:
            raise ValueError(f'{name} is not a motion tool')
        self._check_path(q, goals)
        return goals

    def validate(self, name, arguments):
        self._check_arguments(name, arguments)
        if name in self.mutating and name != 'rby1_gripper':
            self._plan(name, arguments, self._require_armed().q)
        elif name == 'rby1_gripper':
            self._require_armed()
            self._gripper_args(**arguments)
        elif name == 'rby1_pixel_to_position':
            self._frame_dir(arguments['frame_id'])

    # ----- calls --------------------------------------------------------------------------
    def _check_arguments(self, name, arguments):
        """Refuse argument keys a tool does not define (top level and nested objects), instead of ignoring them."""
        tools = {tool['name']: tool for tool in self.tools()}
        if name not in tools:
            raise ValueError(f'unknown tool {name}')

        def walk(schema, value, where):
            if isinstance(value, dict) and schema.get('type') == 'object' and schema.get('additionalProperties') is False:
                unknown = set(value) - set(schema.get('properties', {}))
                if unknown:
                    raise ValueError(f'{name}: unknown argument(s) {sorted(unknown)} in {where}')
                for key, sub in value.items():
                    walk(schema['properties'][key], sub, f'{where}.{key}')
            elif isinstance(value, list) and isinstance(schema.get('items'), dict):
                for k, sub in enumerate(value):
                    walk(schema['items'], sub, f'{where}[{k}]')
        walk(tools[name]['inputSchema'], arguments, 'arguments')

    def call(self, name, arguments):
        self._check_arguments(name, arguments)
        if name in CONTROLLER_TOOLS:
            return self._controller_call(name, arguments)
        return getattr(self, '_' + name.removeprefix('rby1_'))(**arguments)

    def _status(self):
        return self.status(), []

    def _guard(self, components=JOINT_COMPONENTS, ft_start=None):
        """Every control period of a motion: arming, EMO, fresh joint state, wrist force; the snapshot."""
        if not arming.read(self.config.root)['armed']:
            raise MotionStopped('the operator disarmed motion; the robot holds')
        snap = self._snapshot()
        if snap.emo_pressed:
            self._disarm('an emergency stop was pressed', fault='emo')
            raise MotionStopped('an emergency stop was pressed; the session is disarmed')
        self._fresh(snap, components)
        if not np.all(snap.ready):
            names = [n for n, r in zip(JOINT_NAMES, snap.ready) if not r]
            self._disarm(f'drives not ready: {names}', fault='drive')
            raise MotionStopped(f'drives not ready: {names} (a fault; joints may drop); the session is disarmed')
        over = self._over_torque(snap, self.torque_baseline)
        if over:
            raise MotionStopped(f"joint torque over {self.options['torque_guard_fraction']:.0%} of its rating: {over}; "
                                'motion cancelled (pushing on something or an overload)', relieve=True)
        if ft_start is not None and self.options['ft_guard_n']:
            for side, start in ft_start.items():
                rise = float(np.linalg.norm(np.subtract(snap.ft[side]['force'], start)))
                if rise > self.options['ft_guard_n']:
                    raise MotionStopped(f'{side} wrist force rose {rise:.0f} N (guard {self.options["ft_guard_n"]:g} '
                                        'N): contact; motion cancelled')
        return snap

    def _over_torque(self, snap, baseline=None):
        """{joint: 'tau N m (fraction of rated)'} of joints whose measured torque is over the guard.

        With a baseline (the torques when the motion started), a joint that was already over the guard
        then counts only once it rises a further torque_rise_fraction of its rating: a motion that holds
        or eases a load (backing off, setting an object down) is never blocked by that load.
        """
        if snap.torque is None:
            return {}
        limit = self.kinematics.torque_limit
        fraction = np.abs(snap.torque) / limit
        guard = self.options['torque_guard_fraction']
        over = fraction > guard
        if baseline is not None:
            start = np.abs(baseline) / limit
            over &= ~((start > guard) & (fraction <= start + self.options['torque_rise_fraction']))
        return {JOINT_NAMES[j]: f'{snap.torque[j]:+.0f} N m ({fraction[j]:.0%} of {limit[j]:g})'
                for j in np.flatnonzero(over)}

    def _relieve(self, why):
        """Re-target the torso and arms to where they are now: a zero-length move.

        A joint held short of its target (an end stop, a contact) keeps pushing until its drive
        faults; a target at the measured position ends the push. Waits for the cancel of any
        running command and for the joints to slow first, so the new target is where they stop.
        """
        snap, after = self._retarget_here()
        self._event('re-targeted torso and arms in place', why=why,
                    moved_rad=round(float(np.max(np.abs(after.q[2:22] - snap.q[2:22]))), 4),
                    torque_after=self._over_torque(after) or 'within the guard')
        return after

    def _retarget_here(self):
        """Cancel running commands, let the joints slow, then a joint-position target at the measured pose.

        Returns (snapshot before, snapshot after); the move is zero-length by construction.
        """
        self.robot.stop()
        end = time.monotonic() + 1.0
        snap = self._snapshot()
        while snap.velocity is not None and np.max(np.abs(snap.velocity[2:22])) > 0.01 and time.monotonic() < end:
            time.sleep(0.05)
            snap = self._snapshot()
        goals = {c: component(snap.q, c) for c in BODY}
        o = self.options
        handle = self.robot.send_joints(goals, minimum_time=0.5, velocity={c: self._speed(c) for c in goals},
                                        acceleration={c: self._accel(c) for c in goals},
                                        hold_s=o['control_hold_time_s'], priority=o['priority'])
        handle.wait(3000)
        return snap, self._snapshot()

    def _hold_all(self):
        errors = {}
        if self.base_stream is not None:
            try:
                self.base_stream.close()
            except Exception as exc:
                errors['base'] = str(exc)[:200]
            self.base_stream = None
        try:  # the controller decelerates and holds the joints where they stop
            self.robot.stop()
        except Exception as exc:
            errors['joints'] = str(exc)[:200]
        if self.gripper is not None:
            try:
                self.gripper.hold()
            except Exception as exc:
                errors['gripper'] = str(exc)[:200]
        return errors

    def _hold(self, mode='cancel'):
        errors = self._hold_all()  # not behind the motion lock: a hold must never wait for a motion
        if mode == 'stop_command':
            try:
                self.robot.controller_command('stop', None, {}, hold_s=self.options['control_hold_time_s'],
                                              priority=self.options['priority']).wait(2000)
            except Exception as exc:
                errors['stop_command'] = str(exc)[:200]
        return {'held': not errors, **({'errors': errors} if errors else {}), **self._state_report()}, []

    def _held(self, exc):
        """Mark a handled stop: everything holds at a measured state, nothing else is pending."""
        errors = self._hold_all()
        if getattr(exc, 'relieve', False) and not errors:
            try:
                self._relieve(str(exc))
            except Exception as relief:
                errors['relieve'] = str(relief)[:200]
        exc.handled_stop = not errors  # a backend stop with its complete reason (no traceback needed)
        exc.robot_state = self._state_report()
        if errors:  # the hold itself failed: say so with the state
            exc.robot_state['hold_errors'] = errors
        return exc

    def _failed(self, exc):
        """An unexpected exception during a motion (an SDK or driver error): hold, attach the measured
        state, and let the original exception through unchanged so the agent sees its type, message
        and traceback."""
        errors = self._hold_all()
        try:
            exc.robot_state = self._state_report()
            if errors:
                exc.robot_state['hold_errors'] = errors
        except Exception as state_error:  # the state could not be read: say that instead
            exc.robot_state = {'state_unavailable': str(state_error)[:300], **({'hold_errors': errors} if errors else {})}
        return exc

    def _speed(self, name):
        return self.options[{'torso': 'torso_speed_radps', 'head': 'head_speed_radps'}.get(name, 'joint_speed_radps')]

    def _accel(self, name):
        return self.options[{'torso': 'torso_accel_radps2', 'head': 'head_accel_radps2'}.get(name,
                                                                                             'joint_accel_radps2')]

    def _duration(self, step, name):
        """Time of a trapezoidal (or triangular) velocity profile over step at that component's caps."""
        v, a = self._speed(name), self._accel(name)
        return step / v + v / a if step >= v * v / a else 2.0 * np.sqrt(step / a)

    def _run(self, q, goals):
        """Joint targets for one or more components as one timed command on the robot's controller.

        Every control period re-checks arming, the EMO, state freshness and (for arm and torso
        moves) the wrist forces; a component that makes no progress, a deadline, or a
        controller finish code other than Ok cancels the command and holds.
        """
        o = self.options
        steps = {c: float(np.max(np.abs(g - component(q, c)))) for c, g in goals.items()}
        duration = max([self._duration(steps[c], c) for c in goals] + [0.5])
        deadline_s = 2.0 * duration + o['settle_timeout_s'] + 2.0
        period = 1.0 / o['control_hz']
        moving = tuple(goals)
        arms = {'torso', 'right_arm', 'left_arm'} & set(goals)
        started = time.monotonic()

        def remaining(snap, c):
            return float(np.max(np.abs(component(snap.q, c) - goals[c])))
        try:
            snap = self._guard(moving)
            ft_start = {side: snap.ft[side]['force'] for side in ARMS} if arms else None
            handle = self.robot.send_joints(goals, minimum_time=duration,
                                            velocity={c: self._speed(c) for c in goals},
                                            acceleration={c: self._accel(c) for c in goals},
                                            hold_s=o['control_hold_time_s'], priority=o['priority'])
            best = {c: remaining(snap, c) for c in goals}
            progressed = {c: time.monotonic() for c in goals}
            while not handle.is_done():
                snap = self._guard(moving, ft_start)
                now = time.monotonic()
                for c in goals:
                    left = remaining(snap, c)
                    if left < best[c] - o['motion_progress_rad']:
                        best[c], progressed[c] = left, now
                    elif left > o['joint_tolerance_rad'] and now - progressed[c] > o['motion_stall_s']:
                        raise MotionStopped(f'{c} made no progress for {o["motion_stall_s"]:g} s ({left:.3f} rad from '
                                            'the target); motion cancelled')
                if now - started > deadline_s:
                    raise MotionStopped(f'motion did not finish within {deadline_s:.0f} s; cancelled')
                time.sleep(period)
            finish = handle.finish()
            if finish not in ('Ok', 'Unknown'):
                raise MotionStopped(f"the robot's controller ended the motion with {finish}")
            settle = time.monotonic() + o['settle_timeout_s']
            while True:
                snap = self._guard(moving, ft_start)
                errors = {c: remaining(snap, c) for c in goals}
                if max(errors.values()) <= o['joint_tolerance_rad']:
                    break
                if time.monotonic() > settle:
                    worst = max(errors, key=errors.get)
                    raise MotionStopped(f'{worst} did not reach the target (largest joint error {errors[worst]:.3f} '
                                        'rad after the motion finished); re-targeted where it is', relieve=True)
                time.sleep(period)
            blocked = self._blocked(goals, moving, ft_start, period)
        except MotionStopped as exc:
            raise self._held(exc)
        except Exception as exc:
            raise self._failed(exc)
        result = {'seconds': round(time.monotonic() - started, 2), 'planned_s': round(duration, 2),
                  'max_joint_error_rad': round(max(errors.values()), 4), 'settled': True, 'finish': finish}
        if blocked:
            self._relieve(f'stopped short of the target: {blocked}')
            result.update(settled=False, blocked_rad=blocked,
                          warning='these joints stopped short of the target (an end stop or a contact) and were '
                                  're-targeted where they are; do not retry the same target: back off or check '
                                  'with rby1_observe')
        return result

    def _blocked(self, goals, moving, ft_start, period):
        """{joint: shortfall rad} of torso / arm joints still off their target after exact_timeout_s."""
        body = [c for c in goals if c in BODY]
        end = time.monotonic() + self.options['exact_timeout_s']
        while body:
            snap = self._guard(moving, ft_start)
            short = {}
            for c in body:
                start = COMPONENTS[c][0]
                for k, (now, goal) in enumerate(zip(component(snap.q, c), goals[c])):
                    if abs(now - goal) > self.options['joint_exact_rad']:
                        short[JOINT_NAMES[start + k]] = round(float(goal - now), 4)
            if not short or time.monotonic() > end:
                return short
            time.sleep(period)
        return {}

    def _body_goals(self, q, targets):
        if not targets:
            raise ValueError('give at least one component in targets')
        unknown = set(targets) - set(JOINT_COMPONENTS)
        if unknown:
            raise ValueError(f'unknown components {sorted(unknown)}; choose from {list(JOINT_COMPONENTS)}')
        return {c: self._joint_goal(q, c, t.get('positions'), t.get('preset'), t.get('relative', False))
                for c, t in targets.items()}

    def _move_joints(self, component, positions=None, preset=None, relative=False):
        with self.lock:
            snap = self._require_armed()
            goal = self._joint_goal(snap.q, component, positions, preset, relative)
            self._check_path(snap.q, {component: goal})
            result = self._run(snap.q, {component: goal})
            result.update(component=component, target=np.round(goal, 4).tolist(), **self._state_report())
            self.last_motion = {'tool': 'rby1_move_joints', 'at': time.time(), 'component': component}
            return result, []

    def _move_body(self, targets):
        with self.lock:
            snap = self._require_armed()
            goals = self._body_goals(snap.q, targets)
            self._check_path(snap.q, goals)
            result = self._run(snap.q, goals)
            result.update(components=sorted(goals), targets={c: np.round(g, 4).tolist() for c, g in goals.items()},
                          **self._state_report())
            self.last_motion = {'tool': 'rby1_move_body', 'at': time.time(), 'components': sorted(goals)}
            return result, []

    def _eef_as_cartesian(self, arm, position, quaternion=None, relative=False):
        target = {'link': EE_FRAME[arm], 'position': position, 'relative': relative}
        if quaternion is not None:
            target['quaternion'] = quaternion
        return {'component': f'{arm}_arm', 'targets': [target]}

    def _controller_info(self, what, name=None, count=50):
        if what in ('pid_gains', 'parameter') and not name:
            raise ValueError(f'{what} needs name')
        if what == 'pid_gains' and name not in ('torso', 'right_arm', 'left_arm', 'head'):
            raise ValueError('pid_gains name: torso, right_arm, left_arm or head')
        return {'what': what, 'value': self.robot.controller_info(what, name, count)}, []

    def _move_eef(self, arm, position, quaternion=None, relative=False):
        if self.options['controller_commands']:  # the controller's Cartesian command solves the IK
            value, images = self._controller_call('rby1_cartesian',
                                                  self._eef_as_cartesian(arm, position, quaternion, relative))
            value.update(arm=arm, position_error_mm=value['targets'][0]['position_error_mm'],
                         target_position=value['targets'][0]['target_position'])
            self.last_motion = {'tool': 'rby1_move_eef', 'at': time.time(), 'arm': arm, 'mode': 'cartesian'}
            return value, images
        with self.lock:
            snap = self._require_armed()
            solution, target_p, target_q = self._eef_goal(snap.q, arm, position, quaternion, relative)
            self._check_path(snap.q, {f'{arm}_arm': solution})
            result = self._run(snap.q, {f'{arm}_arm': solution})
            p, _ = pose(self.kinematics.frame(EE_FRAME[arm], self._snapshot().q))
            result.update(arm=arm, target_position=np.round(target_p, 4).tolist(),
                          target_quaternion=np.round(target_q, 4).tolist(),
                          position_error_mm=round(float(np.linalg.norm(p - target_p)) * 1000, 1),
                          **self._state_report())
            self.last_motion = {'tool': 'rby1_move_eef', 'at': time.time(), 'arm': arm}
            return result, []

    def _look_at(self, point):
        with self.lock:
            snap = self._require_armed()
            goal, error = self._look_at_goal(snap.q, point)
            self._check_path(snap.q, {'head': goal})
            result = self._run(snap.q, {'head': goal})
            result.update(point=list(point), predicted_error_deg=round(error, 2), target=np.round(goal, 4).tolist(),
                          **self._state_report())
            self.last_motion = {'tool': 'rby1_look_at', 'at': time.time()}
            return result, []

    def _base_move(self, forward=0.0, turn=0.0, max_linear=None, max_angular=None):
        o = self.options
        with self.lock:
            forward, turn, linear, angular = self._base_plan(forward, turn, max_linear, max_angular)
            snap = self._require_armed()
            self._drive_posture(snap.q)
            x0, y0, yaw0 = snap.odometry
            period = 1.0 / o['base_hz']
            segments = []
            try:
                self.base_stream = self.robot.base_stream(priority=o['priority'], hold_s=o['base_hold_s'],
                                                          accel=(o['base_linear_accel'], o['base_angular_accel']))
                if abs(forward) > o['base_position_tolerance_m']:
                    segments.append(self._drive(forward, linear, (x0, y0, yaw0), period))
                if abs(turn) > o['base_angle_tolerance_rad']:
                    segments.append(self._turn(turn, angular, period))
                self.base_stream.close()
                self.base_stream = None
            except MotionStopped as exc:
                raise self._held(exc)
            except Exception as exc:
                raise self._failed(exc)
            x, y, yaw = self._snapshot().odometry
            dx, dy = x - x0, y - y0
            measured = {'forward': round(dx * np.cos(yaw0) + dy * np.sin(yaw0), 4),
                        'sideways': round(-dx * np.sin(yaw0) + dy * np.cos(yaw0), 4), 'turn': round(_wrap(yaw - yaw0), 4)}
            self.last_motion = {'tool': 'rby1_base_move', 'at': time.time(), 'commanded': [forward, turn]}
            return {'commanded': {'forward': forward, 'turn': turn}, 'measured_by_odometry': measured,
                    'segments': segments, 'odometry': 'wheel (slips; positions measured before the move are stale)',
                    **self._state_report()}, []

    def _drive(self, distance, cap, start, period):
        o = self.options
        x0, y0, yaw0 = start
        deadline = time.monotonic() + 3.0 * abs(distance) / cap + 5.0
        t0 = last_moved = time.monotonic()
        last = None
        while True:
            snap = self._guard(('wheels',))
            x, y, yaw = snap.odometry
            done = (x - x0) * np.cos(yaw0) + (y - y0) * np.sin(yaw0)
            left = distance - done
            if abs(left) <= o['base_position_tolerance_m']:
                break
            now = time.monotonic()
            if last is None or np.hypot(x - last[0], y - last[1]) > 0.002:
                last, last_moved = (x, y), now
            elif now - last_moved > o['base_stall_s']:
                raise MotionStopped(f'the base did not move for {o["base_stall_s"]:g} s ({left:+.3f} m left); stopped')
            if now > deadline:
                raise MotionStopped(f'the drive did not finish in time ({left:+.3f} m left); stopped')
            v = float(np.sign(left) * np.clip(abs(left) * 1.5, 0.03, cap))
            wz = float(np.clip(1.0 * _wrap(yaw0 - yaw), -0.2, 0.2))  # hold the start heading
            self.base_stream.send(v, wz, period)
            time.sleep(period)
        self.base_stream.send(0.0, 0.0, period)
        return {'segment': 'forward', 'target_m': distance, 'seconds': round(time.monotonic() - t0, 2)}

    def _turn(self, angle, cap, period):
        o = self.options
        yaw_goal = self._snapshot().odometry[2] + angle
        deadline = time.monotonic() + 3.0 * abs(angle) / cap + 5.0
        t0 = last_moved = time.monotonic()
        last = None
        while True:
            snap = self._guard(('wheels',))
            left = _wrap(yaw_goal - snap.odometry[2])
            if abs(left) <= o['base_angle_tolerance_rad']:
                break
            now = time.monotonic()
            if last is None or abs(_wrap(snap.odometry[2] - last)) > 0.002:
                last, last_moved = snap.odometry[2], now
            elif now - last_moved > o['base_stall_s']:
                raise MotionStopped(f'the base did not turn for {o["base_stall_s"]:g} s ({left:+.3f} rad left); stopped')
            if now > deadline:
                raise MotionStopped(f'the turn did not finish in time ({left:+.3f} rad left); stopped')
            self.base_stream.send(0.0, float(np.sign(left) * np.clip(abs(left) * 1.5, 0.08, cap)), period)
            time.sleep(period)
        self.base_stream.send(0.0, 0.0, period)
        return {'segment': 'turn', 'target_rad': angle, 'seconds': round(time.monotonic() - t0, 2)}

    def _gripper(self, side, action, width_m=None, current_a=None):
        with self.lock:
            side, width, current = self._gripper_args(side, action, width_m, current_a)
            self._require_armed()
            homed_now = False
            try:
                if not self.gripper.homed:
                    self.gripper.home(guard=lambda: self._guard(()))
                    homed_now = True
                self.gripper.command(side, width, current)
                end = time.monotonic() + self.options['gripper_settle_s']
                while time.monotonic() < end:
                    self._guard(())
                    time.sleep(0.1)
            except MotionStopped as exc:
                raise self._held(exc)
            except Exception as exc:
                raise self._failed(exc)
            measured = self.gripper.widths()
            self.last_motion = {'tool': 'rby1_gripper', 'at': time.time(), 'side': side, 'action': action}
            value = {'side': side, 'action': action, 'target_width_m': width, 'current_a': current,
                     'measured_width_m': measured.get(side), 'homed_now': homed_now, 'openings_m': measured}
            if measured.get(side) is not None and action != 'open':
                value['blocked_by_object'] = bool(measured[side] - width > 0.005)
            return value, []

    # ----- planning queries (read-only: no arming, no robot command) ----------------------
    def _frame_names(self):
        return sorted(set(self.kinematics.links) - {'base'} | {HEAD_CAMERA})

    def _hypothetical(self, q, joints):
        """q with components replaced as in rby1_move_body targets, without the limit checks (reported instead)."""
        q = np.array(q, dtype=np.float64)
        for name, spec in (joints or {}).items():
            current = component(q, name)
            if 'preset' in spec:
                if spec['preset'] not in PRESETS[name]:
                    raise ValueError(f"unknown {name} preset {spec['preset']!r}; choose from {sorted(PRESETS[name])}")
                values = np.asarray(PRESETS[name][spec['preset']], dtype=np.float64)
            elif 'positions' in spec:
                values = np.asarray(spec['positions'], dtype=np.float64)
                if values.shape != current.shape or not np.all(np.isfinite(values)):
                    raise ValueError(f'{name} needs {current.size} finite positions')
                if spec.get('relative'):
                    values = current + values
            else:
                raise ValueError(f'{name}: give positions or preset')
            q = with_component(q, name, values)
        return q

    def _pose_report(self, T):
        p, quat = pose(T)
        return {'position': np.round(p, 4).tolist(), 'quaternion': np.round(quat, 4).tolist(),
                'rpy_deg': np.round(Rotation.from_quat(quat).as_euler('xyz', degrees=True), 1).tolist()}

    def _body_report(self, q):
        """Hands, collisions, limits and gravity load of a posture."""
        kin = self.kinematics
        frames = kin.frames([*EE_FRAME.values(), *SHOULDER_FRAME.values()], q)
        load = np.abs(kin.gravity(q)) * self.options['gravity_model_scale'] / kin.torque_limit
        top = [j for j in np.argsort(-load)[:4] if np.isfinite(load[j])]
        outside = {}
        for name in JOINT_COMPONENTS:
            lower, upper = self._limits(name)
            start = COMPONENTS[name][0]
            for k, v in enumerate(component(q, name)):
                if v < lower[k] - 1e-9 or v > upper[k] + 1e-9:
                    outside[JOINT_NAMES[start + k]] = {'value': round(float(v), 4), 'allowed': [round(float(lower[k]), 3),
                                                                                              round(float(upper[k]), 3)]}
        return {'end_effectors': {side: self._pose_report(frames[f]) for side, f in EE_FRAME.items()},
                'hand_from_shoulder_m': {side: round(float(np.linalg.norm(frames[EE_FRAME[side]][:3, 3]
                                                                          - frames[SHOULDER_FRAME[side]][:3, 3])), 3)
                                         for side in ARMS},
                'self_collisions': [list(pair) for pair in sorted(kin.collisions(q))],
                'outside_limits': outside,
                'gravity_load': {JOINT_NAMES[j]: f'{load[j]:.0%}' for j in top},
                'gravity_load_limit': f"{self.options['static_torque_fraction']:.0%}"}

    def _center_of_mass(self, q):
        kin = self.kinematics
        with kin.lock:
            kin.state.set_q(np.asarray(q, dtype=np.float64))
            kin.dyn.compute_forward_kinematics(kin.state)
            total, moment = 0.0, np.zeros(3)
            for i in range(len(kin.links)):
                m = float(kin.dyn.compute_mass(kin.state, i))
                moment += m * np.asarray(kin.dyn.compute_center_of_mass(kin.state, 0, i)).ravel()
                total += m
        return (moment / total).round(3).tolist()

    def _fk(self, joints=None, frames=None):
        q = self._hypothetical(self._snapshot().q, joints)
        report = {'hypothetical': bool(joints), **{c: np.round(component(q, c), 4).tolist() for c in JOINT_COMPONENTS},
                  **self._body_report(q),
                  'head_camera': self._pose_report(self.kinematics.frame(HEAD_CAMERA, q)),
                  'center_of_mass': self._center_of_mass(q),
                  'wheel_axle_x_m': 0.228}
        if frames:
            unknown = set(frames) - set(self._frame_names())
            if unknown:
                raise ValueError(f'unknown frames {sorted(unknown)}')
            poses = self.kinematics.frames(list(frames), q)
            report['frames'] = {name: self._pose_report(T) for name, T in poses.items()}
        return report, []

    def _check(self, tool, arguments):
        if not isinstance(arguments, dict):
            raise ValueError('arguments must be an object')
        schema = next(t['inputSchema'] for t in self.tools() if t['name'] == tool)
        unknown = set(arguments) - set(schema['properties'])
        missing = set(schema['required']) - set(arguments)
        if unknown or missing:
            raise ValueError(f'{tool} arguments: unknown {sorted(unknown)}, missing {sorted(missing)}')
        snap = self._snapshot()
        value = {'tool': tool}
        controller_solved = tool in ('rby1_cartesian', 'rby1_optimal_control', 'rby1_cartesian_impedance',
                                     'rby1_link_impedance') or (tool == 'rby1_move_eef' and
                                                                self.options['controller_commands'])
        try:
            goals = self._plan(tool, arguments, snap.q)
        except ValueError as exc:
            value.update(accepted=False, reason=str(exc))
        else:
            value['accepted'] = True
            if controller_solved:
                value['checked'] = ('targets only: the robot controller solves the joints and the path, which are '
                                    'watched while the command runs')
            if goals:
                q = snap.q.copy()
                for name, goal in goals.items():
                    q = with_component(q, name, goal)
                steps = {c: float(np.max(np.abs(g - component(snap.q, c)))) for c, g in goals.items()}
                value.update(goals={c: np.round(g, 4).tolist() for c, g in goals.items()},
                             planned_s=round(max([self._duration(steps[c], c) for c in goals] + [0.5]), 2),
                             after=self._body_report(q))
        state = arming.read(self.config.root)
        try:
            manager = self.robot.manager_state()
        except Exception as exc:
            manager = f'unknown ({str(exc)[:80]})'
        hot = snap.temperature is not None and float(np.max(snap.temperature)) > self.options['max_motor_temp_c']
        value['ready_to_move'] = {'armed': state['armed'], 'emo_pressed': snap.emo_pressed, 'controller': manager,
                                  'drive_too_hot': bool(hot)}
        return value, []

    # ----- observation --------------------------------------------------------------------
    def _frame_dir(self, frame_id):
        if not isinstance(frame_id, str) or not frame_id.isalnum():
            raise ValueError('invalid frame_id')
        path = self.frames_root/frame_id
        if not (path/'meta.json').is_file():
            raise ValueError(f'unknown frame_id {frame_id}')
        return path

    def _observe(self):
        if self.camera is None:
            raise RuntimeError(f'the head camera is not available in this session ({self.camera_error or "disabled"})'
                               '; the operator checks the ZED')
        latest = self.camera.read()
        if latest is None or time.time() - latest[0] > self.options['camera_max_age_s']:
            raise RuntimeError('no recent head-camera image'
                               + (f' ({self.camera.error})' if getattr(self.camera, 'error', None) else '')
                               + '; the operator checks the ZED')
        stamp, rgb, depth = latest
        snap = self._snapshot()
        frame_id = time.strftime('%Y%m%d%H%M%S') + uuid.uuid4().hex[:6]
        out_dir = self.frames_root/frame_id
        out_dir.mkdir(parents=True, exist_ok=True)
        bgr = cv.cvtColor(np.asarray(rgb), cv.COLOR_RGB2BGR)
        cv.imwrite(str(out_dir/'head_camera.jpg'), bgr, [cv.IMWRITE_JPEG_QUALITY, 95])
        entry = {'rgb': 'head_camera.jpg', 'depth': None, 'width': bgr.shape[1], 'height': bgr.shape[0],
                 'K': self.camera.intrinsics(), 'T_base_cam': self.kinematics.frame(HEAD_CAMERA, snap.q).tolist(),
                 'captured_at': stamp}
        if self.depth and depth is not None:
            depth_mm = np.nan_to_num(np.asarray(depth, dtype=np.float32) * 1000.0, nan=0.0, posinf=0.0,
                                     neginf=0.0).clip(0, 65535).astype(np.uint16)
            cv.imwrite(str(out_dir/'head_camera_depth.png'), depth_mm)
            entry['depth'] = 'head_camera_depth.png'
        meta = {'frame_id': frame_id, 'time': stamp, 'joints_time': snap.t,
                'joints': {n: round(float(v), 5) for n, v in zip(JOINT_NAMES, snap.q)},
                'depth_units': 'mm (uint16, 0 = invalid)', 'cameras': {'head_camera': entry}}
        (out_dir/'meta.json').write_text(json.dumps(meta, indent=1))
        scale = self.options['agent_image_width'] / bgr.shape[1]
        small = cv.resize(bgr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA) if scale < 1 else bgr
        ok, jpeg = cv.imencode('.jpg', small, [cv.IMWRITE_JPEG_QUALITY, 85])
        images = [{'type': 'image', 'mimeType': 'image/jpeg', 'data': base64.b64encode(jpeg).decode()}]
        report = {'head_camera': {'preview_scale': round(min(scale, 1.0), 4), 'width': entry['width'],
                                  'height': entry['height'], 'depth': entry['depth'] is not None,
                                  'age_s': round(time.time() - stamp, 3)}}
        return {'frame_id': frame_id, 'directory': f'runtime/frames/{frame_id}', 'images_in_order': ['head_camera'],
                'cameras': report, **self._state_report(snap)}, images

    def _pixel_to_position(self, frame_id, camera, pixels, radius=3):
        directory = self._frame_dir(frame_id)
        meta = json.loads((directory/'meta.json').read_text())
        entry = meta['cameras'].get(camera)
        if entry is None or not entry['depth']:
            raise ValueError(f'frame {frame_id} has no depth for camera {camera}')
        depth = cv.imread(str(directory/entry['depth']), cv.IMREAD_UNCHANGED)
        K = np.asarray(entry['K'])
        T_base_cam = np.asarray(entry['T_base_cam'])
        sx, sy = depth.shape[1] / entry['width'], depth.shape[0] / entry['height']
        points = []
        for u, v in pixels:
            if not (0 <= u < entry['width'] and 0 <= v < entry['height']):
                points.append({'pixel': [u, v], 'error': 'outside the image'})
                continue
            r = int(radius)
            ui, vi = int(round(u * sx)), int(round(v * sy))
            window = depth[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]
            valid = window[window > 0]
            if valid.size == 0:
                points.append({'pixel': [u, v], 'error': 'no valid depth here'})
                continue
            z = float(np.median(valid)) / 1000.0
            p_cam = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
            p_base = (T_base_cam @ np.append(p_cam, 1.0))[:3]
            points.append({'pixel': [u, v], 'position': np.round(p_base, 4).tolist(), 'depth_m': round(z, 4),
                           'valid_fraction': round(valid.size / window.size, 2)})
        return {'frame_id': frame_id, 'camera': camera, 'points': points}, []

    def _stream(self, cameras=None, seconds_ago=None, sheet=None):
        from services import stream as recording
        if self.stream is None:
            raise ValueError('the continuous recording is disabled in this session')
        if seconds_ago is not None and sheet is not None:
            raise ValueError('give seconds_ago or sheet, not both')
        names = list(cameras or self.cameras)
        window_ms = int(self.options['stream_window_s'] * 1000)
        now_ms = int(time.time() * 1000)
        images, report = [], {'now_ms': now_ms, 'cameras': names}

        def encode(image, width):
            scale = width / image.shape[1]
            small = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA) if scale < 1 else image
            ok, jpeg = cv.imencode('.jpg', small, [cv.IMWRITE_JPEG_QUALITY, 85])
            images.append({'type': 'image', 'mimeType': 'image/jpeg', 'data': base64.b64encode(jpeg).decode()})

        if sheet is not None:
            seconds = min(float(sheet['seconds']), window_ms / 1000)
            report['sheets'] = []
            for name in names:
                grid, times = recording.contact_sheet(self.stream, name, now_ms - int(seconds * 1000), now_ms,
                                                      int(sheet.get('count', 12)), now_ms)
                if grid is None:
                    report['sheets'].append({'camera': name, 'error': 'no frames in that span'})
                    continue
                encode(grid, 1280)
                report['sheets'].append({'camera': name, 'frame_ages_s': [round((now_ms - t) / 1000, 1) for t in times]})
            report['robot'] = {'start': recording.robot_at(self.stream, now_ms - int(seconds * 1000)),
                               'end': recording.robot_at(self.stream, now_ms)}
            return report, images
        ages = list(seconds_ago if seconds_ago is not None else [0])
        if len(ages) * max(1, len(names)) > 12:
            raise ValueError('at most 12 images per call (seconds_ago x cameras)')
        report['frames'] = []
        for age in ages:
            if age * 1000 > window_ms:
                raise ValueError(f"only the last {self.options['stream_window_s']:g} s are kept")
            target = now_ms - int(float(age) * 1000)
            entry = {'seconds_ago': age, 'robot': recording.robot_at(self.stream, target), 'images': []}
            for name in names:
                found = recording.nearest(self.stream, name, target)
                if found is None:
                    entry['images'].append({'camera': name, 'error': 'no frame near that time'})
                    continue
                t, path = found
                image = recording.label(cv.imread(str(path)), f'{name} -{(now_ms - t) / 1000:.1f}s')
                encode(image, self.options['agent_image_width'])
                entry['images'].append({'camera': name, 't_ms': t, 'age_s': round((now_ms - t) / 1000, 2),
                                        'file': str(Path('runtime/stream')/path.relative_to(self.stream))})
            report['frames'].append(entry)
        return report, images

    # ----- watchdog (between calls) -------------------------------------------------------
    def _watch(self):
        period = 1.0 / self.options['watchdog_hz']
        while not self._watch_stop.wait(period):
            try:
                self._watch_once()
            except Exception:
                pass  # a failed read is retried next period; motions check the robot themselves

    def _watch_once(self):
        """Faults and drives not ready disarm the session; a joint pushing over its rating is re-targeted."""
        robot = self.robot
        if robot is None:
            return
        manager = robot.manager_state()
        if manager in FAULTS:
            if self.fault_disarmed_at is None:
                self._disarm(f'the robot controller is in {manager}', fault=manager)
            return
        if not self.enabled or manager != 'Enabled':
            return  # this session has not taken the drives yet
        snap = self._snapshot()
        if not np.all(snap.ready):
            if self.fault_disarmed_at is None:
                self._disarm(f'drives not ready: {[n for n, r in zip(JOINT_NAMES, snap.ready) if not r]}',
                             fault='drive')
            return
        over = self._over_torque(snap)
        now = time.monotonic()
        if not over:
            self.over_since = None
            return
        if self.over_since is None:
            self.over_since = now
        if now - self.over_since < self.options['watchdog_torque_s']:
            return
        if not self.lock.acquire(blocking=False):
            return  # a motion is running and its own guard handles it
        try:
            self._relieve(f'watchdog, between calls: {over}')
        finally:
            self.over_since = now + 5.0  # at most one re-target per 5 s while the load stays high
            self.lock.release()

    # ----- harness hooks ------------------------------------------------------------------
    def _state_report(self, snap=None):
        snap = snap or self._snapshot()
        report = {name: np.round(component(snap.q, name), 4).tolist() for name in JOINT_COMPONENTS}
        frames = self.kinematics.frames(list(EE_FRAME.values()), snap.q)
        for side, frame in EE_FRAME.items():
            p, q = pose(frames[frame])
            report[f'{side}_ee'] = {'position': np.round(p, 4).tolist(), 'quaternion': np.round(q, 4).tolist(),
                                    'rpy_deg': np.round(Rotation.from_quat(q).as_euler('xyz', degrees=True),
                                                        1).tolist()}
        report['odometry'] = {'x': round(snap.odometry[0], 4), 'y': round(snap.odometry[1], 4),
                              'yaw': round(snap.odometry[2], 4)}
        return report

    def _robot_record(self):
        """The recorder's robot state record: joints, end-effector positions, odometry, wrist forces."""
        snap = self._snapshot()
        record = {name: np.round(component(snap.q, name), 4).tolist() for name in JOINT_COMPONENTS}
        frames = self.kinematics.frames(list(EE_FRAME.values()), snap.q)
        for side, frame in EE_FRAME.items():
            record[f'{side}_ee'] = np.round(frames[frame][:3, 3], 4).tolist()
        record['odometry'] = [round(v, 4) for v in snap.odometry]
        record['ft_force'] = {side: np.round(snap.ft[side]['force'], 2).tolist() for side in ARMS}
        return record

    def status(self):
        snap = self._snapshot()
        value = {'robot': 'rainbow_rby1_a', 'motion': arming.read(self.config.root), 'emo_pressed': snap.emo_pressed,
                 **self._state_report(snap),
                 'wrist_force_n': {side: np.round(snap.ft[side]['force'], 1).tolist() for side in ARMS},
                 'battery': snap.battery,
                 'limits': {name: dict(zip(('lower', 'upper'), (np.round(a, 3).tolist() for a in self._limits(name))))
                            for name in JOINT_COMPONENTS},
                 'stale_joints': [n for n, age in zip(JOINT_NAMES[2:], snap.age_s[2:])
                                  if not age <= self.options['state_max_age_s']],
                 'base': {'max_linear_mps': self.options['base_max_linear_mps'],
                          'max_angular_radps': self.options['base_max_angular_radps'], 'odometry': 'wheel'},
                 'enabled': self.enabled, 'enable_steps': self.enable_steps,
                 'safety': self._safety_report(snap),
                 'grippers': None if self.gripper is None else {'homed': self.gripper.homed,
                                                               'openings_m': self.gripper.widths()},
                 'cameras': list(self.cameras), 'depth': self.depth,
                 'sensors': self._sensor_status(),
                 'last_motion': self.last_motion,
                 'stream': None if self.stream is None else {
                     'directory': 'runtime/stream', 'window_s': self.options['stream_window_s'],
                     'fps': self.options['stream_fps'], 'robot_hz': self.options['robot_log_hz'],
                     'now_ms': int(time.time() * 1000)}}
        try:
            value['control'] = self.robot.control_state()
        except Exception as exc:
            value['control'] = {'error': str(exc)[:200]}
        return value

    def _safety_report(self, snap):
        o = self.options
        report = {'fault': self.fault, 'events': self.events[-5:],
                  'speeds_radps': {'arms': o['joint_speed_radps'], 'torso': o['torso_speed_radps'],
                                   'head': o['head_speed_radps']},
                  'torso_max_squat_rad': o['torso_max_squat_rad'],
                  'torque_guard_fraction': o['torque_guard_fraction']}
        if snap.torque is not None:
            fraction = np.abs(snap.torque) / self.kinematics.torque_limit
            top = np.argsort(-fraction)[:3]
            report['highest_torque_load'] = {JOINT_NAMES[j]: f'{fraction[j]:.0%}' for j in top}
        if snap.temperature is not None:
            j = int(np.argmax(snap.temperature))
            report['hottest_drive'] = {JOINT_NAMES[j]: float(snap.temperature[j])}
        return report

    def _sensor_status(self):
        """Which perception sources are delivering now."""
        if self.camera is None:
            return {'head_camera': f'unavailable ({self.camera_error or "disabled"})'}
        latest = self.camera.read()
        age = None if latest is None else time.time() - latest[0]
        if age is not None and age <= self.options['camera_max_age_s']:
            return {'head_camera': 'ok'}
        error = getattr(self.camera, 'error', None)
        return {'head_camera': 'no recent images' + (f' ({error})' if error else '')}

    def close(self):
        self._watch_stop.set()
        if self.watchdog is not None:
            self.watchdog.join(timeout=2.0)
            self.watchdog = None
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
        if self.robot is None:
            return
        try:
            self._hold_all()
        finally:
            for part in (self.gripper, self.camera, self.robot):
                try:
                    if part is not None:
                        part.close()
                except Exception:
                    pass
            self.robot = None
