"""Bounded R1Pro control through the unchanged BEHAVIOR 23-D action interface.

This module imports neither Isaac nor torch at import time. It reads robot-only
kinematics and computes joint commands; the injected ``step_fn`` is the *only*
way it advances physics. It does not implement semantic grasp/place success,
collision-free planning, target lookup, teleportation, or simulator state edits.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


ACTION_INDICES = {
    'base': [0, 1, 2], 'trunk': [3, 4, 5, 6],
    'arm_left': list(range(7, 14)), 'gripper_left': [14],
    'arm_right': list(range(15, 22)), 'gripper_right': [22],
}
MAX_STEPS = 120
MAX_TRANSLATION_M = .30
MAX_JOINT_INCREMENT_RAD = .04


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _vector(value, size, name):
    result = _array(value)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite {size}-vector')
    return result.copy()


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number')
    result = _array(value)
    if result.shape != () or not np.isfinite(result):
        raise ValueError(f'{name} must be a finite number')
    return float(result)


def _steps(value, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not minimum <= value <= MAX_STEPS:
        raise ValueError(f'steps must be an integer in [{minimum}, {MAX_STEPS}]')
    return int(value)


def _side(value):
    if value not in ('left', 'right'):
        raise ValueError('side must be left or right')
    return value


def _quat(value):
    q = _vector(value, 4, 'orientation (xyzw)')
    scale = np.max(np.abs(q))
    if scale == 0:
        raise ValueError('orientation must be a nonzero xyzw quaternion')
    # Normalize the finite direction without overflow/underflow in sum(q*q).
    q = q / scale
    return q / np.linalg.norm(q)


def _quat_mul(a, b):
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]),
                 a[3] * b[3] - np.dot(a[:3], b[:3])]


def _quat_inverse(q):
    return np.r_[-q[:3], q[3]]


def quaternion_matrix(value):
    """Rotation matrix for an xyzw quaternion (pure NumPy)."""
    x, y, z, w = _quat(value)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def orientation_error(target, current):
    """Shortest rotation vector, expressed in the poses' shared reference frame."""
    error = _quat_mul(_quat(target), _quat_inverse(_quat(current)))
    if error[3] < 0:
        error = -error
    sine = np.linalg.norm(error[:3])
    if sine < 1e-10:
        return 2 * error[:3]
    return error[:3] * (2 * np.arctan2(sine, np.clip(error[3], 0, 1)) / sine)


def damped_least_squares(jacobian, error, damping=.05, max_delta=MAX_JOINT_INCREMENT_RAD):
    """Return a finite, uniformly bounded joint increment, including at singularities."""
    jacobian, error = _array(jacobian), _array(error)
    damping, max_delta = _scalar(damping, 'damping'), _scalar(max_delta, 'max_delta')
    if (jacobian.ndim != 2 or not all(jacobian.shape)
            or error.shape != (jacobian.shape[0],)
            or not np.isfinite(jacobian).all() or not np.isfinite(error).all()
            or damping <= 0 or max_delta <= 0):
        raise ValueError('DLS requires a finite matrix/vector and positive damping/max_delta')
    delta = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + damping**2 * np.eye(jacobian.shape[0]), error)
    scale = max(1., np.max(np.abs(delta)) / max_delta)
    return delta / scale


def _bounded_norm(value, maximum):
    return value * min(1., maximum / max(np.linalg.norm(value), 1e-12))


class R1ProControl:
    """Robot-only control for a *started* ``BehaviorEnvironment``.

    Public units: metres, radians, m/s, rad/s; orientation is xyzw.
    ``base`` means the current robot base_footprint frame, not the articulation
    root or a camera. Gripper ``opening`` is a fraction, 0 closed / 1 open.
    Every result reports measured errors; none implies grasp/place success.
    """

    def __init__(self, env, step_fn):
        if not callable(step_fn):
            raise TypeError('step_fn must be callable')
        self.env, self.step_fn = env, step_fn
        self.backend, self.robot = env.backend, env.backend.robot
        self._execution_failed = False
        self.env._require_ready(allow_audit=False)
        self.dt = float(self.backend.og.sim.get_sim_step_dt())
        if not np.isclose(self.dt, 1 / 30):
            raise RuntimeError('R1ProControl requires the registered 30 Hz action interface')
        actual = {name: _array(idx).astype(int).tolist() for name, idx in self.robot.controller_action_idx.items()
                  if len(idx)}
        if self.robot.action_dim != 23 or actual != ACTION_INDICES or self.robot._action_normalize:
            raise RuntimeError('Unexpected R1Pro 23-D action ordering or normalization')
        self.action_idx = {name: np.asarray(idx, dtype=int) for name, idx in actual.items()}
        self.joint_idx = {'trunk': _array(self.robot.trunk_control_idx).astype(int)}
        for side in ('left', 'right'):
            self.joint_idx[f'arm_{side}'] = _array(self.robot.arm_control_idx[side]).astype(int)
            self.joint_idx[f'gripper_{side}'] = _array(self.robot.gripper_control_idx[side]).astype(int)
        if any(len(self.joint_idx[name]) != count for name, count in
               [('trunk', 4), ('arm_left', 7), ('arm_right', 7), ('gripper_left', 2), ('gripper_right', 2)]):
            raise RuntimeError('Unexpected R1Pro physical joint layout')
        self._validate_controllers()
        hold = self._new_hold()
        # Preserve the last explicit finger command while other tools move. A
        # closed hand must not be silently reopened to its contact-limited pose.
        self._held_grippers = {side: float(hold[self.action_idx[f'gripper_{side}']][0])
                               for side in ('left', 'right')}

    def _controller(self, name):
        group, member = self.robot._controllers[name]
        return self.backend.ControllerView._controller_groups[group], int(member)

    def _validate_controllers(self):
        for name in ('trunk', 'arm_left', 'arm_right'):
            controller, _ = self._controller(name)
            if (type(controller).__name__ != 'JointController' or controller.motor_type != 'position'
                    or controller.use_delta_commands or controller.command_input_limits is not None
                    or controller.command_output_limits is not None):
                raise RuntimeError(f'{name} must retain absolute, unnormalized JointController position commands')
            if not np.array_equal(_array(controller.dof_idx).astype(int), self.joint_idx[name]):
                raise RuntimeError(f'{name} controller does not match its physical joint indices')
        base, _ = self._controller('base')
        if (type(base).__name__ != 'HolonomicBaseJointController' or base.motor_type != 'velocity'
                or base.use_delta_commands):
            raise RuntimeError('base must retain the registered velocity HolonomicBaseJointController')
        self.base_input = np.stack([_vector(v, 3, 'base input limits') for v in base.command_input_limits])
        self.base_output = np.stack([_vector(v, 3, 'base output limits') for v in base.command_output_limits])
        if not (np.allclose(self.base_input, [-np.ones(3), np.ones(3)]) and
                np.allclose(self.base_output, [[-.75, -.75, -1], [.75, .75, 1]])):
            raise RuntimeError('Base command mapping differs from behavior_ap r1pro.yaml')
        for side in ('left', 'right'):
            controller, _ = self._controller(f'gripper_{side}')
            if (type(controller).__name__ != 'MultiFingerGripperController'
                    or controller._motor_type != 'position' or controller._mode != 'smooth'
                    or controller._inverted or controller.command_input_limits is None
                    or not np.allclose(np.stack([_array(v).reshape(-1)
                                                 for v in controller.command_input_limits]), [[-1], [1]])):
                raise RuntimeError('Gripper must retain non-inverted smooth position commands in [-1, 1]')
            lower, upper = self._limits(f'gripper_{side}')
            output = np.array([float(_array(v).reshape(-1)[0]) for v in controller.command_output_limits])
            if not np.allclose(output, [lower.mean(), upper.mean()]):
                raise RuntimeError('Gripper output mapping differs from physical joint limits')

    def _limits(self, name):
        idx = self.joint_idx[name]
        lower, upper = _array(self.robot.joint_lower_limits)[idx], _array(self.robot.joint_upper_limits)[idx]
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
            raise RuntimeError(f'Invalid physical joint limits for {name}')
        return lower, upper

    def _q(self, name):
        return _array(self.robot.get_joint_positions())[self.joint_idx[name]]

    def _new_hold(self):
        action = _vector(self.backend._make_hold_action(self.robot), 23, 'hold action')
        action[self.action_idx['base']] = 0.
        for side, command in getattr(self, '_held_grippers', {}).items():
            action[self.action_idx[f'gripper_{side}']] = command
        return action

    def adopt_external_action(self, action):
        """Bookkeeping only: preserve the last learned gripper command on handback."""
        action = _vector(action, 23, 'external policy action')
        for side in ('left', 'right'):
            self._held_grippers[side] = float(action[self.action_idx[f'gripper_{side}']][0])

    def _advance(self, action):
        if self._execution_failed:
            raise RuntimeError('Previous physical action/ACK failed; this control instance cannot continue')
        self.env._require_ready(allow_audit=False)
        if self.env.done:
            return False
        action = _vector(action, 23, 'action').astype(np.float32)
        before = int(self.env.steps)
        # Never retry this call: a callback failure can follow physical advance.
        try:
            self.step_fn(action)
            if int(self.env.steps) != before + 1:
                raise RuntimeError('step_fn must acknowledge exactly one physical action; continuation rejected')
        except BaseException:
            self._execution_failed = True
            raise
        return True

    def _pose(self, side, frame='base'):
        values = (self.robot.get_relative_eef_pose(side) if frame == 'base'
                  else self.robot.eef_links[side].get_position_orientation())
        return _vector(values[0], 3, 'measured EEF position'), _quat(values[1])

    def _base_pose(self):
        pos, quat = self.robot.get_position_orientation()
        return _vector(pos, 3, 'measured base position'), _quat(quat)

    def _relative_target(self, position, orientation, frame):
        if frame == 'base':
            return position, orientation
        base_pos, base_quat = self._base_pose()
        return (quaternion_matrix(base_quat).T @ (position - base_pos),
                _quat_mul(_quat_inverse(base_quat), orientation))

    def _arm_jacobian(self, side):
        # Runtime-only dependency. Exactly the indexing used by this OG version's
        # InverseKinematicsController: body index excludes root; floating-base
        # virtual columns precede joint columns. Do not assume robot view row 0.
        from omnigibson.utils.usd_utils import ControllableObjectViewAPI as View
        controller, member = self._controller(f'arm_{side}')
        route = controller.routing_path
        row = int(_array(controller.view_row_indices)[member])
        all_q = View.get_all_joint_positions(route)
        all_j = View.get_all_relative_jacobians(route)
        link = View.get_link_index(route, self.robot.eef_link_names[side]) - 1
        offset = all_j.shape[-1] - all_q.shape[-1]
        columns = self.joint_idx[f'arm_{side}'] + offset
        if offset not in (0, 6) or link < 0:
            raise RuntimeError('Unexpected articulation Jacobian layout')
        jacobian = _array(all_j[row, link])[:, columns]
        if jacobian.shape != (6, 7) or not np.isfinite(jacobian).all():
            raise RuntimeError('Invalid measured arm Jacobian')
        return jacobian

    def _result(self, before, requested, **values):
        return {'advanced_steps': int(self.env.steps) - before, 'requested_steps': requested,
                'step': int(self.env.steps), 'environment_done': bool(self.env.done), **values}

    def read_state(self):
        """Fresh robot kinematics only; no object scope, object poses or scorer data."""
        self.env._require_ready()
        base_pos, base_quat = self._base_pose()
        joints = _array(self.robot.get_joint_positions())
        velocities = _array(self.robot.get_joint_velocities())
        if joints.shape != velocities.shape or not np.isfinite(joints).all() or not np.isfinite(velocities).all():
            raise RuntimeError('Invalid measured robot joint state')
        components = {}
        for name, idx in self.joint_idx.items():
            lower, upper = self._limits(name)
            components[name] = {'joint_positions': joints[idx].tolist(), 'joint_velocities': velocities[idx].tolist(),
                                'lower_limits': lower.tolist(), 'upper_limits': upper.tolist(),
                                'joint_indices': idx.tolist()}
        eef = {}
        for side in ('left', 'right'):
            eef[side] = {}
            for frame in ('base', 'world'):
                pos, quat = self._pose(side, frame)
                eef[side][frame] = {'position': pos.tolist(), 'orientation_xyzw': quat.tolist()}
        return {'step': int(self.env.steps), 'base_world': {'position': base_pos.tolist(),
                'orientation_xyzw': base_quat.tolist(),
                'linear_velocity': _vector(self.robot.get_linear_velocity(), 3, 'base linear velocity').tolist(),
                'angular_velocity': _vector(self.robot.get_angular_velocity(), 3, 'base angular velocity').tolist()},
                'components': components, 'eef': eef,
                'units': {'position': 'metres', 'joints': 'radians (gripper: metres)', 'orientation': 'xyzw'},
                'control_limits': {'max_tool_steps': MAX_STEPS, 'max_translation_m': MAX_TRANSLATION_M,
                                   'action_frequency_hz': 1 / self.dt}}

    def move_base(self, vx, vy, yaw_rate, steps=60):
        """Body-frame velocity command; ``steps`` includes a final zero-velocity step.

        This uses the existing velocity servo. It is not collision-avoidance or
        navigation; actual displacement and final velocity are returned for the
        caller to inspect before issuing the next bounded command.
        """
        steps = _steps(steps, minimum=2)
        velocity = np.array([_scalar(vx, 'vx'), _scalar(vy, 'vy'), _scalar(yaw_rate, 'yaw_rate')])
        if np.any(velocity < self.base_output[0]) or np.any(velocity > self.base_output[1]):
            raise ValueError('base velocity exceeds registered limits: vx/vy ±0.75 m/s, yaw ±1 rad/s')
        duration = (steps - 1) * self.dt
        if np.linalg.norm(velocity[:2]) * duration > MAX_TRANSLATION_M + 1e-9:
            raise ValueError('base command exceeds 0.30 m per call; use shorter observed segments')
        if abs(velocity[2]) * duration > np.pi / 2 + 1e-9:
            raise ValueError('base turn exceeds pi/2 per call; use shorter observed segments')
        self.env._require_ready(allow_audit=False)
        before = int(self.env.steps)
        start, start_quat = self._base_pose()
        action = self._new_hold()
        scaled = self.base_input[0] + ((velocity - self.base_output[0]) /
                 (self.base_output[1] - self.base_output[0])) * (self.base_input[1] - self.base_input[0])
        commanded = 0
        travelled = 0.
        previous = start
        for i in range(steps):
            # The last action is always braking. Also brake once the measured
            # path reaches the bound (e.g. unexpected external displacement).
            moving = i < steps - 1 and travelled < MAX_TRANSLATION_M
            action[self.action_idx['base']] = scaled if moving else 0.
            if not self._advance(action):
                break
            commanded += int(moving)
            current, _ = self._base_pose()
            travelled += float(np.linalg.norm((current - previous)[:2]))
            previous = current
            if not moving:
                break
        end, end_quat = self._base_pose()
        return self._result(before, steps, tool='move_base', commanded_velocity=velocity.tolist(),
                            velocity_command_steps=commanded, displacement_world=(end - start).tolist(),
                            displacement_start_base=(quaternion_matrix(start_quat).T @ (end - start)).tolist(),
                            measured_path_length_m=travelled,
                            rotation_delta_rad=orientation_error(end_quat, start_quat).tolist(),
                            final_linear_velocity_world=_array(self.robot.get_linear_velocity()).tolist(),
                            final_angular_velocity_world=_array(self.robot.get_angular_velocity()).tolist())

    def move_arm(self, side, position, orientation=None, frame='base', steps=60):
        """Closed-loop DLS IK; each step emits seven absolute joint targets.

        No orientation means hold the starting EEF orientation in ``frame``.
        A base-frame target stays relative to the moving base; a world-frame
        target stays fixed in world coordinates. The other arm and trunk hold.
        """
        side, steps = _side(side), _steps(steps)
        if frame not in ('base', 'world'):
            raise ValueError('frame must be base or world')
        position = _vector(position, 3, 'position')
        self.env._require_ready(allow_audit=False)
        initial_pos, initial_quat = self._pose(side, frame)
        if np.linalg.norm(position - initial_pos) > MAX_TRANSLATION_M + 1e-9:
            raise ValueError('EEF target exceeds 0.30 m from current pose; use shorter observed segments')
        goal_quat = initial_quat if orientation is None else _quat(orientation)
        before = int(self.env.steps)
        action, name = self._new_hold(), f'arm_{side}'
        lower, upper = self._limits(name)
        for _ in range(steps):
            if self.env.done:
                break
            target_pos, target_quat = self._relative_target(position, goal_quat, frame)
            current_pos, current_quat = self._pose(side)
            translation = target_pos - current_pos
            rotation = orientation_error(target_quat, current_quat)
            if np.linalg.norm(translation) <= .005 and np.linalg.norm(rotation) <= .04:
                break
            error = np.r_[_bounded_norm(translation, .025), _bounded_norm(rotation, .10)]
            # Orientation residual/Jacobian share the base reference frame. A
            # conservative damping and joint increment bound limit singularity
            # amplification; this is a local IK servo, not a path planner.
            delta = damped_least_squares(self._arm_jacobian(side), error)
            action[self.action_idx[name]] = np.clip(self._q(name) + delta, lower, upper)
            if not self._advance(action):
                break
        final_pos, final_quat = self._pose(side, frame)
        position_error = float(np.linalg.norm(position - final_pos))
        angular_error = float(np.linalg.norm(orientation_error(goal_quat, final_quat)))
        return self._result(before, steps, tool='move_arm', side=side, frame=frame,
                            target_position=position.tolist(), target_orientation_xyzw=goal_quat.tolist(),
                            actual_position=final_pos.tolist(), actual_orientation_xyzw=final_quat.tolist(),
                            position_error_m=position_error, orientation_error_rad=angular_error,
                            pose_tolerance_met=bool(position_error <= .005 and angular_error <= .04),
                            joint_positions=self._q(name).tolist())

    def gripper(self, side, opening, steps=30):
        """Set opening fraction [0,1]. Returned finger motion is not a grasp verdict."""
        side, steps, opening = _side(side), _steps(steps), _scalar(opening, 'opening')
        if not 0 <= opening <= 1:
            raise ValueError('opening must be in [0, 1]: 0 closed, 1 open')
        self.env._require_ready(allow_audit=False)
        before = int(self.env.steps)
        action, name = self._new_hold(), f'gripper_{side}'
        lower, upper = self._limits(name)
        initial_command = float(action[self.action_idx[name]][0])
        goal_command = 2 * opening - 1
        for i in range(steps):
            action[self.action_idx[name]] = initial_command + (goal_command - initial_command) * (i + 1) / steps
            if not self._advance(action):
                break
            self._held_grippers[side] = float(action[self.action_idx[name]][0])
        actual = self._q(name)
        return self._result(before, steps, tool='gripper', side=side, requested_opening_fraction=opening,
                            actual_joint_positions_m=actual.tolist(),
                            measured_opening_fraction=float(np.mean(np.clip((actual - lower) / (upper - lower), 0, 1))),
                            requested_finger_positions_m=(lower + opening * (upper - lower)).tolist())

    def torso(self, joints, steps=60):
        """Interpolate four absolute torso angles, respecting physical joint limits."""
        steps, goal = _steps(steps), _vector(joints, 4, 'torso joints')
        self.env._require_ready(allow_audit=False)
        lower, upper = self._limits('trunk')
        if np.any(goal < lower) or np.any(goal > upper):
            raise ValueError('torso target is outside physical joint limits')
        initial = self._q('trunk')
        if np.max(np.abs(goal - initial)) > steps * MAX_JOINT_INCREMENT_RAD + 1e-9:
            raise ValueError('torso interpolation exceeds 0.04 rad per step; use more steps or smaller segments')
        before = int(self.env.steps)
        action = self._new_hold()
        for i in range(steps):
            desired = initial + (goal - initial) * (i + 1) / steps
            measured = self._q('trunk')
            # Do not build up a large servo jump when blocked by contact.
            target = np.clip(desired, measured - MAX_JOINT_INCREMENT_RAD, measured + MAX_JOINT_INCREMENT_RAD)
            action[self.action_idx['trunk']] = np.clip(target, lower, upper)
            if not self._advance(action):
                break
        actual = self._q('trunk')
        return self._result(before, steps, tool='torso', target_joints=goal.tolist(),
                            actual_joints=actual.tolist(), joint_error_rad=(goal - actual).tolist())
