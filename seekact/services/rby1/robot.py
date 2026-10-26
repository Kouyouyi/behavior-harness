"""The thin live half of the RB-Y1 backend: the only module that calls rby1_sdk's robot API.

It connects to the robot's controller over gRPC (`rby1_sdk.create_robot_a`, firmware/SDK
0.8.3 on this robot) and offers the backend a small surface: a state snapshot, bringing
power / servo / control manager up (a controller fault is reset only when the caller says
the operator allowed it), timed joint-position commands, and a base velocity stream. Everything in it is a plain call of the SDK's public API: no vendor
code is patched, and no robot parameter is written (`set_parameter` is never called).

SDK facts it relies on (rby1_sdk 0.8.3):
- `connect()` only opens the gRPC channel: it changes no power, servo or control state.
- Dropping a command handler or stream cancels its command (their destructors call
  Cancel), so every handler that must keep running is kept on this object.
- Cancelling a command stops the joints with the controller's own deceleration, and the
  drives then hold where they stopped.
- A finished joint-position command leaves the joints holding at its target. A joint that
  cannot reach that target (an end stop, a contact) keeps pushing: the drive's integral
  term winds up to its current limit and the drive faults (seen on the robot: torso_2 at
  2.6 x its rated torque for 1.2 s, then a MajorFault and every joint dropped).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from .kinematics import COMPONENTS

BODY_SETTERS = {'torso': 'set_torso_command', 'right_arm': 'set_right_arm_command',
                'left_arm': 'set_left_arm_command'}


@dataclass
class Snapshot:
    """One robot state read; every array is in Model_A.robot_joint_names order."""
    t: float                         # time.time() when the reply arrived
    q: np.ndarray                    # (24,) rad
    ready: np.ndarray                # (24,) bool: joint drive is ready
    age_s: np.ndarray                # (24,) s since that joint last reported
    emo_pressed: bool
    ft: dict                         # {'right'|'left': {'force': [3] N, 'torque': [3] Nm, 'age_s'}}
    odometry: tuple                  # (x m, y m, yaw rad) of the base in the odometry frame
    battery: dict = field(default_factory=dict)
    torque: np.ndarray = None        # (24,) N m measured by the drives
    velocity: np.ndarray = None      # (24,) rad/s
    temperature: np.ndarray = None   # (24,) deg C of each drive


def _seconds(delta):
    return delta.total_seconds() if hasattr(delta, 'total_seconds') else float(delta)


class Handle:
    """A sent command: kept alive here so that the SDK does not cancel it."""

    def __init__(self, handler):
        self.handler = handler

    def is_done(self):
        return bool(self.handler.is_done())

    def finish(self):
        """The finish code name once done ('Ok', 'Canceled', 'Preempted', ...)."""
        feedback = self.handler.get()
        return feedback.finish_code.name

    def cancel(self):
        if not self.handler.is_done():
            self.handler.cancel()

    def wait(self, timeout_ms):
        """True once the command is done (finished or cancelled), waiting at most timeout_ms."""
        return bool(self.handler.wait_for(int(timeout_ms)))


class BaseStream:
    """An SE(2) velocity command stream; silence (no send within hold_s) stops the base."""

    MAX_RENEWALS = 2

    def __init__(self, robot, priority, accel, hold_s):
        import rby1_sdk as rby
        self.rby, self.robot, self.priority, self.accel, self.hold_s = rby, robot, priority, accel, hold_s
        self.stream = robot.create_command_stream(priority)
        self.renewals = 0

    def _command(self, vx, wz, minimum_time):
        rby = self.rby
        command = (rby.SE2VelocityCommandBuilder()
                   .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(self.hold_s))
                   .set_velocity(np.array([float(vx), 0.0]), float(wz))
                   .set_acceleration_limit(np.array([self.accel[0], self.accel[0]]), self.accel[1])
                   .set_minimum_time(float(minimum_time)))
        return rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_mobility_command(command))

    def send(self, vx, wz, minimum_time):
        """Send one velocity. A stream the controller expired underneath us (a late cancel of the previous
        control, seen right after a previous base move) is replaced and the send repeated, at most
        MAX_RENEWALS times per stream; the caller's loop closes on odometry, so continuing is safe."""
        try:
            self.stream.send_command(self._command(vx, wz, minimum_time), 1000)
        except RuntimeError as exc:
            if 'expired' not in str(exc) or self.renewals >= self.MAX_RENEWALS:
                raise
            self.renewals += 1
            self.stream = self.robot.create_command_stream(self.priority)
            self.stream.send_command(self._command(vx, wz, minimum_time), 1000)

    def close(self, timeout_ms=1000):
        """Zero velocity, then let the hold time end the stream; cancel only if it is still running.

        The SDK's cancel cancels the robot's current control, and it can land after the next command has
        started (a base move sent 10-30 ms after the previous one failed with "This stream is expired").
        Ending by the hold time cancels nothing.
        """
        try:
            if not self.stream.is_done():
                self.send(0.0, 0.0, 0.1)
        finally:
            if not self.stream.wait_for(int(self.hold_s * 1000) + int(timeout_ms)):
                self.stream.cancel()
                self.stream.wait_for(int(timeout_ms))


class SdkRobot:
    def __init__(self, address, *, retries=5, timeout_ms=2000):
        import rby1_sdk as rby
        self.rby = rby
        self.address = address
        self.robot = rby.create_robot_a(address)
        if not self.robot.connect(max_retries=retries, timeout_ms=timeout_ms):
            raise RuntimeError(f'cannot reach the RB-Y1 controller at {address}; is the robot on and the UPC '
                               'on its network?')
        self.handlers = []   # every live command handler (see the module docstring)

    # ----- reading ------------------------------------------------------------------------
    def urdf(self):
        return self.robot.get_robot_model()

    def info(self):
        info = self.robot.get_robot_info()
        return {'robot_model': info.robot_model_name, 'robot_model_version': info.robot_model_version,
                'sdk_version': info.sdk_version, 'version': info.version}

    def state(self):
        s = self.robot.get_state()
        pressed = self.rby.EMOState.State.Pressed
        ft = {}
        for side in ('right', 'left'):
            sensor = getattr(s, f'ft_sensor_{side}')
            ft[side] = {'force': np.asarray(sensor.force, dtype=np.float64).tolist(),
                        'torque': np.asarray(sensor.torque, dtype=np.float64).tolist(),
                        'age_s': _seconds(sensor.time_since_last_update)}
        odometry = np.asarray(s.odometry, dtype=np.float64)
        battery = s.battery_state
        return Snapshot(
            t=time.time(), q=np.asarray(s.position, dtype=np.float64),
            ready=np.asarray(s.is_ready, dtype=bool),
            age_s=np.array([_seconds(j.time_since_last_update) for j in s.joint_states]),
            emo_pressed=any(e.state == pressed for e in s.emo_states), ft=ft,
            odometry=(float(odometry[0, 2]), float(odometry[1, 2]), float(np.arctan2(odometry[1, 0], odometry[0, 0]))),
            battery={'voltage': round(float(battery.voltage), 2), 'current': round(float(battery.current), 2),
                     'level_percent': round(float(battery.level_percent), 1)},
            torque=np.array([float(j.torque) for j in s.joint_states]),
            velocity=np.asarray(s.velocity, dtype=np.float64),
            temperature=np.array([float(j.temperature) for j in s.joint_states]))

    def control_state(self):
        """{'power': bool, 'servo': bool, 'manager': 'Idle'|'Enabled'|'MinorFault'|..., 'control': ...}."""
        cms = self.robot.get_control_manager_state()
        return {'power': bool(self.robot.is_power_on('.*')), 'servo': bool(self.robot.is_servo_on('.*')),
                'manager': cms.state.name, 'control': cms.control_state.name,
                'unlimited_mode': bool(cms.unlimited_mode_enabled)}

    def manager_state(self):
        """The control manager state name alone ('Enabled', 'Idle', 'MinorFault', 'MajorFault', ...)."""
        return self.robot.get_control_manager_state().state.name

    # ----- bring-up -----------------------------------------------------------------------
    def enable(self, unlimited_mode=False, reset_faults=False):
        """Power, servo and control manager on, in the vendor's order; returns the steps taken.

        A faulted control manager is reset only with reset_faults (the operator armed after
        seeing the fault); otherwise RuntimeError. The drives then hold the joints where they
        are; no motion command is sent.
        """
        steps = []
        robot = self.robot
        if not robot.is_power_on('.*'):
            if not robot.power_on('.*'):
                raise RuntimeError('power_on failed')
            steps.append('power_on')
        if not robot.is_servo_on('.*'):
            if not robot.servo_on('.*'):
                raise RuntimeError('servo_on failed')
            steps.append('servo_on')
        state = robot.get_control_manager_state().state
        faults = (self.rby.ControlManagerState.State.MinorFault, self.rby.ControlManagerState.State.MajorFault)
        if state in faults:
            if not reset_faults:
                raise RuntimeError(f'the control manager is in {state.name}; the operator inspects the robot and '
                                   'arms again before it is reset')
            if not robot.reset_fault_control_manager():
                raise RuntimeError(f'the control manager is in {state.name} and the fault reset failed')
            steps.append(f'reset_fault ({state.name})')
        if robot.get_control_manager_state().state != self.rby.ControlManagerState.State.Enabled:
            if not robot.enable_control_manager(unlimited_mode_enabled=bool(unlimited_mode)):
                raise RuntimeError('enable_control_manager failed')
            steps.append('enable_control_manager')
        return steps

    # ----- commands -----------------------------------------------------------------------
    def _joint_builder(self, goal, minimum_time, velocity, acceleration, hold_s):
        rby = self.rby
        n = len(goal)
        return (rby.JointPositionCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(float(hold_s)))
                .set_position(np.asarray(goal, dtype=np.float64))
                .set_velocity_limit(np.full(n, float(velocity)) if np.isscalar(velocity)
                                    else np.asarray(velocity, dtype=np.float64))
                .set_acceleration_limit(np.full(n, float(acceleration)) if np.isscalar(acceleration)
                                        else np.asarray(acceleration, dtype=np.float64))
                .set_minimum_time(float(minimum_time)))

    def joint_command(self, goals, minimum_time, velocity, acceleration, hold_s):
        """One RobotCommandBuilder that moves the given components ({component: goal}) together."""
        rby = self.rby
        unknown = set(goals) - set(COMPONENTS)
        if unknown:
            raise ValueError(f'unknown components {sorted(unknown)}')
        command = rby.ComponentBasedCommandBuilder()
        body = {c: g for c, g in goals.items() if c != 'head'}
        if body:
            builder = rby.BodyComponentBasedCommandBuilder()
            for c, goal in body.items():
                getattr(builder, BODY_SETTERS[c])(self._joint_builder(goal, minimum_time, velocity[c],
                                                                      acceleration[c] if isinstance(acceleration, dict) else acceleration,
                                                                      hold_s))
            command.set_body_command(builder)
        if 'head' in goals:
            command.set_head_command(self._joint_builder(goals['head'], minimum_time, velocity['head'],
                                                         acceleration['head'] if isinstance(acceleration, dict)
                                                         else acceleration, hold_s))
        return rby.RobotCommandBuilder().set_command(command)

    def send_joints(self, goals, *, minimum_time, velocity, acceleration, hold_s, priority):
        return self._send(self.joint_command(goals, minimum_time, velocity, acceleration, hold_s), priority)

    def stop(self, cancel_timeout_ms=500):
        """Cancel every live command and wait until each cancel is done; returns how many were running.

        The controller then decelerates the joints and holds them where they stop (on the robot:
        0.04 rad past the cancel at 0.3 rad/s, then within 0.0001 rad over 3 s). No hold command
        is sent: the SDK's cancel cancels the robot's current control, which took a hold sent
        after it with it, and an accepted hold at the pre-deceleration position would pull the
        joints back, a motion after a stop.
        """
        running = 0
        for handle in self.handlers:
            try:
                if not handle.is_done():
                    running += 1
                handle.cancel()
                handle.wait(cancel_timeout_ms)
            except Exception:
                pass
        self.handlers = []
        return running

    def base_stream(self, *, priority, accel, hold_s):
        return BaseStream(self.robot, priority, accel, hold_s)

    # ----- the controller's own command modes (each an SDK builder, used as shipped) ------
    def controller_command(self, kind, component, spec, *, hold_s, priority):
        """Send one of the controller's command modes for a component; returns a Handle.

        kind: 'cartesian' (CartesianCommandBuilder), 'optimal_control' (OptimalControlCommandBuilder),
        'cartesian_impedance' (CartesianImpedanceControlCommandBuilder), 'link_impedance'
        (ImpedanceControlCommandBuilder), 'joint_impedance' (JointImpedanceControlCommandBuilder),
        'joint_group' (JointGroupPositionCommandBuilder, torso), 'jog' (JogCommandBuilder, one joint),
        'stop' (StopCommandBuilder). spec holds plain values (lists, floats, 4x4 lists); this method only
        maps them onto the builder calls. component: right_arm, left_arm, torso or body (jog / stop: None).
        """
        rby = self.rby
        header = rby.CommandHeaderBuilder().set_control_hold_time(float(hold_s))
        T = lambda m: np.asarray(m, dtype=np.float64)  # noqa: E731
        if kind == 'cartesian':
            b = rby.CartesianCommandBuilder().set_command_header(header).set_minimum_time(float(spec['minimum_time']))
            for t in spec['targets']:
                b.add_target(t['ref_link'], t['link'], T(t['T']), float(t['linear_velocity_limit']),
                             float(t['angular_velocity_limit']), float(t['acceleration_limit_scaling']))
            for j in spec.get('joint_targets', []):
                b.add_joint_position_target(j['joint'], float(j['position']), j.get('velocity_limit'),
                                            j.get('acceleration_limit'))
            b.set_stop_position_tracking_error(float(spec['stop_position_tracking_error']))
            b.set_stop_orientation_tracking_error(float(spec['stop_orientation_tracking_error']))
            b.set_stop_joint_position_tracking_error(float(spec['stop_joint_position_tracking_error']))
        elif kind == 'optimal_control':
            b = rby.OptimalControlCommandBuilder().set_command_header(header)
            for t in spec['targets']:
                b.add_cartesian_target(t['ref_link'], t['link'], T(t['T']), float(t['translation_weight']),
                                       float(t['rotation_weight']))
            for j in spec.get('joint_targets', []):
                b.add_joint_position_target(j['joint'], float(j['position']), float(j['weight']))
            if spec.get('center_of_mass') is not None:
                com = spec['center_of_mass']
                b.set_center_of_mass_target(com['ref_link'], T(com['position']), float(com['weight']))
            (b.set_velocity_limit_scaling(float(spec['velocity_limit_scaling']))
             .set_acceleration_limit_scaling(float(spec['acceleration_limit_scaling']))
             .set_error_scaling(float(spec['error_scaling'])).set_stop_cost(float(spec['stop_cost']))
             .set_min_delta_cost(float(spec['min_delta_cost'])).set_patience(int(spec['patience'])))
        elif kind == 'cartesian_impedance':
            b = (rby.CartesianImpedanceControlCommandBuilder().set_command_header(header)
                 .set_minimum_time(float(spec['minimum_time'])))
            for t in spec['targets']:
                b.add_target(t['ref_link'], t['link'], T(t['T']), float(t['linear_velocity_limit']),
                             float(t['angular_velocity_limit']), float(t['linear_acceleration_limit']),
                             float(t['angular_acceleration_limit']))
            for j in spec.get('joint_targets', []):
                b.add_joint_position_target(j['joint'], float(j['position']), j.get('velocity_limit'),
                                            j.get('acceleration_limit'))
            for name, lower, upper in spec.get('joint_limits', []):
                b.add_joint_limit(name, float(lower), float(upper))
            (b.set_joint_stiffness(T(spec['stiffness'])).set_joint_damping_ratio(float(spec['damping_ratio']))
             .set_joint_torque_limit(T(spec['torque_limit'])).set_reset_reference(bool(spec['reset_reference']))
             .set_stop_position_tracking_error(float(spec['stop_position_tracking_error']))
             .set_stop_orientation_tracking_error(float(spec['stop_orientation_tracking_error']))
             .set_stop_joint_position_tracking_error(float(spec['stop_joint_position_tracking_error'])))
        elif kind == 'link_impedance':
            b = (rby.ImpedanceControlCommandBuilder().set_command_header(header)
                 .set_reference_link_name(spec['ref_link']).set_link_name(spec['link'])
                 .set_transformation(T(spec['T'])).set_translation_weight(T(spec['translation_weight']))
                 .set_rotation_weight(T(spec['rotation_weight'])).set_damping_ratio(float(spec['damping_ratio'])))
        elif kind == 'joint_impedance':
            b = (rby.JointImpedanceControlCommandBuilder().set_command_header(header)
                 .set_position(T(spec['position'])).set_minimum_time(float(spec['minimum_time']))
                 .set_velocity_limit(T(spec['velocity_limit'])).set_acceleration_limit(T(spec['acceleration_limit']))
                 .set_stiffness(T(spec['stiffness'])).set_damping_ratio(float(spec['damping_ratio']))
                 .set_torque_limit(T(spec['torque_limit'])))
        elif kind == 'joint_group':
            b = (rby.JointGroupPositionCommandBuilder().set_command_header(header)
                 .set_joint_names(list(spec['joints'])).set_position(T(spec['position']))
                 .set_minimum_time(float(spec['minimum_time'])).set_velocity_limit(T(spec['velocity_limit']))
                 .set_acceleration_limit(T(spec['acceleration_limit'])))
        elif kind == 'jog':
            mode = {'absolute': rby.JogCommandBuilder.AbsolutePosition, 'relative': rby.JogCommandBuilder.RelativePosition,
                    'one_step': rby.JogCommandBuilder.OneStep}[spec['mode']]
            jog = (rby.JogCommandBuilder().set_command_header(header).set_joint_name(spec['joint'])
                   .set_command(mode(float(spec['value']))).set_velocity_limit(float(spec['velocity_limit']))
                   .set_acceleration_limit(float(spec['acceleration_limit'])))
            return self._send(rby.RobotCommandBuilder().set_command(jog), priority)
        elif kind == 'stop':
            stop = rby.WholeBodyCommandBuilder().set_command(rby.StopCommandBuilder().set_command_header(header))
            return self._send(rby.RobotCommandBuilder().set_command(stop), priority)
        else:
            raise ValueError(f'unknown controller command {kind}')
        return self._send(self._wrap(component, b), priority)

    def _wrap(self, component, builder):
        """A component-level builder inside the RobotCommandBuilder layers the SDK expects."""
        rby = self.rby
        if component == 'body':
            body = rby.BodyCommandBuilder().set_command(builder)
        elif component in ('right_arm', 'left_arm'):
            arm = rby.ArmCommandBuilder().set_command(builder)
            parts = rby.BodyComponentBasedCommandBuilder()
            getattr(parts, f'set_{component}_command')(arm)
            body = rby.BodyCommandBuilder().set_command(parts)
        elif component == 'torso':
            parts = rby.BodyComponentBasedCommandBuilder().set_torso_command(rby.TorsoCommandBuilder().set_command(builder))
            body = rby.BodyCommandBuilder().set_command(parts)
        else:
            raise ValueError(f'unknown component {component}')
        return rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(body))

    def _send(self, command, priority):
        self.handlers = [h for h in self.handlers if not h.is_done()]
        handle = Handle(self.robot.send_command(command, int(priority)))
        self.handlers.append(handle)
        return handle

    # ----- read-only controller information -----------------------------------------------
    def controller_info(self, what, name=None, count=50):
        """Read-only queries of the controller (no setter is reachable from here)."""
        r = self.robot
        if what == 'robot_info':
            info = r.get_robot_info()
            return {'robot_model': info.robot_model_name, 'robot_model_version': info.robot_model_version,
                    'sdk_version': info.sdk_version, 'version': info.version}
        if what == 'control_manager':
            cms = r.get_control_manager_state()
            return {'state': cms.state.name, 'control': cms.control_state.name,
                    'unlimited_mode': bool(cms.unlimited_mode_enabled), 'time_scale': float(r.get_time_scale())}
        if what == 'pid_gains':
            getter = {'torso': r.get_torso_position_pid_gains, 'right_arm': r.get_right_arm_position_pid_gains,
                      'left_arm': r.get_left_arm_position_pid_gains, 'head': r.get_head_position_pid_gains}[name]
            return [{'p': g.p_gain, 'i': g.i_gain, 'd': g.d_gain} for g in getter()]
        if what == 'parameter_list':
            return [{'name': n, 'type': t} for n, t in r.get_parameter_list()]
        if what == 'parameter':
            return {'name': name, 'value': r.get_parameter(name)}
        if what == 'fault_logs':
            return list(r.get_fault_log_list())
        if what == 'log':
            return [str(entry) for entry in r.get_last_log(int(count))]
        raise ValueError(f'unknown controller query {what}')

    def set_tool_flange_output_voltage(self, side, volts):
        return self.robot.set_tool_flange_output_voltage(side, int(volts))

    def close(self):
        self.stop()
        try:
            self.robot.disconnect()
        except Exception:
            pass
