"""The RB-Y1 controller's own command modes as MCP tools: wired, not reimplemented.

Each tool maps its arguments onto one rby1_sdk command builder, used as shipped
(`SdkRobot.controller_command`): Cartesian (CartesianCommandBuilder), whole-body optimal
control (OptimalControlCommandBuilder), Cartesian impedance
(CartesianImpedanceControlCommandBuilder), link impedance (ImpedanceControlCommandBuilder),
joint impedance (JointImpedanceControlCommandBuilder), torso joint group
(JointGroupPositionCommandBuilder), jog (JogCommandBuilder) and the stop command
(StopCommandBuilder). The robot's controller solves the IK and the path; the backend
solves nothing itself.

The agent chooses what to do (mode, targets, a speed below the cap), never how the
controller behaves: impedance gains, torque limits and solver settings (stop costs,
patience, scaling, tracking tolerances) are backend options from the vendor examples,
set by the operator. Nothing here writes a robot parameter, and gravity compensation is
not wired.

Because the controller picks the joint path, the checks before a command are on its
targets (height, travel, reach, joint limits) and the checks while it runs are on the
measured state every control period: arming, EMO, fresh state, drives ready, torque and
wrist force (`_guard`), and an envelope (joints moving towards a limit, the squat cap, the
upright torso, end-effector height, new self-collisions). A command ends with the joints
re-targeted where they are, so whatever the controller holds afterwards is a joint hold
at the measured position.

Characterized on the robot on 2026-10-08 (scripts/rby1_controller_check.py, RBY1.md) and on
by default (`controller_commands`); `controller_tools_disabled` lists the modes that failed
there: link impedance with the vendor's gains overloaded the wrist (202 % of its rating in
1.3 s) and Cartesian impedance ended 66 mm from a 3 cm target. Re-enabling them means new
gains (operator options) and a new characterization.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import COMPONENTS, EE_FRAME, JOINT_NAMES, SHOULDER_FRAME, component, matrix, pose

TORSO_LINK = 'link_torso_5'
TARGET_LINKS = ('ee_right', 'ee_left', TORSO_LINK)
LINK_COMPONENT = {'ee_right': 'right_arm', 'ee_left': 'left_arm', TORSO_LINK: 'torso'}
REACH_M = 0.69
ARM_ONLY = ('right_arm', 'left_arm')

CONTROLLER_DEFAULTS = {
    'controller_commands': True,                # list the controller-mode tools (characterized 2026-10-08)
    'controller_tools_disabled': ['rby1_cartesian_impedance', 'rby1_link_impedance'],  # failed on the robot
    'cartesian_linear_velocity_mps': 0.1,       # cap (and default) of Cartesian hand / torso speed
    'cartesian_angular_velocity_radps': 0.3,
    'cartesian_acceleration_limit_scaling': 0.3,
    'cartesian_stop_position_error_m': 1e-3,    # vendor 09_demo_motion: 1e-3 m
    'cartesian_stop_orientation_error_rad': 1e-3,
    'cartesian_stop_joint_error_rad': 1e-2,
    # Below this smallest arm-Jacobian singular value a Cartesian arm move is refused: the controller
    # faulted (MinorFault) on one from 0.016 (left elbow at -0.38 rad); 0.14 worked (2026-10-09).
    'cartesian_min_singular_value': 0.04,
    'cartesian_max_translation_m': 0.25,        # per call, any target link
    'cartesian_max_rotation_rad': 1.0,
    'cartesian_max_torso_translation_m': 0.1,
    'optimal_control_velocity_limit_scaling': 0.2,
    'optimal_control_acceleration_limit_scaling': 0.2,
    'optimal_control_error_scaling': 1.0,
    'optimal_control_stop_cost': 2e-3,          # vendor 09_demo_motion: WEIGHT^2 * 2e-3
    'optimal_control_min_delta_cost': 2e-3,
    'optimal_control_patience': 10,
    'optimal_control_max_weight': 10.0,
    'optimal_control_upright_weight': 10.0,     # torso_0 = torso_4 = 0 targets added for the upright rule
    'impedance_stiffness_nm_per_rad': 100.0,    # vendor 22_joint_impedance_control: 100
    'impedance_damping_ratio': 1.0,
    'impedance_torque_fraction': 0.6,           # torque limit per joint: this fraction of its rating
    'impedance_linear_acceleration_limit': 0.5,
    'impedance_angular_acceleration_limit': 1.0,
    'link_impedance_translation_weight': [3000.0, 3000.0, 3000.0],  # vendor 07_impedance_control (x, y)
    'link_impedance_rotation_weight': [50.0, 50.0, 50.0],
    'link_impedance_max_seconds': 30.0,
    'jog_max_relative_rad': 0.3,
    'envelope_margin_rad': 0.03,                # a joint moving closer than this to its URDF limit stops
    'controller_command_timeout_s': 30.0,
}
CONTROLLER_TOOLS = ('rby1_cartesian', 'rby1_optimal_control', 'rby1_cartesian_impedance', 'rby1_link_impedance',
                    'rby1_joint_impedance', 'rby1_joint_group', 'rby1_jog')


def _schema(properties=None, required=()):
    return {'type': 'object', 'properties': properties or {}, 'required': list(required),
            'additionalProperties': False}


NUMBER = {'type': 'number'}
VEC3 = {'type': 'array', 'items': NUMBER, 'minItems': 3, 'maxItems': 3}
QUAT = {'type': 'array', 'items': NUMBER, 'minItems': 4, 'maxItems': 4}
TARGET = {'link': {'type': 'string', 'enum': list(TARGET_LINKS)}, 'position': VEC3, 'quaternion': QUAT,
          'relative': {'type': 'boolean'}}


class ControllerModes:
    """Mixin for Rby1Backend: tool definitions, checks and supervised execution of the controller modes."""

    # ----- definitions --------------------------------------------------------------------
    def _controller_tools(self):
        o = self.options
        if not o['controller_commands']:
            return []
        disabled = set(o['controller_tools_disabled'])
        return [tool for tool in self._controller_tool_list(o) if tool['name'] not in disabled]

    def _controller_tool_list(self, o):
        weight = {'type': 'number', 'minimum': 0, 'maximum': o['optimal_control_max_weight']}
        speed = (f"at most {o['cartesian_linear_velocity_mps']} m/s and {o['cartesian_angular_velocity_radps']} "
                 'rad/s')
        joint_target = _schema({'joint': {'type': 'string'}, 'position': NUMBER}, ['joint', 'position'])
        return [
            {'name': 'rby1_cartesian', 'description':
                "The controller's Cartesian command (CartesianCommandBuilder): drive end effectors and/or the "
                f'torso link ({", ".join(TARGET_LINKS)}) to base-frame poses; the controller solves the IK and the '
                'path on the robot. component: right_arm / left_arm (that arm only), torso, or body (torso and both '
                'arms together, whole-body). quaternion omitted keeps the link orientation; relative adds position. '
                f'Optional joint_targets pin joints of that component. Speed {speed}; at most '
                f"{o['cartesian_max_translation_m']} m (torso {o['cartesian_max_torso_translation_m']} m) and "
                f"{o['cartesian_max_rotation_rad']} rad per target per call. Checked on its targets before and on "
                'the measured joints while it runs (limits, squat cap, upright torso, hand height, self-collision, '
                'torque, force). Measured: 5 cm in 2.4 s, final error 0.1-0.2 mm, path within 1 mm of the straight '
                'line. Requires operator arming.',
             'inputSchema': _schema({'component': {'type': 'string', 'enum': ['right_arm', 'left_arm', 'torso', 'body']},
                                     'targets': {'type': 'array', 'minItems': 1, 'maxItems': 3,
                                                 'items': _schema(TARGET, ['link', 'position'])},
                                     'joint_targets': {'type': 'array', 'maxItems': 10, 'items': joint_target},
                                     'linear_velocity': {'type': 'number', 'minimum': 0.01,
                                                         'maximum': o['cartesian_linear_velocity_mps']},
                                     'angular_velocity': {'type': 'number', 'minimum': 0.05,
                                                          'maximum': o['cartesian_angular_velocity_radps']}},
                                    ['component', 'targets'])},
            {'name': 'rby1_optimal_control', 'description':
                "The controller's whole-body optimal control (OptimalControlCommandBuilder): one weighted problem "
                'over the torso and both arms (component body) or the torso: link pose targets with translation / '
                'rotation weights, joint targets with weights, and an optional centre-of-mass target (base frame). '
                'The controller solves it on the robot and stops when the cost settles. The upright torso rule is '
                f"added as torso_0 = torso_4 = 0 targets (weight {o['optimal_control_upright_weight']}). Solver "
                'settings and speed scaling are fixed by the backend. Same checks as rby1_cartesian. Measured: '
                '4-5 cm hand targets in 3.8 s ending within 2-3.3 mm, 10 cm in 7.7 s within 0.2 mm with the torso '
                'taking part; hand paths within 1 mm of straight. Requires operator arming.',
             'inputSchema': _schema({'component': {'type': 'string', 'enum': ['body', 'torso']},
                                     'targets': {'type': 'array', 'maxItems': 3, 'items': _schema(
                                         {**TARGET, 'translation_weight': weight, 'rotation_weight': weight},
                                         ['link', 'position'])},
                                     'joint_targets': {'type': 'array', 'maxItems': 10, 'items': _schema(
                                         {'joint': {'type': 'string'}, 'position': NUMBER, 'weight': weight},
                                         ['joint', 'position'])},
                                     'center_of_mass': _schema({'position': VEC3, 'weight': weight}, ['position'])},
                                    ['component'])},
            {'name': 'rby1_cartesian_impedance', 'description':
                "The controller's Cartesian impedance control (CartesianImpedanceControlCommandBuilder) for one arm: "
                'the hand goes to a base-frame pose compliantly, yielding to contact instead of pushing hard. Joint '
                f"stiffness {o['impedance_stiffness_nm_per_rad']:g} N m/rad, damping ratio "
                f"{o['impedance_damping_ratio']:g}, torque limit {o['impedance_torque_fraction']:.0%} of each "
                'joint rating (backend settings). Use it for the last centimetres into contact; it is not '
                f'accurate under load. Speed {speed}. Requires operator arming.',
             'inputSchema': _schema({'arm': {'type': 'string', 'enum': ['right', 'left']}, 'position': VEC3,
                                     'quaternion': QUAT, 'relative': {'type': 'boolean'}}, ['arm', 'position'])},
            {'name': 'rby1_link_impedance', 'description':
                "The controller's link impedance control (ImpedanceControlCommandBuilder) for one arm: the hand is "
                'held at a base-frame pose like a spring for `seconds`, then held in joint space where it ended. '
                'Stiffness and damping are backend settings. Requires operator arming.',
             'inputSchema': _schema({'arm': {'type': 'string', 'enum': ['right', 'left']}, 'position': VEC3,
                                     'quaternion': QUAT, 'relative': {'type': 'boolean'},
                                     'seconds': {'type': 'number', 'minimum': 0.5,
                                                 'maximum': o['link_impedance_max_seconds']}},
                                    ['arm', 'position', 'seconds'])},
            {'name': 'rby1_joint_impedance', 'description':
                "The controller's joint impedance control (JointImpedanceControlCommandBuilder) for one arm: a "
                'joint-space move that yields to contact (stiffness, damping, torque limits are backend settings). '
                'Same targets and limits as rby1_move_joints. Requires operator arming.',
             'inputSchema': _schema({'arm': {'type': 'string', 'enum': ['right', 'left']},
                                     'positions': {'type': 'array', 'items': NUMBER, 'minItems': 7, 'maxItems': 7},
                                     'preset': {'type': 'string'}, 'relative': {'type': 'boolean'}}, ['arm'])},
            {'name': 'rby1_joint_group', 'description':
                "The controller's joint group position command (JointGroupPositionCommandBuilder, torso only): "
                'move some torso joints by name while the others hold. The resulting torso must keep the upright '
                'rule and the squat cap. Requires operator arming.',
             'inputSchema': _schema({'joints': {'type': 'array', 'minItems': 1, 'maxItems': 6,
                                                'items': {'type': 'string', 'enum': list(JOINT_NAMES[2:8])}},
                                     'positions': {'type': 'array', 'items': NUMBER, 'minItems': 1, 'maxItems': 6}},
                                    ['joints', 'positions'])},
            {'name': 'rby1_jog', 'description':
                "The controller's jog command (JogCommandBuilder) for one joint: absolute position, relative step "
                f"(at most {o['jog_max_relative_rad']} rad), or one_step. Speed and acceleration are that part's "
                'caps. Path checked like rby1_move_joints. Requires operator arming.',
             'inputSchema': _schema({'joint': {'type': 'string', 'enum': list(JOINT_NAMES[2:])},
                                     'mode': {'type': 'string', 'enum': ['absolute', 'relative', 'one_step']},
                                     'value': NUMBER}, ['joint', 'mode', 'value'])},
        ]

    # ----- targets (checks before anything moves) -----------------------------------------
    def _link_pose_now(self, q, link):
        return self.kinematics.frame(link, q)

    def _resolve_target(self, q, spec, component_name):
        """Base-frame 4x4 target for one link, after the boundary checks."""
        o = self.options
        link = spec['link']
        if LINK_COMPONENT[link] != component_name and component_name not in ('body',) and not (
                component_name == 'torso' and link == TORSO_LINK):
            raise ValueError(f'{link} is not part of component {component_name}')
        T_now = self._link_pose_now(q, link)
        p_now, quat_now = pose(T_now)
        p = np.asarray(spec['position'], dtype=np.float64)
        if not np.all(np.isfinite(p)):
            raise ValueError('position must be finite')
        if spec.get('relative'):
            p = p_now + p
        quat = np.asarray(spec.get('quaternion', quat_now), dtype=np.float64)
        if not np.isfinite(np.linalg.norm(quat)) or np.linalg.norm(quat) < 0.5:
            raise ValueError('quaternion must be a unit quaternion [x, y, z, w]')
        quat = quat / np.linalg.norm(quat)
        travel = float(np.linalg.norm(p - p_now))
        cap = o['cartesian_max_torso_translation_m'] if link == TORSO_LINK else o['cartesian_max_translation_m']
        if travel > cap:
            raise ValueError(f'{link} target is {travel:.3f} m away (at most {cap} m per call); split the move')
        turn = float((Rotation.from_quat(quat) * Rotation.from_quat(quat_now).inv()).magnitude())
        if turn > o['cartesian_max_rotation_rad']:
            raise ValueError(f"{link} target turns {turn:.2f} rad (at most {o['cartesian_max_rotation_rad']} rad per "
                             'call); split the move')
        if link != TORSO_LINK:
            if p[2] < o['ee_min_z_m']:
                raise ValueError(f"{link} target z {p[2]:.3f} m is below {o['ee_min_z_m']} m")
            if component_name in ARM_ONLY:  # the shoulder stays where it is
                side = 'right' if link == 'ee_right' else 'left'
                shoulder = self.kinematics.frame(SHOULDER_FRAME[side], q)[:3, 3]
                if np.linalg.norm(p - shoulder) > REACH_M:
                    raise ValueError(f'out of reach for the arm alone: {np.linalg.norm(p - shoulder):.2f} m from the '
                                     f'{side} shoulder (reach {REACH_M} m); use component body, the torso or the base')
        return {'link': link, 'T': matrix(p, quat), 'position': p, 'quaternion': quat, 'travel_m': travel}

    def _joint_targets(self, q, items, component_name, weighted=False):
        names = {JOINT_NAMES[i] for c in (('torso', 'right_arm', 'left_arm') if component_name == 'body'
                                          else (component_name,)) for i in range(*COMPONENTS[c])}
        out = []
        for item in items or []:
            joint, value = item['joint'], float(item['position'])
            if joint not in names:
                raise ValueError(f'{joint} is not a joint of component {component_name}')
            index = JOINT_NAMES.index(joint)
            comp = next(c for c, (a, b) in COMPONENTS.items() if a <= index < b)
            lower, upper = self._limits(comp)
            k = index - COMPONENTS[comp][0]
            if not lower[k] <= value <= upper[k]:
                raise ValueError(f'{joint} target {value:.3f} outside [{lower[k]:.3f}, {upper[k]:.3f}]')
            out.append({'joint': joint, 'position': value, **({'weight': float(item.get('weight', 1.0))}
                                                               if weighted else {})})
        return out

    def _upright_targets(self, component_name, weighted):
        if component_name not in ('torso', 'body') or not self.options['torso_guard']:
            return []
        extra = {'weight': self.options['optimal_control_upright_weight']} if weighted else {}
        return [{'joint': 'torso_0', 'position': 0.0, **extra}, {'joint': 'torso_4', 'position': 0.0, **extra}]

    def _controller_plan(self, name, arguments, q):
        """(kind, component, spec, moving components, summary) of a controller-mode call; ValueError if refused."""
        o = self.options
        if name in ('rby1_cartesian', 'rby1_optimal_control'):
            comp = arguments['component']
            targets = [self._resolve_target(q, t, comp) for t in arguments.get('targets', [])]
            links = [t['link'] for t in targets]
            if len(set(links)) != len(links):
                raise ValueError('each link at most once')
            weighted = name == 'rby1_optimal_control'
            joints = self._joint_targets(q, arguments.get('joint_targets'), comp, weighted) + \
                self._upright_targets(comp, weighted)
            moving = ('torso', 'right_arm', 'left_arm') if comp == 'body' else (comp,)
            if name == 'rby1_cartesian':
                for side in ('right', 'left') if comp in ('right_arm', 'left_arm') else ():
                    if comp != f'{side}_arm':
                        continue
                    sigma = self.kinematics.arm_singular_value(side, q)
                    if sigma < o['cartesian_min_singular_value']:
                        raise ValueError(
                            f'the {side} arm is near a singular pose (elbow {side}_arm_3 = '
                            f'{component(q, comp)[3]:+.2f} rad, smallest Jacobian singular value {sigma:.3f} < '
                            f"{o['cartesian_min_singular_value']}): the controller faults on a Cartesian move from "
                            'here. Bend the elbow with a joint move first, or move with rby1_move_eef / '
                            'rby1_move_joints')
                linear = float(arguments.get('linear_velocity', o['cartesian_linear_velocity_mps']))
                angular = float(arguments.get('angular_velocity', o['cartesian_angular_velocity_radps']))
                travel = max([t['travel_m'] for t in targets] + [0.0])
                spec = {'targets': [{'ref_link': 'base', 'link': t['link'], 'T': t['T'].tolist(),
                                     'linear_velocity_limit': linear, 'angular_velocity_limit': angular,
                                     'acceleration_limit_scaling': o['cartesian_acceleration_limit_scaling']}
                                    for t in targets],
                        'joint_targets': joints, 'minimum_time': max(1.0, travel / linear),
                        'stop_position_tracking_error': o['cartesian_stop_position_error_m'],
                        'stop_orientation_tracking_error': o['cartesian_stop_orientation_error_rad'],
                        'stop_joint_position_tracking_error': o['cartesian_stop_joint_error_rad']}
                return 'cartesian', comp, spec, moving, targets
            if not targets and not arguments.get('joint_targets') and not arguments.get('center_of_mass'):
                raise ValueError('give at least one target, joint target or center_of_mass')
            com = arguments.get('center_of_mass')
            if com is not None and not np.all(np.isfinite(com['position'])):
                raise ValueError('center_of_mass position must be finite')
            spec = {'targets': [{'ref_link': 'base', 'link': t['link'], 'T': t['T'].tolist(),
                                 'translation_weight': float(a.get('translation_weight', 1.0)),
                                 'rotation_weight': float(a.get('rotation_weight', 1.0))}
                                for t, a in zip(targets, arguments.get('targets', []))],
                    'joint_targets': joints,
                    'center_of_mass': None if com is None else {'ref_link': 'base', 'position': list(com['position']),
                                                                'weight': float(com.get('weight', 1.0))},
                    'velocity_limit_scaling': o['optimal_control_velocity_limit_scaling'],
                    'acceleration_limit_scaling': o['optimal_control_acceleration_limit_scaling'],
                    'error_scaling': o['optimal_control_error_scaling'], 'stop_cost': o['optimal_control_stop_cost'],
                    'min_delta_cost': o['optimal_control_min_delta_cost'], 'patience': o['optimal_control_patience']}
            return 'optimal_control', comp, spec, moving, targets
        if name in ('rby1_cartesian_impedance', 'rby1_link_impedance'):
            arm = arguments['arm']
            comp = f'{arm}_arm'
            target = self._resolve_target(q, {'link': EE_FRAME[arm], 'position': arguments['position'],
                                              'quaternion': arguments.get('quaternion'),
                                              'relative': arguments.get('relative', False)}
                                          if arguments.get('quaternion') is not None else
                                          {'link': EE_FRAME[arm], 'position': arguments['position'],
                                           'relative': arguments.get('relative', False)}, comp)
            if name == 'rby1_link_impedance':
                spec = {'ref_link': 'base', 'link': EE_FRAME[arm], 'T': target['T'].tolist(),
                        'translation_weight': o['link_impedance_translation_weight'],
                        'rotation_weight': o['link_impedance_rotation_weight'],
                        'damping_ratio': o['impedance_damping_ratio'], 'seconds': float(arguments['seconds'])}
                return 'link_impedance', comp, spec, (comp,), [target]
            linear, angular = o['cartesian_linear_velocity_mps'], o['cartesian_angular_velocity_radps']
            spec = {'targets': [{'ref_link': 'base', 'link': EE_FRAME[arm], 'T': target['T'].tolist(),
                                 'linear_velocity_limit': linear, 'angular_velocity_limit': angular,
                                 'linear_acceleration_limit': o['impedance_linear_acceleration_limit'],
                                 'angular_acceleration_limit': o['impedance_angular_acceleration_limit']}],
                    'joint_limits': self._impedance_joint_limits(comp),
                    'stiffness': [o['impedance_stiffness_nm_per_rad']] * 7, 'damping_ratio': o['impedance_damping_ratio'],
                    'torque_limit': self._impedance_torque_limit(comp), 'reset_reference': True,
                    'minimum_time': max(1.0, target['travel_m'] / linear),
                    'stop_position_tracking_error': o['cartesian_stop_position_error_m'],
                    'stop_orientation_tracking_error': o['cartesian_stop_orientation_error_rad'],
                    'stop_joint_position_tracking_error': o['cartesian_stop_joint_error_rad']}
            return 'cartesian_impedance', comp, spec, (comp,), [target]
        if name == 'rby1_joint_impedance':
            comp = f"{arguments['arm']}_arm"
            goal = self._joint_goal(q, comp, arguments.get('positions'), arguments.get('preset'),
                                    arguments.get('relative', False))
            self._check_path(q, {comp: goal})
            step = float(np.max(np.abs(goal - component(q, comp))))
            spec = {'position': goal.tolist(), 'minimum_time': self._duration(step, comp),
                    'velocity_limit': [self._speed(comp)] * 7, 'acceleration_limit': [self._accel(comp)] * 7,
                    'stiffness': [o['impedance_stiffness_nm_per_rad']] * 7, 'damping_ratio': o['impedance_damping_ratio'],
                    'torque_limit': self._impedance_torque_limit(comp)}
            return 'joint_impedance', comp, spec, (comp,), {comp: goal}
        if name == 'rby1_joint_group':
            joints, values = list(arguments['joints']), [float(v) for v in arguments['positions']]
            if len(joints) != len(values) or len(set(joints)) != len(joints):
                raise ValueError('give one position per distinct joint')
            torso = component(q, 'torso').copy()
            for joint, value in zip(joints, values):
                torso[JOINT_NAMES.index(joint) - COMPONENTS['torso'][0]] = value
            goal = self._joint_goal(q, 'torso', torso.tolist())
            self._check_path(q, {'torso': goal})
            step = float(np.max(np.abs(goal - component(q, 'torso'))))
            spec = {'joints': joints, 'position': values, 'minimum_time': self._duration(step, 'torso'),
                    'velocity_limit': [self._speed('torso')] * len(joints),
                    'acceleration_limit': [self._accel('torso')] * len(joints)}
            return 'joint_group', 'torso', spec, ('torso',), {'torso': goal}
        if name == 'rby1_jog':
            joint, mode, value = arguments['joint'], arguments['mode'], float(arguments['value'])
            index = JOINT_NAMES.index(joint)
            comp = next(c for c, (a, b) in COMPONENTS.items() if a <= index < b)
            current = float(q[index])
            if mode == 'absolute':
                target = value
            else:
                if abs(value) > o['jog_max_relative_rad']:
                    raise ValueError(f"jog step {value:.3f} rad exceeds {o['jog_max_relative_rad']} rad")
                target = current + value
            values = component(q, comp).copy()
            values[index - COMPONENTS[comp][0]] = target
            goal = self._joint_goal(q, comp, values.tolist())
            self._check_path(q, {comp: goal})
            spec = {'joint': joint, 'mode': mode, 'value': value, 'velocity_limit': self._speed(comp),
                    'acceleration_limit': self._accel(comp)}
            return 'jog', None, spec, (comp,), {comp: goal}
        raise ValueError(f'{name} is not a controller-mode tool')

    def _impedance_torque_limit(self, comp):
        start, stop = COMPONENTS[comp]
        return (self.kinematics.torque_limit[start:stop] * self.options['impedance_torque_fraction']).tolist()

    def _impedance_joint_limits(self, comp):
        lower, upper = self._limits(comp)
        start = COMPONENTS[comp][0]
        return [(JOINT_NAMES[start + k], float(lower[k]), float(upper[k])) for k in range(len(lower))]

    # ----- execution ----------------------------------------------------------------------
    def _envelope(self, snap, start, moving):
        """MotionStopped if the measured state leaves the envelope while the controller drives it."""
        from .backend import MotionStopped
        o, kin = self.options, self.kinematics
        margin = o['envelope_margin_rad']
        for comp in moving:
            a, b = COMPONENTS[comp]
            for i in range(a, b):
                room = min(snap.q[i] - kin.lower[i], kin.upper[i] - snap.q[i])
                was = min(start[i] - kin.lower[i], kin.upper[i] - start[i])
                if room < margin and room < was - 0.002:
                    raise MotionStopped(f'{JOINT_NAMES[i]} at {snap.q[i]:.3f} rad came within {margin} rad of its '
                                        'limit; command cancelled', relieve=True)
        if 'torso' in moving:
            t = component(snap.q, 'torso')
            cap = o['torso_max_squat_rad']
            if t[1] > cap + 0.02 or t[2] < -2 * cap - 0.04:
                raise MotionStopped(f'torso passed the squat cap (torso_1 {t[1]:.3f}, torso_2 {t[2]:.3f}); cancelled',
                                    relieve=True)
            if o['torso_guard'] and (abs(t[0]) > 0.05 or abs(t[4]) > 0.05):
                raise MotionStopped(f'torso left the upright rule (torso_0 {t[0]:.3f}, torso_4 {t[4]:.3f}); cancelled',
                                    relieve=True)
        if {'torso', 'right_arm', 'left_arm'} & set(moving):
            frames = kin.frames(list(EE_FRAME.values()), snap.q)
            for side, frame in EE_FRAME.items():
                z = float(frames[frame][2, 3])
                if z < o['ee_min_z_m']:
                    raise MotionStopped(f"{side} end effector at z={z:.3f} m, below {o['ee_min_z_m']} m; cancelled",
                                        relieve=True)

    def _supervise(self, handle, moving, *, seconds=None):
        """Watch a controller command every control period until it finishes (or for `seconds`)."""
        from .backend import ARMS, MotionStopped
        o = self.options
        period = 1.0 / o['control_hz']
        snap = self._guard(moving)
        start_q = snap.q.copy()
        ft_start = {side: snap.ft[side]['force'] for side in ARMS}
        initial = self.kinematics.collisions(start_q)
        started = time.monotonic()
        deadline = started + (seconds if seconds is not None else o['controller_command_timeout_s'])
        tick = 0
        while (seconds is not None and time.monotonic() < deadline) or (seconds is None and not handle.is_done()):
            snap = self._guard(moving, ft_start)
            self._envelope(snap, start_q, moving)
            tick += 1
            if tick % 5 == 0:
                new = self.kinematics.collisions(snap.q) - initial
                if new:
                    raise MotionStopped(f'self-collision {sorted(new)} while the controller moved; cancelled',
                                        relieve=True)
            if seconds is None and time.monotonic() > deadline:
                raise MotionStopped(f"the controller command did not finish within {o['controller_command_timeout_s']:g} "
                                    's; cancelled', relieve=True)
            time.sleep(period)
        finish = 'Timed' if seconds is not None else handle.finish()
        if seconds is None and finish not in ('Ok', 'Unknown'):
            raise MotionStopped(f"the robot's controller ended the command with {finish}", relieve=True)
        return finish, round(time.monotonic() - started, 2)

    def _controller_call(self, name, arguments):
        from .backend import MotionStopped
        o = self.options
        with self.lock:
            snap = self._require_armed()
            kind, comp, spec, moving, planned = self._controller_plan(name, arguments, snap.q)
            seconds = spec.pop('seconds', None)
            try:
                handle = self.robot.controller_command(kind, comp, spec, hold_s=max(o['control_hold_time_s'],
                                                                                   seconds or 0.0),
                                                       priority=o['priority'])
                finish, elapsed = self._supervise(handle, moving, seconds=seconds)
                self._retarget_here()  # end every controller command in a joint hold at the measured position
            except MotionStopped as exc:
                raise self._held(exc)
            except Exception as exc:
                raise self._failed(exc)
            now = self._snapshot().q
            result = {'mode': kind, 'component': comp, 'finish': finish, 'seconds': elapsed}
            if isinstance(planned, list):
                frames = self.kinematics.frames([t['link'] for t in planned], now)
                result['targets'] = [{'link': t['link'], 'target_position': np.round(t['position'], 4).tolist(),
                                      'measured_position': np.round(frames[t['link']][:3, 3], 4).tolist(),
                                      'position_error_mm': round(float(np.linalg.norm(
                                          frames[t['link']][:3, 3] - t['position'])) * 1000, 1)}
                                     for t in planned]
            else:
                result['joint_error_rad'] = round(max(float(np.max(np.abs(component(now, c) - g)))
                                                      for c, g in planned.items()), 4)
            self.last_motion = {'tool': name, 'at': time.time(), 'mode': kind}
            result.update(self._state_report())
            return result, []
