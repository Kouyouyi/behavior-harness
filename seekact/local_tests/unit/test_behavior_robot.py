"""CPU-only contracts: no Isaac initialization and no real robot execution."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import unittest
from unittest.mock import patch

from services.behavior_ap.robot import (
    ACTION_INDICES, R1ProControl, damped_least_squares, orientation_error, quaternion_matrix,
)


def controller(kind, **values):
    return type(kind, (), values)()


class FakeRobot:
    action_dim = 23
    _action_normalize = False
    controller_action_idx = {name: np.array(idx) for name, idx in ACTION_INDICES.items()}
    trunk_control_idx = np.arange(6, 10)
    arm_control_idx = {'left': np.arange(10, 17), 'right': np.arange(17, 24)}
    gripper_control_idx = {'left': np.arange(24, 26), 'right': np.arange(26, 28)}
    eef_link_names = {'left': 'left_eef_link', 'right': 'right_eef_link'}
    _controllers = {name: (name, 0) for name in ACTION_INDICES}

    def __init__(self):
        self.q = np.zeros(28)
        self.q[24:] = .05
        self.joint_lower_limits = np.r_[np.full(24, -2.), np.zeros(4)]
        self.joint_upper_limits = np.r_[np.full(24, 2.), np.full(4, .05)]
        self.position = np.zeros(3)
        self.orientation = np.array([0., 0., 0., 1.])
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.eef_links = {side: SimpleNamespace(get_position_orientation=lambda side=side: self.eef_world(side))
                          for side in ('left', 'right')}

    def get_joint_positions(self):
        return self.q.copy()

    def get_joint_velocities(self):
        return np.zeros(28)

    def get_position_orientation(self):
        return self.position.copy(), self.orientation.copy()

    def get_relative_eef_pose(self, side):
        # Linear, redundant mock kinematics. Only a contract test, not an R1Pro model.
        return self.q[self.arm_control_idx[side]][:3].copy(), np.array([0, 0, 0, 1.])

    def eef_world(self, side):
        p, _ = self.get_relative_eef_pose(side)
        return quaternion_matrix(self.orientation) @ p + self.position, self.orientation.copy()

    def get_linear_velocity(self):
        return self.linear_velocity.copy()

    def get_angular_velocity(self):
        return self.angular_velocity.copy()


class FakeEnvironment:
    def __init__(self):
        self.steps, self.done = 0, False
        self.calls = []
        self.terminal_at = None
        self.robot = FakeRobot()
        groups = {}
        for name, idx in [('trunk', self.robot.trunk_control_idx),
                          ('arm_left', self.robot.arm_control_idx['left']),
                          ('arm_right', self.robot.arm_control_idx['right'])]:
            groups[name] = controller('JointController', motor_type='position', use_delta_commands=False,
                                      command_input_limits=None, command_output_limits=None, dof_idx=idx,
                                      routing_path='/robot', view_row_indices=np.array([0]))
        groups['base'] = controller('HolonomicBaseJointController', motor_type='velocity',
                                    use_delta_commands=False, command_input_limits=([-1]*3, [1]*3),
                                    command_output_limits=([-.75, -.75, -1], [.75, .75, 1]))
        for side in ('left', 'right'):
            groups[f'gripper_{side}'] = controller('MultiFingerGripperController', _motor_type='position',
                                                  _mode='smooth', _inverted=False,
                                                  command_input_limits=([-1], [1]), command_output_limits=([0], [.05]))
        self.backend = SimpleNamespace(robot=self.robot,
                                       og=SimpleNamespace(sim=SimpleNamespace(get_sim_step_dt=lambda: 1/30)),
                                       ControllerView=SimpleNamespace(_controller_groups=groups),
                                       _make_hold_action=self.hold)

    def _require_ready(self, **kwargs):
        pass

    def hold(self, robot):
        result = np.zeros(23)
        result[3:7] = robot.q[6:10]
        result[7:14] = robot.q[10:17]
        result[15:22] = robot.q[17:24]
        result[14] = 2 * robot.q[24:26].mean() / .05 - 1
        result[22] = 2 * robot.q[26:28].mean() / .05 - 1
        return result

    def step(self, action):
        self.calls.append(action.copy())
        robot = self.robot
        robot.q[6:10], robot.q[10:17], robot.q[17:24] = action[3:7], action[7:14], action[15:22]
        robot.q[24:26], robot.q[26:28] = (action[14] + 1) * .025, (action[22] + 1) * .025
        robot.linear_velocity = quaternion_matrix(robot.orientation) @ (np.r_[action[:2] * .75, 0.])
        robot.angular_velocity = np.r_[0., 0., action[2]]
        robot.position += robot.linear_velocity / 30
        self.steps += 1
        if self.terminal_at is not None and self.steps >= self.terminal_at:
            self.done = True


def make_rig():
    env = FakeEnvironment()
    control = R1ProControl(env, env.step)
    control._arm_jacobian = lambda side: np.c_[np.eye(6), np.zeros(6)]
    return env, control


class R1ProControlTests(unittest.TestCase):
    def test_dls_singular_and_bounded(self):
        self.assertTrue(np.array_equal(damped_least_squares(np.zeros((6, 7)), np.ones(6)), np.zeros(7)))
        delta = damped_least_squares(np.c_[np.eye(6), np.zeros(6)], np.ones(6) * 100)
        self.assertEqual(delta.shape, (7,))
        self.assertLessEqual(np.max(np.abs(delta)), .04 + 1e-12)
        self.assertTrue(np.all(delta[:6] > 0))

    def test_dls_rejects_invalid(self):
        cases = [([[np.nan]], [0], .1), ([[1]], [0, 1], .1), ([[1]], [1], 0), ([[1]], [1], np.inf)]
        for jac, error, damping in cases:
            with self.subTest(jac=jac, error=error, damping=damping), self.assertRaises(ValueError):
                damped_least_squares(jac, error, damping=damping)

    def test_orientation_error_handles_pi_sign_and_reference_frame(self):
        self.assertTrue(np.allclose(orientation_error([1, 0, 0, 0], [0, 0, 0, 1]), [np.pi, 0, 0]))
        self.assertTrue(np.allclose(orientation_error([0, 0, 0, -1], [0, 0, 0, 1]), 0))
        quarter = [0, 0, np.sqrt(.5), np.sqrt(.5)]
        self.assertTrue(np.allclose(quaternion_matrix(quarter) @ [1, 0, 0], [0, 1, 0], atol=1e-10))

    def test_orientation_finite_extreme_scales_do_not_overflow(self):
        for scale in (1e308, 1e-300):
            with self.subTest(scale=scale), np.errstate(all='raise'):
                matrix = quaternion_matrix([0, 0, scale, scale])
                self.assertTrue(np.allclose(matrix @ [1, 0, 0], [0, 1, 0], atol=1e-10))
                self.assertTrue(np.allclose(orientation_error([0, 0, scale, scale], [0, 0, 0, 1]),
                                           [0, 0, np.pi/2]))

    def test_base_maps_physical_velocity_and_brakes_in_budget(self):
        env, control = make_rig()
        result = control.move_base(.15, 0, 0, 31)
        self.assertEqual(result['advanced_steps'], 31)
        self.assertEqual(result['velocity_command_steps'], 30)
        self.assertTrue(np.allclose(env.calls[0][:3], [.2, 0, 0]))
        self.assertTrue(np.array_equal(env.calls[-1][:3], np.zeros(3)))
        self.assertAlmostEqual(result['displacement_world'][0], .15, places=6)
        self.assertTrue(all(action.shape == (23,) and action.dtype == np.float32 for action in env.calls))
        self.assertTrue(np.allclose(env.calls[0][[14, 22]], 1))

    def test_base_default_steps_matches_schema(self):
        env, control = make_rig()
        result = control.move_base(.1, 0, 0)
        self.assertEqual(result['advanced_steps'], 60)
        self.assertEqual(result['velocity_command_steps'], 59)

    def test_base_rejects_unbounded_or_invalid_before_actions(self):
        for args in [(1, 0, 0, 10), (.5, 0, 0, 60), (0, 0, 1, 120), (0, 0, 0, 1),
                     (0, 0, 0, 121), (0, 0, 0, True), (np.nan, 0, 0, 2)]:
            with self.subTest(args=args):
                env, control = make_rig()
                with self.assertRaises(ValueError):
                    control.move_base(*args)
                self.assertFalse(env.calls)

    def test_arm_closed_loop_uses_absolute_joint_coordinates_and_stops_on_tolerance(self):
        env, control = make_rig()
        env.robot.q[10:17] = [.2, 0, 0, 0, 0, 0, 0]
        env.robot.q[17:24] = .4
        result = control.move_arm('left', [.3, .05, 0], steps=30)
        self.assertTrue(result['pose_tolerance_met'])
        self.assertLessEqual(result['position_error_m'], .005)
        self.assertTrue(0 < result['advanced_steps'] <= 30)
        self.assertGreater(env.calls[0][7], .2)  # Absolute target, not a delta/normalized action.
        self.assertTrue(all(np.allclose(action[15:22], .4) for action in env.calls))
        self.assertTrue(all(np.array_equal(action[:3], [0, 0, 0]) for action in env.calls))

    def test_arm_world_target_is_transformed_to_base(self):
        env, control = make_rig()
        env.robot.position = np.array([1., 2., 0.])
        env.robot.orientation = np.array([0., 0., np.sqrt(.5), np.sqrt(.5)])
        result = control.move_arm('right', [1., 2.1, 0.], frame='world', steps=30)
        self.assertTrue(result['pose_tolerance_met'])
        self.assertAlmostEqual(env.robot.q[17], .1, delta=.005)
        self.assertLessEqual(abs(env.robot.q[18]), .005)

    def test_arm_respects_joint_limits_and_reports_unreached(self):
        env, control = make_rig()
        env.robot.joint_upper_limits[10] = .02
        result = control.move_arm('left', [.2, 0, 0], steps=8)
        self.assertFalse(result['pose_tolerance_met'])
        self.assertEqual(result['advanced_steps'], 8)
        self.assertTrue(all(action[7] <= .02 + 1e-8 for action in env.calls))
        self.assertNotIn('success', result)

    def test_arm_invalid_command_never_advances(self):
        cases = [{'side': 'left', 'position': [.31, 0, 0]}, {'side': 'bad', 'position': [0, 0, 0]},
                 {'side': 'left', 'position': [0, 0, 0], 'orientation': [0, 0, 0, 0]},
                 {'side': 'left', 'position': [0, 0, 0], 'frame': 'camera'},
                 {'side': 'left', 'position': [0, 0, 0], 'steps': 121}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                env, control = make_rig()
                with self.assertRaises(ValueError):
                    control.move_arm(**kwargs)
                self.assertFalse(env.calls)

    def test_gripper_fraction_mapping_and_hold_during_other_tools(self):
        env, control = make_rig()
        result = control.gripper('left', 0, steps=5)
        self.assertEqual(result['advanced_steps'], 5)
        self.assertEqual(result['measured_opening_fraction'], 0)
        self.assertEqual(env.calls[-1][14], -1)
        # Contact-limited fingers must retain the closing command while moving.
        env.robot.q[24:26] = .02
        control.move_base(0, 0, 0, 2)
        self.assertEqual(env.calls[-1][14], -1)
        self.assertNotIn('grasped', result)
        self.assertNotIn('success', result)

    def test_torso_interpolates_absolute_targets(self):
        env, control = make_rig()
        env.robot.q[6:10] = [.5, -.5, .2, 0]
        result = control.torso([.6, -.4, .2, .1], steps=10)
        self.assertEqual(result['advanced_steps'], 10)
        self.assertTrue(np.allclose(env.calls[0][3:7], [.51, -.49, .2, .01]))
        self.assertLess(np.max(np.abs(result['joint_error_rad'])), 1e-6)

    def test_torso_invalid_limit_and_speed_rejected(self):
        env, control = make_rig()
        for target, steps in [([3, 0, 0, 0], 120), ([.5, 0, 0, 0], 2)]:
            with self.subTest(target=target, steps=steps), self.assertRaises(ValueError):
                control.torso(target, steps)
        self.assertFalse(env.calls)

    def test_terminal_stops_mid_tool_and_count_is_actual(self):
        env, control = make_rig()
        env.terminal_at = 3
        result = control.gripper('right', 0, steps=30)
        self.assertEqual(result['advanced_steps'], 3)
        self.assertTrue(result['environment_done'])
        self.assertEqual(len(env.calls), 3)

    def test_callback_partial_failure_is_never_retried(self):
        env, control = make_rig()
        def partial(action):
            env.step(action)
            raise IOError('ACK transport interrupted after physical advance')
        control.step_fn = partial
        with self.assertRaises(IOError):
            control.gripper('left', 0, 10)
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(env.steps, 1)
        control.step_fn = env.step
        with self.assertRaisesRegex(RuntimeError, 'cannot continue'):
            control.gripper('left', 0, 10)
        self.assertEqual(len(env.calls), 1)

    def test_callback_must_advance_exactly_one_step(self):
        env, control = make_rig()
        control.step_fn = lambda action: None
        with self.assertRaisesRegex(RuntimeError, 'exactly one'):
            control.gripper('left', 0, 10)
        self.assertEqual(env.steps, 0)

    def test_read_state_contains_robot_only_and_does_not_step(self):
        env, control = make_rig()
        class ForbiddenTruth:
            @property
            def env_accessor(self):
                raise AssertionError('Robot proprioception may not access object scope')
        env.backend.state = ForbiddenTruth()
        state = control.read_state()
        self.assertEqual(set(state['components']), {'trunk', 'arm_left', 'arm_right', 'gripper_left', 'gripper_right'})
        self.assertEqual(set(state['eef']), {'left', 'right'})
        self.assertEqual(state['step'], 0)
        self.assertFalse({'target', 'objects', 'success', 'scorer'} & set(state))
        self.assertFalse(env.calls)

    def test_controller_contract_drift_rejected(self):
        for change in ['normalize', 'arm_delta', 'base_scale', 'gripper_inverted', 'action_order']:
            with self.subTest(change=change):
                env = FakeEnvironment()
                if change == 'normalize':
                    env.robot._action_normalize = True
                elif change == 'arm_delta':
                    env.backend.ControllerView._controller_groups['arm_left'].use_delta_commands = True
                elif change == 'base_scale':
                    env.backend.ControllerView._controller_groups['base'].command_output_limits = ([-1]*3, [1]*3)
                elif change == 'gripper_inverted':
                    env.backend.ControllerView._controller_groups['gripper_left']._inverted = True
                else:
                    env.robot.controller_action_idx = {**env.robot.controller_action_idx, 'gripper_left': np.array([22])}
                with self.assertRaises(RuntimeError):
                    R1ProControl(env, env.step)
                self.assertFalse(env.calls)

    def test_native_jacobian_uses_member_row_body_offset_and_floating_columns(self):
        env = FakeEnvironment()
        control = R1ProControl(env, env.step)
        native = env.backend.ControllerView._controller_groups['arm_left']
        native.view_row_indices = np.array([1])
        jacobians = np.zeros((2, 4, 6, 34))
        selected = np.arange(42).reshape(6, 7)
        jacobians[1, 2, :, 16:23] = selected
        api = SimpleNamespace(get_all_joint_positions=lambda route: np.zeros((2, 28)),
                              get_all_relative_jacobians=lambda route: jacobians,
                              get_link_index=lambda route, link: 3)
        modules = {name: ModuleType(name) for name in
                   ('omnigibson', 'omnigibson.utils', 'omnigibson.utils.usd_utils')}
        modules['omnigibson.utils.usd_utils'].ControllableObjectViewAPI = api
        with patch.dict(sys.modules, modules):
            self.assertTrue(np.array_equal(control._arm_jacobian('left'), selected))


if __name__ == '__main__':
    unittest.main()
