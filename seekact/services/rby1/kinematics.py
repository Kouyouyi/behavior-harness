"""RB-Y1 (model A) kinematics from the vendor URDF, through rby1_sdk.dynamics.

Frames are the robot's `base` frame: x forward, y left, z up, metres. The joint vector is
the SDK's 24-vector in `Model_A.robot_joint_names` order (two wheels, six torso joints,
right arm, left arm, two head joints). The head camera's optical frame (z forward, x
right, y down) is `link_head_2` composed with a hand-eye calibration (X_HEAD_CAMERA by default),
so pixel back-projection composes with `frame(HEAD_CAMERA, q)` directly.
"""
from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

JOINT_NAMES = ('right_wheel', 'left_wheel', *(f'torso_{i}' for i in range(6)),
               *(f'right_arm_{i}' for i in range(7)), *(f'left_arm_{i}' for i in range(7)), 'head_0', 'head_1')
COMPONENTS = {'torso': (2, 8), 'right_arm': (8, 15), 'left_arm': (15, 22), 'head': (22, 24)}
EE_FRAME = {'right': 'ee_right', 'left': 'ee_left'}
SHOULDER_FRAME = {'right': 'link_right_arm_0', 'left': 'link_left_arm_0'}
HEAD_LINK = 'link_head_2'
HEAD_CAMERA = 'head_camera'  # pseudo-frame: the head ZED's left optical frame

# link_head_2 -> camera mount and mount -> left optical frame: the hand-eye calibration of the head
# ZED 2i on the unit this backend was characterized on (Tsai, 15 samples, 1.3 mm / 0.21 deg mean
# error). UNIT-SPECIFIC: calibrate your own robot's head camera and pass the 4x4 link_head_2 ->
# optical transform as the backend option `head_camera_extrinsic`.
X_HEAD_MOUNT = np.array([[1.0, 0.0, 0.0, 0.04], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
X_MOUNT_OPTICAL = np.array([
    [-0.010999457159467374, -0.01063192264944987, 0.9998829802346738, 0.0006646805463202913],
    [-0.9999021827497775, -0.008522349734540802, -0.011090287968878705, 0.05961217241487954],
    [0.008639263535020073, -0.9999071615784029, -0.01053714144320872, 0.09163858484685387],
    [0.0, 0.0, 0.0, 1.0]])
X_HEAD_CAMERA = X_HEAD_MOUNT @ X_MOUNT_OPTICAL
OFFLINE_URDF = Path(__file__).resolve().parent/'model'/'rby1a.urdf'  # vendor model A, for offline use


def component(q, name):
    """The joints of one component (torso, right_arm, left_arm, head) of a 24-vector."""
    start, stop = COMPONENTS[name]
    return np.asarray(q, dtype=np.float64)[start:stop]


def with_component(q, name, values):
    """A copy of the 24-vector q with one component replaced."""
    start, stop = COMPONENTS[name]
    out = np.array(q, dtype=np.float64)
    out[start:stop] = values
    return out


class Kinematics:
    """Forward kinematics, a damped least-squares arm IK and self-collision on the vendor model.

    Give the robot's own URDF text (`Robot.get_robot_model()`, matching its firmware) or a
    URDF path (the SDK checkout's `models/rby1a/urdf/model.urdf`, for offline use).
    """

    def __init__(self, urdf_text=None, urdf_path=None, head_camera=None):
        """head_camera: the 4x4 link_head_2 -> head-camera optical transform; None: X_HEAD_CAMERA."""
        import rby1_sdk.dynamics as rd
        if (urdf_text is None) == (urdf_path is None):
            raise ValueError('give exactly one of urdf_text or urdf_path')
        self.head_camera = X_HEAD_CAMERA if head_camera is None else np.asarray(head_camera, dtype=np.float64)
        if self.head_camera.shape != (4, 4) or not np.allclose(self.head_camera[3], [0, 0, 0, 1]):
            raise ValueError('head_camera must be a 4x4 homogeneous transform (last row 0 0 0 1)')
        config = (rd.load_robot_from_urdf_data(urdf_text, 'base') if urdf_text is not None
                  else rd.load_robot_from_urdf(str(urdf_path), 'base'))
        self.dyn = rd.Robot_24(config)
        self.links = list(self.dyn.get_link_names())
        if 'base' in self.links:
            self.links.remove('base')
        self.links.insert(0, 'base')
        self.state = self.dyn.make_state(self.links, list(JOINT_NAMES))
        self.lock = threading.Lock()
        self.lower = np.asarray(self.dyn.get_limit_q_lower(self.state), dtype=np.float64)
        self.upper = np.asarray(self.dyn.get_limit_q_upper(self.state), dtype=np.float64)
        self.qdot_upper = np.asarray(self.dyn.get_limit_qdot_upper(self.state), dtype=np.float64)
        # The URDF's rated joint torques (N m). The wheels have none and the head's 1000 is a placeholder:
        # both are inf here, so torque checks skip them.
        limit = np.asarray(self.dyn.get_limit_torque(self.state), dtype=np.float64)
        limit[[0, 1, 22, 23]] = np.inf
        self.torque_limit = np.where(np.isfinite(limit) & (limit > 0) & (limit < 1e4), limit, np.inf)

    def gravity(self, q):
        """Joint torques (N m, 24-vector) that hold q still against gravity, from the vendor model.

        On the robot the measured holding torques read 10-40 % above this (torso_2: 119 N m measured,
        92 N m modelled), so callers scale it before comparing with a limit.
        """
        zeros = np.zeros(len(JOINT_NAMES))
        with self.lock:
            self.state.set_q(np.asarray(q, dtype=np.float64))
            self.state.set_qdot(zeros)
            self.state.set_qddot(zeros)
            self.state.set_gravity(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -9.81]))
            self.dyn.compute_forward_kinematics(self.state)
            self.dyn.compute_diff_forward_kinematics(self.state)
            self.dyn.compute_2nd_diff_forward_kinematics(self.state)
            self.dyn.compute_inverse_dynamics(self.state)
            return np.asarray(self.state.get_tau(), dtype=np.float64).copy()

    def frames(self, names, q):
        """{name: 4x4 base-frame pose} of URDF links (or HEAD_CAMERA) at the 24-vector q."""
        with self.lock:
            self.state.set_q(np.asarray(q, dtype=np.float64))
            self.dyn.compute_forward_kinematics(self.state)
            out = {}
            for name in names:
                link = HEAD_LINK if name == HEAD_CAMERA else name
                T = np.asarray(self.dyn.compute_transformation(self.state, 0, self.links.index(link)))
                out[name] = T @ self.head_camera if name == HEAD_CAMERA else T
        return out

    def frame(self, name, q):
        return self.frames([name], q)[name]

    def collisions(self, q):
        """Colliding (link, link) pairs at q (penetration depth below zero)."""
        with self.lock:
            self.state.set_q(np.asarray(q, dtype=np.float64))
            self.dyn.compute_forward_kinematics(self.state)
            results = self.dyn.detect_collisions_or_nearest_links(self.state, 0)
        return {tuple(sorted((r.link1, r.link2))) for r in results if r.distance < 0}

    def first_new_collision(self, configurations):
        """(index, pairs) of the first configuration with a collision absent from the first one, else None.

        Contacts already present at the start are tolerated so that a motion can leave them.
        """
        configurations = list(configurations)
        initial = self.collisions(configurations[0]) if configurations else set()
        for i, q in enumerate(configurations[1:], start=1):
            new = self.collisions(q) - initial
            if new:
                return i, sorted(new)
        return None

    def arm_singular_value(self, side, q):
        """Smallest singular value of one arm's 6x7 end-effector Jacobian (finite differences of FK):
        near 0 the arm is near a singular pose (an almost straight elbow), where a small hand motion
        needs very fast joint motion."""
        from scipy.spatial.transform import Rotation
        name, frame = f'{side}_arm', EE_FRAME[side]
        q = np.array(q, dtype=np.float64)
        arm = component(q, name).copy()
        T0 = self.frame(frame, q)
        eps = 1e-6
        J = np.empty((6, arm.size))
        for k in range(arm.size):
            bumped = arm.copy()
            bumped[k] += eps
            T = self.frame(frame, with_component(q, name, bumped))
            J[:3, k] = (T[:3, 3] - T0[:3, 3]) / eps
            J[3:, k] = Rotation.from_matrix(T[:3, :3] @ T0[:3, :3].T).as_rotvec() / eps
        return float(np.linalg.svd(J, compute_uv=False)[-1])

    def solve_arm(self, side, target, q, *, lower, upper, iterations=200, damping=1e-3,
                  position_tolerance=0.002, orientation_tolerance=0.02):
        """Damped least-squares IK for one 7-DoF arm; torso, head and the other arm stay fixed.

        The Jacobian is taken by finite differences of FK, so it does not depend on the SDK's
        twist ordering. Returns (arm joint vector, position error m, orientation error rad).
        """
        from scipy.spatial.transform import Rotation
        name = f'{side}_arm'
        frame = EE_FRAME[side]
        target = np.asarray(target, dtype=np.float64)
        q = np.array(q, dtype=np.float64)
        arm = component(q, name).copy()

        def error(values):
            T = self.frame(frame, with_component(q, name, values))
            rotation = Rotation.from_matrix(target[:3, :3] @ T[:3, :3].T).as_rotvec()
            return np.concatenate([target[:3, 3] - T[:3, 3], rotation])

        eps = 1e-6
        for _ in range(iterations):
            e = error(arm)
            if np.linalg.norm(e[:3]) < position_tolerance and np.linalg.norm(e[3:]) < orientation_tolerance:
                break
            J = np.empty((6, arm.size))
            for k in range(arm.size):
                bumped = arm.copy()
                bumped[k] += eps
                J[:, k] = (e - error(bumped)) / eps
            step = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            arm = np.clip(arm + step, lower, upper)
        e = error(arm)
        return arm, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))


def matrix(position, quaternion):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    T[:3, 3] = position
    return T


def pose(T):
    from scipy.spatial.transform import Rotation
    return np.asarray(T[:3, 3]), Rotation.from_matrix(T[:3, :3]).as_quat()
