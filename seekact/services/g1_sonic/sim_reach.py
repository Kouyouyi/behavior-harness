"""Simulation reach diagnostic. Not registered as a real-robot tool.

Normal collision geometry is retained. IK produces nominal references; measured
MuJoCo poses and native readback are used separately by the experiment runner.
"""
import numpy as np
import mujoco
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .state import NAMES
from .reference import LOOKAHEAD, RATE

SCENE = '/home/user/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'


class Geometry:
    def __init__(self, state=None):
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        if state is not None:
            self.data.qpos[:] = state['qpos']
        self.ids = np.array([self.model.joint(n).id for n in NAMES])
        self.qids = self.model.jnt_qposadr[self.ids]
        self.wrists = {s:self.model.body(s+'_wrist_yaw_link').id for s in ('left', 'right')}
        self.arm_indices = {s:np.array([NAMES.index(s+'_'+part+'_joint') for part in
            ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')])
            for s in ('left', 'right')}
        self.near_body = [g for g in range(self.model.ngeom) if self.collidable(g) and
            any(x in self.body(g) for x in ('hip_', 'torso', 'pelvis'))]
        self.hands = {s:[g for g in range(self.model.ngeom) if self.collidable(g) and self.body(g).startswith(s+'_') and
            any(x in self.body(g) for x in ('hand_', 'wrist_'))] for s in ('left', 'right')}

    def body(self, geom):
        return self.model.body(int(self.model.geom_bodyid[geom])).name

    def collidable(self, geom):
        return bool(self.model.geom_contype[geom] or self.model.geom_conaffinity[geom])

    def forward(self, q=None):
        if q is not None:
            self.data.qpos[self.qids] = q
        mujoco.mj_forward(self.model, self.data)

    def wrist(self, side):
        base = Rotation.from_quat(self.data.qpos[[4, 5, 6, 3]])
        body = self.wrists[side]
        return (base.inv().apply(self.data.xpos[body]-self.data.qpos[:3]),
                base.inv()*Rotation.from_matrix(self.data.xmat[body].reshape(3, 3)))

    def measure(self, state):
        self.data.qpos[:] = state['qpos']; self.data.qvel[:] = state['qvel']; self.forward()
        poses = {s:self.wrist(s) for s in ('left', 'right')}
        contacts = []
        for c in self.data.contact:
            bodies = [self.body(g) for g in (c.geom1, c.geom2)]
            if any(any(x in b for x in ('hand_', 'wrist_', 'elbow_', 'shoulder_')) for b in bodies):
                contacts.append(dict(bodies=bodies, distance_m=float(c.dist)))
        clearance = {s:min(mujoco.mj_geomDistance(self.model, self.data, a, b, .2, None)
                           for a in self.hands[s] for b in self.near_body) for s in poses}
        rotations = self.data.geom_xmat.reshape(-1, 3, 3)
        centers = self.data.geom_xpos+np.einsum('nij,nj->ni', rotations, self.model.geom_aabb[:, :3])
        extents = np.einsum('nij,nj->ni', abs(rotations), self.model.geom_aabb[:, 3:])
        # Some box/mesh distance queries return zero despite separated witness
        # points. Enclosing world AABBs give an independent conservative bound.
        lower_bound = {s:min(float(np.linalg.norm(np.maximum(abs(centers[a]-centers[b])-extents[a]-extents[b], 0)))
                            for a in self.hands[s] for b in self.near_body) for s in poses}
        return dict(wrists_base={s:poses[s][0].tolist() for s in poses},
                    wrist_rotvec_base={s:poses[s][1].as_rotvec().tolist() for s in poses},
                    arm_contacts=contacts, hand_body_clearance_m=clearance,
                    hand_body_clearance_lower_bound_m=lower_bound)

    def solve(self, q, targets):
        q = np.asarray(q, dtype=float).copy()
        details = {}
        for side, target in targets.items():
            indices = self.arm_indices[side]
            self.forward(q)
            initial_rotation = self.wrist(side)[1]
            anchor = q[indices].copy()
            limits = self.model.jnt_range[self.ids[indices]]
            lower = np.maximum(limits[:, 0]+.1, anchor-1.2)
            upper = np.minimum(limits[:, 1]-.1, anchor+1.2)
            def residual(value):
                candidate = q.copy(); candidate[indices] = value; self.forward(candidate)
                pos, rot = self.wrist(side)
                return np.r_[pos-target, .06*(initial_rotation.inv()*rot).as_rotvec(), .002*(value-anchor)]
            fit = least_squares(residual, np.clip(anchor, lower+1e-8, upper-1e-8),
                                bounds=(lower, upper), max_nfev=120)
            residual(fit.x)
            pos, rotation = self.wrist(side)
            error = float(np.linalg.norm(pos-target))
            if not fit.success or error > .005:
                raise ValueError(f'{side} reference IK failed: {error:.4f} m')
            q[indices] = fit.x
            details[side] = dict(target_base_m=np.asarray(target).tolist(), nominal_position_error_m=error,
                nominal_orientation_change_rad=float((initial_rotation.inv()*rotation).magnitude()),
                max_joint_reference_change_rad=float(np.max(abs(fit.x-anchor))))
        return q, details


class Transition:
    def __init__(self, reference):
        self.reference = reference
        self.target = reference['joint_pos'][0].copy()
        self.start_q = self.target.copy()
        self.at = 0.
        self.duration = 4.
        self.phase = 'standing'

    def begin(self, target, now, phase):
        target = np.asarray(target)
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError('Invalid simulation reference target')
        self.start_q = self.target.copy(); self.target = target.copy()
        self.at, self.phase = now, phase

    def fields(self, now, frame):
        u = np.clip((now-self.at+np.arange(LOOKAHEAD)/RATE)/self.duration, 0, 1)
        s = u**3*(10+u*(-15+6*u)); ds = 30*u*u*(1-u)**2/self.duration
        delta = self.target-self.start_q
        return dict(joint_pos=(self.start_q+s[:, None]*delta).astype(np.float32),
            joint_vel=(self.reference['joint_vel'][0]+ds[:, None]*delta).astype(np.float32),
            body_quat=np.repeat(self.reference['body_quat'][:1], LOOKAHEAD, axis=0).astype(np.float32),
            frame_index=np.arange(frame, frame+LOOKAHEAD, dtype=np.int64))


def run_reach(reference, transition, solver, get_state, records, tick, event, save):
    """Bounded physics experiment; no hardware transport or registration here."""
    limits = dict(clearance_m=.01, settled_speed_rad_s=.15, settled_seconds=1.,
                  stage_timeout_s=24., reach_goal_error_m=.03, reach_forward_min_m=.07,
                  return_error_m=.02, orientation_error_rad=.20, reach_offset_base_m=[.10, 0, .04],
                  max_root_tilt_rad=.25, clearance_method='conservative enclosing world AABB lower bound')
    save('reach_criteria.json', limits)
    snapshots = {}

    def plan(q, targets):
        # IK must not pause the 50 Hz native reference publisher.
        solver.data.qpos[:] = get_state()['qpos']
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(solver.solve, q, targets)
            while not future.done():
                tick()
            return future.result()

    def settle(phase):
        deadline = time.monotonic()+limits['stage_timeout_s']
        while time.monotonic() < deadline:
            tick()
            recent = [r for r in records if r['phase'] == phase and
                      r['elapsed_s'] >= records[-1]['elapsed_s']-1.]
            if len(recent) < 20 or recent[-1]['elapsed_s']-recent[0]['elapsed_s'] < .95:
                continue
            if any(r['native_target_qdq_max_error'] > 1e-4 or
                   max(abs(np.asarray(r['dq']))) >= limits['settled_speed_rad_s'] for r in recent):
                continue
            if any(r['geometry']['arm_contacts'] or
                   min(r['geometry']['hand_body_clearance_lower_bound_m'].values()) < limits['clearance_m'] for r in recent):
                continue
            snapshot = dict(
                elapsed_s=records[-1]['elapsed_s'],
                wrist_base=np.median([r['wrist_base'] for r in recent], axis=0).tolist(),
                wrist_rotvec_base=Rotation.from_rotvec([r['geometry']['wrist_rotvec_base']['right'] for r in recent]).mean().as_rotvec().tolist(),
                q=np.median([r['q'] for r in recent], axis=0).tolist(),
                minimum_hand_body_clearance_lower_bound_m=min(min(r['geometry']['hand_body_clearance_lower_bound_m'].values()) for r in recent),
                max_joint_speed_rad_s=max(max(abs(np.asarray(r['dq']))) for r in recent),
                native_target_received=True, arm_contact_samples=0)
            snapshots[phase] = snapshot
            save('reach_stages.json', snapshots)
            event('stage_settled', phase=phase, **snapshot)
            return snapshot
        raise RuntimeError('Reach stage did not settle with reference received and arms clear: '+phase)

    q0 = np.asarray(reference['joint_pos'][0], dtype=float)
    solver.data.qpos[:] = get_state()['qpos']; solver.forward(q0)
    clear_targets = {s:solver.wrist(s)[0]+[.08, .08 if s == 'left' else -.08, .06] for s in ('left', 'right')}
    clear_q, clear_details = plan(q0, clear_targets)
    save('clear_reference.json', dict(q=clear_q.tolist(), ik=clear_details))
    transition.begin(clear_q, time.monotonic(), 'clear')
    event('stage_started', phase='clear')
    clear = settle('clear')
    reach_target = np.asarray(clear['wrist_base'])+limits['reach_offset_base_m']
    reach_q, reach_details = plan(clear_q, {'right':reach_target})
    save('reach_reference.json', dict(q=reach_q.tolist(), ik=reach_details, measured_clear=clear))
    transition.begin(reach_q, time.monotonic(), 'reach')
    event('stage_started', phase='reach')
    reach = settle('reach')
    transition.begin(clear_q, time.monotonic(), 'return_clear')
    event('stage_started', phase='return_clear')
    returned = settle('return_clear')
    delta = np.asarray(reach['wrist_base'])-clear['wrist_base']
    goal_error = float(np.linalg.norm(np.asarray(reach['wrist_base'])-reach_target))
    return_error = float(np.linalg.norm(np.asarray(returned['wrist_base'])-clear['wrist_base']))
    rot0 = Rotation.from_rotvec(clear['wrist_rotvec_base'])
    reach_rot = float((rot0.inv()*Rotation.from_rotvec(reach['wrist_rotvec_base'])).magnitude())
    return_rot = float((rot0.inv()*Rotation.from_rotvec(returned['wrist_rotvec_base'])).magnitude())
    arm_ids = np.r_[solver.arm_indices['left'], solver.arm_indices['right']]
    other_ids = np.array([i for i in range(29) if i not in arm_ids])
    after_clear = [r for r in records if r['elapsed_s'] >= clear['elapsed_s']]
    clearance = min(min(r['geometry']['hand_body_clearance_lower_bound_m'].values()) for r in after_clear)
    contact_count = sum(bool(r['geometry']['arm_contacts']) for r in after_clear)
    max_tilt = max(r['root_tilt_rad'] for r in records)
    return dict(status='completed', case='reach', samples=len(records),
        criteria_passed=bool(goal_error <= limits['reach_goal_error_m'] and delta[0] >= limits['reach_forward_min_m'] and
                             return_error <= limits['return_error_m'] and max(reach_rot, return_rot) <= limits['orientation_error_rad'] and
                             clearance >= limits['clearance_m'] and contact_count == 0 and max_tilt <= limits['max_root_tilt_rad']),
        reach_displacement_base_m=delta.tolist(), reach_goal_error_m=goal_error,
        return_to_clear_error_m=return_error, reach_orientation_change_rad=reach_rot, return_orientation_error_rad=return_rot,
        reference_other_joints_max_change_rad=float(np.max(abs(reach_q[other_ids]-q0[other_ids]))),
        max_root_tilt_rad=max_tilt,
        minimum_hand_body_clearance_lower_bound_after_clear_m=clearance,
        arm_contact_samples_after_clear=contact_count,
        max_joint_return_to_clear_error_rad=float(np.max(abs(np.asarray(returned['q'])-clear['q']))),
        stages=snapshots)
