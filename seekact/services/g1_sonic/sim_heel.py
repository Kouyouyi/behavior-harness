"""Simulation-only toe-pivot reference and measured heel/load audit.

Not registered with the field executor. No thermal model or cooling claim.
"""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .state import NAMES
from .reference import RATE
from .sim_reach import SCENE

TOE = np.array([.12, 0., -.035])
HEEL = np.array([-.05, 0., -.035])


def make(reference, side='left', angle=.20, shift=.035, foot_lift=0.):
    if (side not in ('left', 'right') or not 0 <= foot_lift <= .08 or
            not (0 <= angle <= .45 if foot_lift else .05 <= angle <= .45) or not 0 <= shift <= .12):
        raise ValueError('Invalid simulation heel candidate')
    model = mujoco.MjModel.from_xml_path(SCENE); data = mujoco.MjData(model)
    ids = np.array([model.joint(n).id for n in NAMES]); qids = model.jnt_qposadr[ids]
    anchor = np.asarray(reference['joint_pos'][0], dtype=float)
    data.qpos[:3] = [0, 0, .8]; data.qpos[3:7] = reference['body_quat'][0]
    data.qpos[qids] = anchor; mujoco.mj_forward(model, data)
    poses = {s:(data.body(s+'_ankle_roll_link').xpos.copy(),
                data.body(s+'_ankle_roll_link').xmat.reshape(3, 3).copy()) for s in ('left', 'right')}
    pelvis_shift = np.array([0., -shift if side == 'left' else shift, -.01 if shift else 0.])
    reports = {}

    def solve(lift, height=0.):
        q = anchor.copy()
        for s in poses:
            indices = np.array([NAMES.index(s+'_'+part+'_joint') for part in
                                ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')])
            pos, rot = poses[s]
            target_rot = rot @ Rotation.from_rotvec([0, lift if s == side else 0, 0]).as_matrix()
            target_pos = pos + rot @ TOE - target_rot @ TOE - pelvis_shift
            if s == side:
                target_pos = target_pos + [0., 0., height]
            limits = model.jnt_range[ids[indices]]
            lo = np.maximum(limits[:, 0]+.02, anchor[indices]-.8)
            hi = np.minimum(limits[:, 1]-.02, anchor[indices]+.8)
            def residual(values):
                data.qpos[qids] = q; data.qpos[qids[indices]] = values
                mujoco.mj_forward(model, data)
                body = data.body(s+'_ankle_roll_link')
                return np.r_[body.xpos-target_pos,
                             .15*Rotation.from_matrix(target_rot.T @ body.xmat.reshape(3,3)).as_rotvec(),
                             .0001*(values-anchor[indices])]
            fit = least_squares(residual, np.clip(anchor[indices], lo+1e-8, hi-1e-8),
                                bounds=(lo,hi), max_nfev=200)
            error = residual(fit.x)
            if not fit.success or np.linalg.norm(error[:3])>.002 or np.linalg.norm(error[3:6])>.003:
                raise ValueError(f'{s} heel IK failed: {error[:6]}')
            q[indices] = fit.x
            reports[f'{lift:.3f}_{height:.3f}_{s}'] = dict(position_error_m=float(np.linalg.norm(error[:3])),
                                            orientation_error_rad=float(np.linalg.norm(error[3:6])/.15))
        return q

    shifted, lifted = solve(0.), solve(angle, foot_lift)
    # A synthetic warning selects one side once. No automatic repeat or field hook.
    segments = [(2., anchor, 'baseline'), (3., shifted, 'shift_to_support'),
                (1., shifted, 'shift_hold'), (3., lifted, 'heel_up'),
                (3., lifted, 'heel_hold'), (3., shifted, 'heel_down'),
                (3., anchor, 'recenter'), (4., anchor, 'settle')]
    if foot_lift:
        segments = [(duration,target,name.replace('heel','foot')) for duration,target,name in segments]
    qs=[]; dqs=[]; phases=[]; current=anchor; at=0.
    idle_dq=np.asarray(reference['joint_vel'][0])
    for duration, target, name in segments:
        u=np.arange(round(duration*RATE))/RATE/duration
        blend=u**3*(10-15*u+6*u*u); derivative=30*u*u*(1-u)**2/duration
        qs.append(current+blend[:,None]*(target-current))
        dqs.append(idle_dq+derivative[:,None]*(target-current))
        phases.append(dict(name=name,start_s=at,end_s=at+duration))
        current=target;at+=duration
    qs.append(np.tile(anchor,(61,1)));dqs.append(np.tile(idle_dq,(61,1)))
    q=np.concatenate(qs);dq=np.concatenate(dqs)
    return dict(joint_pos=q.astype(np.float32),joint_vel=dq.astype(np.float32),
                body_quat=np.repeat(reference['body_quat'][:1],len(q),axis=0),duration_s=at), dict(
        side=side,angle_rad=angle,pelvis_shift_nominal_m=pelvis_shift.tolist(),
        mode='whole_foot' if foot_lift else 'toe_pivot',nominal_whole_foot_lift_m=foot_lift,
        nominal_local_heel_rise_m=float(.17*np.sin(angle)),phases=phases,ik=reports,
        trigger='synthetic single ankle high-temperature event; no motor thermal simulation',
        hardware_enabled=False,thermal_relief_validated=False)


def audit(records, details):
    side=details['side']; other='right' if side=='left' else 'left'
    baseline=[r for r in records if .3<=r['elapsed_s']<1.8]
    raised=[r for r in records if 10<=r['elapsed_s']<12]
    late=[r for r in records if r['elapsed_s']>=records[-1]['elapsed_s']-2]
    def stats(rows):
        return {s:dict(load_n=float(np.mean([r['loads'][s] for r in rows])),
            heel_height_m=float(np.mean([r['ankle_telemetry'][s]['heel_min_z_m'] for r in rows])),
            toe_height_m=float(np.mean([r['ankle_telemetry'][s]['toe_min_z_m'] for r in rows])),
            toe_contact_fraction=float(np.mean([r['ankle_telemetry'][s]['toe_contact_force_n']>1 for r in rows])),
            pitch_torque_rms_nm=float(np.sqrt(np.mean([r['ankle_telemetry'][s]['pitch_torque_nm']**2 for r in rows]))),
            roll_torque_rms_nm=float(np.sqrt(np.mean([r['ankle_telemetry'][s]['roll_torque_nm']**2 for r in rows]))))
                for s in (side,other)}
    before,hold,after=stats(baseline),stats(raised),stats(late)
    unloaded_fraction=float(np.mean([r['loads'][side]<5 and
        min(r['ankle_telemetry'][side]['toe_min_z_m'],r['ankle_telemetry'][side]['heel_min_z_m'])>.005
        for r in raised]))
    quat=np.array([r['root_quat'] for r in records]);tilts=np.arccos(np.clip(1-2*(quat[:,1]**2+quat[:,2]**2),-1,1))
    rise=hold[side]['heel_height_m']-before[side]['heel_height_m']
    drift=float(np.max(np.linalg.norm(np.array([r['root_xyz'] for r in late])-late[0]['root_xyz'],axis=1)))
    arm_contacts=sum(bool(r['geometry']['arm_contacts']) for r in records)
    support=float(np.max(np.abs([r['support_wrench'] for r in records])))
    foot_goal = (unloaded_fraction>.8 if details.get('mode')=='whole_foot' else
                 rise>.008 and hold[side]['toe_contact_fraction']>.8)
    return dict(case='heel',side=side,mode=details.get('mode','toe_pivot'),completed=bool(foot_goal
        and hold[other]['load_n']>100 and max(tilts)<.2 and drift<.02 and not arm_contacts and support==0),
        fully_unloaded_hold_fraction=unloaded_fraction,
        ankle_torque_reduced={axis:hold[side][axis+'_torque_rms_nm']<before[side][axis+'_torque_rms_nm']
                              for axis in ('pitch','roll')},
        mean_heel_rise_m=rise,max_heel_height_m=max(r['ankle_telemetry'][side]['heel_min_z_m'] for r in records),
        baseline=before,heel_hold=hold,returned=after,max_tilt_rad=float(max(tilts)),
        final_two_second_drift_m=drift,arm_contact_samples=arm_contacts,max_support_wrench=support,
        hardware_validated=False,thermal_relief_validated=False,
        note='Physical simulation measurements, not nominal IK. Torque/load changes do not prove cooling.')
