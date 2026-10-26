"""Evaluate recorded movement separately from command completion; no robot I/O."""
import argparse
import json
import math
from pathlib import Path


def rotation_distance(a, b):
    """SO(3) angle between two rotation vectors, via unit quaternions."""
    def quaternion(v):
        angle=math.sqrt(sum(x*x for x in v))
        scale=math.sin(angle/2)/angle if angle else .5
        return [math.cos(angle/2),*(x*scale for x in v)]
    dot=sum(x*y for x,y in zip(quaternion(a),quaternion(b)))
    return 2*math.acos(min(1.,abs(dot)))


def audit(run):
    trials=json.loads((run/'trials.json').read_text())
    rows=[json.loads(line) for line in (run/'samples.jsonl').open()]
    by_direction={}
    for t in trials:
        name=t['direction'];x,y=t['body_frame_displacement_m'];turn=t['measured_yaw_delta_deg']
        direction_ok=True
        if 'forward' in name:direction_ok &= x>.03
        if 'backward' in name:direction_ok &= x<-.03
        if name=='left' or name.endswith('_left') and not name.startswith('rotate'):direction_ok &= y>.03
        if name=='right' or name.endswith('_right') and not name.startswith('rotate'):direction_ok &= y<-.03
        if name=='rotate_left':direction_ok=turn>5 and math.hypot(x,y)<.1
        if name=='rotate_right':direction_ok=turn<-5 and math.hypot(x,y)<.1
        if name=='step_in_place':direction_ok=math.hypot(x,y)<.05
        checks=dict(command_completed=t['completed'],settled=t['settled'],direction_observed=bool(direction_ok),
                    upper_reference_held=t['upper_encoder_max_error_rad']<1e-4,
                    upper_tracking_within_0p15_rad=t['upper_measured_max_error_rad']<.15,
                    no_arm_contact=t['arm_contact_samples']==0,no_external_support=t['max_support_wrench']==0)
        by_direction[name]=dict(checks=checks,passed=all(checks.values()),
            displacement_body_m=[x,y],measured_yaw_delta_deg=turn,
            upper_tracking_max_rad=t['upper_measured_max_error_rad'])
    wrists={s:max(math.dist(r['geometry']['wrists_base'][s],rows[0]['geometry']['wrists_base'][s]) for r in rows)
            for s in ('left','right')}
    wrist_angles={s:math.degrees(max(rotation_distance(r['geometry']['wrist_rotvec_base'][s],
                    rows[0]['geometry']['wrist_rotvec_base'][s]) for r in rows)) for s in ('left','right')}
    tilt=max(math.acos(max(-1,min(1,1-2*(r['root_quat'][1]**2+r['root_quat'][2]**2)))) for r in rows)
    return dict(passed=bool(by_direction) and all(v['passed'] for v in by_direction.values()) and tilt<.25,
        directions=by_direction,max_tilt_rad=tilt,max_wrist_drift_in_base_m=wrists,
        max_wrist_orientation_drift_in_base_deg=wrist_angles,
        upper_reference_max_error_rad=max(r['upper_encoder_error_rad'] for r in rows),
        upper_measured_max_error_rad=max(r['upper_measured_error_rad'] for r in rows),
        hardware_validated=False,payload_retention_tested=False,
        scope='Observed direction and bounded posture tracking, not exact distance/angle or grasp retention. Step-in-place checks drift only.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();report=audit(args.run)
    (args.run/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)
