"""Reconstruct contact geometry from recorded MuJoCo states, without stepping."""
import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    a = p.parse_args()
    run = a.run
    out = run/'contact_audit'; out.mkdir(exist_ok=False)
    manifest = json.loads((run/'manifest.json').read_text())
    m = mujoco.MjModel.from_xml_path('/home/user/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml')
    if manifest.get('hand_collision_ablation') or manifest.get('arm_collision_ablation'):
        for g in range(m.ngeom):
            parts = ('_hand_', '_wrist_', '_elbow_', '_shoulder_') if manifest.get('arm_collision_ablation') else ('_hand_',)
            if any(part in m.body(int(m.geom_bodyid[g])).name for part in parts):
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
    d = mujoco.MjData(m)
    events = [json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
    start = next(e['monotonic'] for e in events if e['event'] == 'job_started')
    reach = manifest.get('case') == 'reach'
    finish = next((e['monotonic'] for e in events if e['event'] == 'completed'), float('inf')) if reach else start+17
    clear_at = next((e['monotonic'] for e in events if e['event'] == 'stage_settled' and e.get('phase') == 'clear'), float('inf'))
    body = lambda g: m.body(int(m.geom_bodyid[g])).name
    near_body = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and
                 any(x in body(g) for x in ('hip_', 'torso', 'pelvis'))]
    hands = {s:[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and
                body(g).startswith(s+'_') and any(x in body(g) for x in ('hand_', 'wrist_'))] for s in ('left', 'right')}
    clear_count = clear_contacts = 0
    bounds = {s:float('inf') for s in hands}
    next_sample = start; count = 0; contacts = {}
    for line in (run/'scene/trajectory.jsonl').open():
        row = json.loads(line)
        if not next_sample <= row['monotonic'] <= finish:
            continue
        next_sample = row['monotonic']+(0 if reach else .1)
        count += 1
        d.qpos[:] = row['qpos']; d.qvel[:] = row['qvel']; mujoco.mj_forward(m, d)
        after_clear = reach and row['monotonic'] >= clear_at
        if after_clear:
            clear_count += 1
            rotations = d.geom_xmat.reshape(-1, 3, 3)
            centers = d.geom_xpos+np.einsum('nij,nj->ni', rotations, m.geom_aabb[:, :3])
            extents = np.einsum('nij,nj->ni', abs(rotations), m.geom_aabb[:, 3:])
            for side in hands:
                # World AABBs enclose each collision geom: disjoint boxes give
                # a conservative lower bound without relying on mj_geomDistance.
                bound = min(np.linalg.norm(np.maximum(abs(centers[a]-centers[b])-extents[a]-extents[b], 0))
                            for a in hands[side] for b in near_body)
                bounds[side] = min(bounds[side], float(bound))
        for c in d.contact:
            bodies = [m.body(int(m.geom_bodyid[g])).name for g in (c.geom1, c.geom2)]
            if not any(any(part in b for part in ('hand_', 'wrist_', 'elbow_', 'shoulder_')) for b in bodies):
                continue
            if after_clear:
                clear_contacts += 1
            key = ' / '.join(sorted(bodies))
            record = contacts.setdefault(key, dict(contact_count=0, min_distance_m=0., first_elapsed_s=row['monotonic']-start))
            record['contact_count'] += 1
            record['min_distance_m'] = min(record['min_distance_m'], float(c.dist))
    report = dict(sampled_states=count, hand_collisions_disabled=bool(manifest.get('hand_collision_ablation')),
        arm_collisions_disabled=bool(manifest.get('arm_collision_ablation')),
        arm_hand_pairs=contacts, physics_advanced=False,
        limitation=('All recorded states after job start (~50 Hz)' if reach else 'Geometry reconstructed at ~10 Hz')+
                   '; not historical contact force measurements or exhaustive physics-substep coverage.')
    if reach:
        report['after_clear'] = dict(sampled_states=clear_count, arm_contact_count=clear_contacts,
            hand_body_aabb_clearance_lower_bound_m=bounds if clear_count else None,
            scope='collidable hands/wrists versus hips/torso/pelvis; conservative world AABB distances')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    shutil.copy2(__file__, out/'g1_audit_sim_contacts.py')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
