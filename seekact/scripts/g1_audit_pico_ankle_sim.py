"""Read-only comparison of recorded Pico ankle demonstrations and simulated replay."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--side',choices=['left','right'],required=True)
    a=p.parse_args();run=a.run
    rows=[json.loads(line) for line in (run/'samples.jsonl').read_text().splitlines()]
    replay=[r for r in rows if r['phase']=='recorded_episode']
    baseline=[r for r in rows if r['phase']=='recorded_start_hold'][-25:]
    returned=[r for r in rows if r['phase']=='returned_start_hold'][-25:]
    meta=json.loads((run/'recorded_episode.json').read_text())
    with np.load(run/'recorded_episode.npz',allow_pickle=False) as f:
        token=f['action.motion_token'];real_q=f['observation.state']
    joint_meta=json.loads((Path(__file__).resolve().parents[1]/'services/g1_sonic/joints.json').read_text())
    names=[j['name'] for j in joint_meta['joints']]
    matched=[];errors=[]
    for r in replay:
        e=np.max(abs(token-np.asarray(r['native_token'])),axis=1)
        matched.append(int(np.argmin(e)));errors.append(float(np.min(e)))
    b=float(np.mean([r['ankle_telemetry'][a.side]['heel_min_z_m'] for r in baseline]))
    peak=max(replay,key=lambda r:r['ankle_telemetry'][a.side]['heel_min_z_m'])
    comparisons={}
    for s in ('left','right'):
        comparisons[s]={}
        for part in ('hip_pitch','hip_roll','knee','ankle_pitch','ankle_roll'):
            name=s+'_'+part+'_joint';i=names.index(name);j=meta['joint_names'].index(name)
            v=np.array([r['body_q'][i] for r in replay])
            comparisons[s][part]=dict(recorded_min=float(real_q[:,j].min()),recorded_max=float(real_q[:,j].max()),
                                      simulated_min=float(v.min()),simulated_max=float(v.max()))
    arm_phases={phase:sum(bool(r['geometry']['arm_contacts']) for r in rows if r['phase']==phase)
                for phase in sorted(set(r['phase'] for r in rows))}
    submitted=sorted(set(r['source_frame'] for r in replay))
    report=dict(episode=meta['episode'],side=a.side,readback_max_nearest_recorded_token_error=max(errors),
        readback_first_last_matched_frames=[matched[0],matched[-1]],
        source_frames_not_submitted=sorted(set(range(len(token)))-set(submitted)),
        baseline_heel_height_m=b,peak_heel_height_m=peak['ankle_telemetry'][a.side]['heel_min_z_m'],
        peak_heel_rise_m=peak['ankle_telemetry'][a.side]['heel_min_z_m']-b,
        returned_heel_height_m=float(np.mean([r['ankle_telemetry'][a.side]['heel_min_z_m'] for r in returned])),
        minimum_replay_foot_load_n=min(r['loads'][a.side] for r in replay),
        arm_contact_samples_by_phase=arm_phases,joints=comparisons,
        interpretation='A faithful token transport check under a different simulated initial state; not a refutation of the real demonstration. No temperature model or real action.',
        original_return_incomplete=True,return_method='synthesized 3s token interpolation to first recorded token, then idle')
    target=run/'demonstration_comparison.json'
    with target.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
