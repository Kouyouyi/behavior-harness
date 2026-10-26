"""Read-only audit of archived real commissioning reach jobs; no transport."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from services.g1_sonic.state import NAMES
from services.g1_sonic.reach import OTHER_IDS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();out=run/'reach_audit';out.mkdir(exist_ok=False)
    events=[json.loads(x) for x in (run/'executor_events.jsonl').read_text().splitlines()]
    jobs={e['id']:dict(start=e['monotonic'],action=e['action']) for e in events if e['event']=='job_started'}
    for e in events:
        if e['event']=='job_finished' and e['id'] in jobs:
            jobs[e['id']].update(end=e['monotonic'],result=e)
    samples=[json.loads(x) for x in (run/'reference_execution_trace.jsonl').read_text().splitlines()]
    native_stats={k:dict(count=0,tilt=0.,speed=0.,transport_faults=0) for k in jobs}
    for line in (run/'controller_trace.jsonl').open():
        row=json.loads(line)
        for key,job in jobs.items():
            if not job['start']<=row['monotonic']<=job['end']:continue
            stat=native_stats[key];stat['count']+=1
            q=np.asarray(row['base_quat']);q=q/np.linalg.norm(q)
            stat['tilt']=max(stat['tilt'],float(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2),-1,1))))
            stat['speed']=max(stat['speed'],max(abs(v) for v in row['body_dq']))
            stat['transport_faults']+=int(bool(row['transport_fault']))
    reports={}
    for key,job in jobs.items():
        rows=[r for r in samples if r['job_id']==key];result=job['result'];stats=native_stats[key]
        times=np.array([r['monotonic'] for r in rows]);sent=np.array([r['sent_q'] for r in rows]);q=np.array([r['measured_q'] for r in rows])
        record=dict(action=job['action'],completed=result['completed'],samples=len(rows),native_samples=stats['count'],
            max_publish_sample_gap_s=float(np.max(np.diff(times))),
            max_native_tilt_rad=stats['tilt'],max_native_joint_speed_rad_s=stats['speed'],native_transport_fault_samples=stats['transport_faults'],
            other_joint_reference_max_change_rad=float(np.max(abs(sent[:,OTHER_IDS]-sent[0,OTHER_IDS]))),
            measured_other_joint_max_change_rad=float(np.max(abs(q[:,OTHER_IDS]-q[0,OTHER_IDS]))))
        if job['action']=='reach_return':
            origin=np.asarray(result['before']['right']['position_m'])
            late=[r for r in rows if r['stage']=='return' and r['monotonic']>=job['end']-1.]
            errors=[float(np.linalg.norm(np.asarray(r['wrists']['right']['position_m'])-origin)) for r in late]
            record['last_second_return_error_m']=dict(min=min(errors),max=max(errors),median=float(np.median(errors)),samples=len(errors))
            record['measured_return_joint_errors_rad']={n:float(abs(q[-1,i]-q[0,i])) for i,n in enumerate(NAMES)}
        reports[key]=record
    report=dict(jobs=reports,hardware_commands_sent=0,
        limitation='FK at measured wrist link; no calibrated BrainCo grasp point, contact force or optical validation.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(__file__,out/'g1_audit_reach.py')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
