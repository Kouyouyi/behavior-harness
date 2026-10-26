"""Compare archived gait reference peaks with native encoder inputs; no control."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.g1_sonic.state import NAMES


def audit(run):
    events=[json.loads(x) for x in (run/'events.jsonl').read_text().splitlines()]
    begin=next(x['monotonic'] for x in events if x['event'] in ('gait_started','walk_started'))
    profile=np.load(run/'gait_reference.npz')
    trace=[]
    with (run/'controller_trace.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            if row['monotonic']>=begin and len(row['encoder_obs'])>=294:trace.append(row)
    times=np.array([r['monotonic']-begin for r in trace])
    q=np.array([r['encoder_obs'][4:33] for r in trace])
    peaks={}
    for side in ('left','right'):
        i=NAMES.index(side+'_knee_joint')
        expected=int(np.argmax(profile['joint_pos'][:,i]));actual=int(np.argmax(q[:,i]))
        peaks[side]=dict(planned_peak_s=expected/50,native_peak_s=float(times[actual]),
            peak_time_difference_s=float(times[actual]-expected/50),
            planned_peak_rad=float(profile['joint_pos'][expected,i]),
            native_peak_rad=float(q[actual,i]),
            peak_value_error_rad=float(abs(q[actual,i]-profile['joint_pos'][expected,i])))
    windows=[]
    with (run/'controller_trace.jsonl.reference.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r['monotonic']>=begin and r['motion']=='streamed':windows.append(r)
    result=dict(hardware_validated=False,run=str(run),knee_reference_peaks=peaks,
        max_buffer_frames=max(r['timesteps'] for r in windows),
        buffer_shrinks_over_100_frames=sum(a['timesteps']-b['timesteps']>100 for a,b in zip(windows,windows[1:])),
        note='Native reference readback, not measured joint tracking. Peak checks alone do not prove every frame played.')
    if windows and 'source_start' in windows[0]:
        cursor=np.array([r['source_start']+r['cursor']*r['source_step'] for r in windows])
        result['source_cursor_max_forward_jump_frames']=int(max(np.diff(cursor)))
        result['source_cursor_backward_samples']=int(sum(np.diff(cursor)<0))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    a=p.parse_args()
    result=audit(a.run.resolve())
    output=a.run/'timing_audit.json'
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
