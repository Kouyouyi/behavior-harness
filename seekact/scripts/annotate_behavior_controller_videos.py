#!/usr/bin/env python3
"""Add exact action-owner badges to sealed videos; never change original evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import os
import time

ROOT=Path(__file__).resolve().parents[1]
SIM_ROOT=Path(os.environ.get('SEEKACT_SIM_ROOT',str(ROOT.parent.parent/'behavior-ap')))
sys.path[:0]=[str(ROOT),str(SIM_ROOT)]
from services.behavior_ap.video import verify_videos
from behavior_ap_sim.backend import _VideoWriter
from behavior_ap_sim.controller_overlay import add_controller_ribbon,VERSION,HEIGHT


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def action_owners(path,steps):
    submitted=[];ack=[]
    for line in path.read_text().splitlines():
        event=json.loads(line)
        if event['event']=='submitted':submitted.append(event)
        elif event['event']=='ack':ack.append(event)
    if len(submitted)!=steps or len(ack)!=steps:
        raise ValueError('Every video frame requires exactly one submitted action and ACK')
    owners=[]
    for index,(s,a) in enumerate(zip(submitted,ack)):
        if s['sequence']!=index or a['sequence']!=index or s['step_before']!=index or a['step_after']!=index+1:
            raise ValueError('Action/ACK/frame sequence mismatch; do not infer controller from time')
        owner=s.get('owner')
        if not isinstance(owner,dict) or owner.get('controller')not in ('ap_vla','champion_vla','native_tools'):
            raise ValueError('Missing/unknown authoritative controller ownership')
        owners.append(owner)
    return owners


def annotate_trial(directory):
    import av
    record=read(directory/'trial_record.json');steps=record['steps']
    if not record['evaluation_valid'] or steps<1 or not record['episode_closed']['execution_integrity']:
        raise ValueError('Annotate only sealed, valid, action-complete trials')
    owners=action_owners(directory/'execution.jsonl',steps)
    output=directory/'sim/videos_annotated';receipt=directory/'video_annotation.json'
    if receipt.exists():
        previous=read(receipt)
        if previous.get('status')!='verified' or previous.get('overlay_version')!=VERSION:
            raise ValueError('Existing annotation cannot be silently overwritten')
        if previous['execution_sha256']!=sha(directory/'execution.jsonl'):
            raise ValueError('Action evidence changed after annotation')
        for role,item in previous['videos'].items():
            if sha(directory/'sim/videos'/(role+'.mp4'))!=item['original_sha256'] or sha(output/(role+'.mp4'))!=item['annotated_sha256']:
                raise ValueError('Video bytes changed after annotation')
        return previous
    output.mkdir(exist_ok=False)
    started=time.monotonic();metadata={};sources={}
    for role in ('robot_cameras','third_person'):
        source=directory/'sim/videos'/(role+'.mp4')
        sources[role]=sha(source)
        pending=output/(role+'.partial.mp4');writer=None;count=0
        try:
            with av.open(str(source))as container:
                stream=container.streams.video[0]
                if stream.average_rate!=30:raise ValueError('Expected30fps physical process recording')
                for frame in container.decode(stream):
                    if count>=steps:raise ValueError('Video has extra frames beyond action journal')
                    rgb=add_controller_ribbon(frame.to_ndarray(format='rgb24'),owners[count],step=count+1)
                    if writer is None:writer=_VideoWriter(pending,rgb,fps=30)
                    writer.write(rgb);count+=1
                    if count%3000==0:print(json.dumps({'task':directory.parent.name,'role':role,'frames':count,'total':steps}),flush=True)
            if count!=steps:raise ValueError('Decoded video frame count differs from action journal')
        finally:
            if writer:writer.close()
        if writer is None:raise ValueError('No frames decoded')
        destination=output/(role+'.mp4');pending.replace(destination)
        metadata[role]=dict(writer.metadata(),path=str(destination))
        if sha(source)!=sources[role]:raise ValueError('Original source changed during annotation')
    verified=verify_videos({'steps':steps,'videos':metadata},steps)
    segments=[]
    for index,owner in enumerate(owners):
        if not segments or segments[-1]['controller']!=owner['controller']:
            segments.append({'first_frame':index,'last_frame':index,'first_step':index+1,'last_step':index+1,'controller':owner['controller']})
        else:segments[-1].update(last_frame=index,last_step=index+1)
    result={'status':'verified','overlay_version':VERSION,'header_pixels':HEIGHT,
        'source':'execution.jsonl submitted owner matched against every ACK',
        'alignment':'frame0 is post-action step1; label is executing controller, not planner intent',
        'steps':steps,'controller_frames':dict(Counter(o['controller']for o in owners)),
        'physical_video_duration_seconds':steps/30,'model_waits_included':False,
        'original_files_preserved':True,'execution_sha256':sha(directory/'execution.jsonl'),
        'trial_record_sha256':sha(directory/'trial_record.json'),'controller_segments':segments,
        'verification':verified,'videos':{},'wall_seconds':time.monotonic()-started}
    for role,item in verified['videos'].items():
        result['videos'][role]={'original_path':str(directory/'sim/videos'/(role+'.mp4')),
            'annotated_path':str(output/(role+'.mp4')),'original_sha256':sources[role],
            'annotated_sha256':sha(output/(role+'.mp4')),'frames':steps,'full_decode_passed':item['full_decode_passed']}
    save(receipt,result)
    print(json.dumps({'task':directory.parent.name,'status':'verified','frames':steps,'views':2}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);args=p.parse_args();root=args.output.resolve()
    records=sorted(root.glob('*/zero_shot_*/trial_record.json'))
    if not records:raise ValueError('No sealed experiment trials')
    results=[annotate_trial(path.parent)for path in records]
    save(root/'controller_video_annotations.json',{'status':'verified','trials':len(results),'videos':sum(len(r['videos'])for r in results),'overlay_version':VERSION,'results':results})
    from report_behavior_vla_suite import write_report
    write_report(root)

if __name__=='__main__':main()
