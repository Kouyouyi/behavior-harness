#!/usr/bin/env python3
"""30 real calibration steps only; never counted as a task evaluation."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import run_behavior_experiment as base
from services.behavior_ap.backend import request
from services.behavior_ap.timing import WallClockLedger


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--vla-only',action='store_true');p.add_argument('--closed-loop',action='store_true');p.add_argument('--interaction-only',action='store_true');args=p.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=False)
    sock=Path('/tmp')/('symb-vla-cal-'+uuid.uuid4().hex[:16]+'.sock')
    options={'enabled':True,'ap_url':os.environ.get('SEEKACT_AP_URL','http://127.0.0.1:18770'),'champion_url':os.environ.get('SEEKACT_CHAMPION_URL','http://127.0.0.1:18778'),
             'champion_skills':{'putting_shoes_on_rack':22},'champion_cross_task_transfer':False}
    options['vla_only']=args.vla_only
    options['closed_loop']=args.closed_loop
    options['interaction_only']=args.interaction_only
    if args.closed_loop:assert args.vla_only
    config={'output':str(out),'socket_path':str(sock),'uid':os.getuid(),'gid':os.getgid(),
            'task':'putting_shoes_on_rack','prompt':'Calibration only; not task evaluation.',
            'max_steps':None if args.interaction_only else 10000,'seed':20261020,'vla_tools':options,
            'deadline_monotonic':time.monotonic()+900 if args.interaction_only else None}
    base.save(out/'worker_config.json',config);ledger=WallClockLedger(out/'supervisor_timing')
    worker=None;error=None
    def call(name,arguments):
        value,images=request(sock,name,arguments,timeout=90)
        ledger.emit('rpc_result',tool=name,images=len(images),value=value)
        return value
    try:
        with (out/'worker.log').open('x')as log:
            worker=subprocess.Popen(base.worker_command(out/'worker_config.json'),cwd=base.ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        base.save(out/'processes.json',{'worker_supervisor_pid':worker.pid})
        with ledger.span('initialization'):
            base.wait_for_ready(worker,out,sock,'putting_shoes_on_rack')
        jobs=[('behavior_active_perception_vla',{'motion_tags':['turn_left'],'target_description':'鞋子所在的地面','chunks':1}),
                               ('behavior_champion_vla',{'skill':'putting_shoes_on_rack','chunks':1})]
        if args.closed_loop:jobs=jobs+[(name,dict(arguments))for name,arguments in jobs]
        for job_index,(name,arguments) in enumerate(jobs):
            if args.vla_only and name=='behavior_champion_vla':
                # Integration fixture only, never a claim that the shoe task target was found.
                frame=call('behavior_observe',{})['frame_id']
                before=call('behavior_state',{})['step']
                call('behavior_handoff_to_champion',{'frame_id':frame,'target_description':'calibration floor region',
                    'reason':'Integration-only handoff fixture after AP; this is not a benchmark target-found or task-success claim.'})
                assert call('behavior_state',{})['step']==before
            if args.closed_loop:
                arguments=dict(arguments);arguments.pop('chunks')
                arguments.update(frame_id=call('behavior_observe',{})['frame_id'],step_budget=[129,141,5,9][job_index],
                    visual_progress='initial'if job_index==0 else 'no_progress',
                    reason='Integration fixture verifies bounded execution and re-perception, not task performance',next_goal='Validate exact prefix and owner switch')
            if args.interaction_only and name=='behavior_champion_vla':arguments['interaction_goal']='Calibration current contact only'
            result=call(name,arguments);ident=result['job_id']
            while True:
                result=call('behavior_vla_continue',{'job_id':ident,'wait_seconds':20})
                j=result.get('vla_job',result)
                if j['done']:break
            if args.interaction_only and name=='behavior_champion_vla':
                try:call('behavior_champion_vla',arguments)
                except RuntimeError as exc:assert 'assess_interaction' in str(exc)
                else:raise AssertionError('Unassessed champion continuation was admitted')
                frame=call('behavior_observe',{})['frame_id']
                call('behavior_assess_interaction',{'frame_id':frame,'outcome':'completed','evidence':'Integration fixture only; verifies return to AP, not a task-success claim'})
        if args.vla_only:
            try:call('behavior_move_base',{'vx':0,'vy':0,'yaw_rate':0,'steps':2})
            except RuntimeError as exc:assert 'disabled' in str(exc)
            else:raise AssertionError('Native motion was admitted')
        else:call('behavior_move_base',{'vx':0,'vy':0,'yaw_rate':0,'steps':2})
        assert call('behavior_state',{})['step']==(284 if args.closed_loop else 28 if args.vla_only else 30)
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        if worker is not None:
            base.finalize_worker(worker,sock,ledger)
        timing=ledger.finish({'calibration_not_task_evaluation':True,'error':error})
    total=284 if args.closed_loop else 28 if args.vla_only else 30
    closed=base.read_json(out/'episode_closed.json');assert closed['steps']==total and closed['execution_integrity']
    import numpy as np
    raw=[json.loads(x)for x in(out/'execution.jsonl').read_text().splitlines()]
    sent=[x for x in raw if x['event']=='submitted'];counts=Counter(x['owner']['controller']for x in sent)
    assert counts==({'ap_vla':134,'champion_vla':150} if args.closed_loop else {'ap_vla':8,'champion_vla':20} if args.vla_only else {'ap_vla':8,'champion_vla':20,'native_tools':2})
    a=np.load(out/'actions.npy')
    if not args.vla_only:assert np.array_equal(a[27,[14,22]],a[28,[14,22]]) and np.array_equal(a[28,[14,22]],a[29,[14,22]])
    for jobdir in sorted((out/'vla').glob('job_*')):
        for chunk in sorted(jobdir.glob('chunk_*')):
            prediction=np.load(chunk/'predicted_actions.npy')
            prefix=base.read_json(chunk/'execution_prefix.json')
            first=prefix['step_before'];count=prefix['executed_prefix']
            assert np.array_equal(a[first:first+count],prediction[:count])
    if args.closed_loop:
        evidence_vla=base.read_json(out/'vla/summary.json')
        assert [d['kind']for d in evidence_vla['decisions']]==['ap','champion','ap','champion']
        assert [j['executed_steps']for j in evidence_vla['completed_jobs']]==[129,141,5,9]
        assert len(evidence_vla['handoffs'])==2
    evidence={'status':'passed','physical_steps':total,'not_a_task_evaluation':True,'controller_steps':dict(counts),
        'interaction_only':args.interaction_only,'gpt_assessment_gate_verified':args.interaction_only,
        'closed_loop':args.closed_loop,'repeated_policy_switching_verified':args.closed_loop,
        'vla_only':args.vla_only,'native_motion_rejected_without_physical_advance':args.vla_only,
        'raw_prefixes_equal_executed_actions':True,'native_handback_preserves_gripper_commands':None if args.vla_only else True,
        'video_verification':closed['videos'],'wall_seconds':timing['total_seconds'],'worker_exit_code':worker.returncode}
    base.save(out/'acceptance.json',evidence);print(json.dumps(evidence,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
