#!/usr/bin/env python3
"""Real monitored champion job, before benchmark trials. No task success claim."""
import argparse,os,subprocess,uuid
from pathlib import Path
import run_behavior_experiment as base
from services.behavior_ap.backend import request
from services.behavior_ap.timing import WallClockLedger

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--protocol',required=True,type=Path);a=p.parse_args()
    out=a.output.resolve();out.mkdir(exist_ok=False)
    protocol=base.read_json(a.protocol)
    options=dict(protocol['vla_tools'],champion_skills={'putting_shoes_on_rack':22})
    socket=Path('/tmp')/('symb-qwen-cal-'+uuid.uuid4().hex[:16]+'.sock')
    config=dict(output=str(out),socket_path=str(socket),uid=os.getuid(),gid=os.getgid(),
        task='putting_shoes_on_rack',prompt='Integration calibration only, not evaluation',
        max_steps=None,seed=20261020,deadline_monotonic=None,wall_limit_seconds=None,vla_tools=options)
    base.save(out/'worker_config.json',config)
    ledger=WallClockLedger(out/'supervisor_timing');worker=None
    def call(name,**kw):return request(socket,name,kw,timeout=90)[0]
    def execute(name,budget,**kw):
        frame=call('behavior_observe')['frame_id']
        j=call(name,frame_id=frame,step_budget=budget,visual_progress='initial',
            reason='Integration calibration only',next_goal='Verify monitored action ownership',**kw)
        while not j['done']:
            r=call('behavior_vla_continue',job_id=j['job_id'],wait_seconds=20);j=r.get('vla_job',r)
        return j
    try:
        with (out/'worker.log').open('x') as log:
            worker=subprocess.Popen(base.worker_command(out/'worker_config.json'),cwd=base.ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        base.save(out/'processes.json',{'worker_supervisor_pid':worker.pid})
        base.wait_for_ready(worker,out,socket,'putting_shoes_on_rack')
        execute('behavior_active_perception_vla',8,motion_tags=['turn_left'],target_description='地面鞋子')
        call('behavior_handoff_to_champion',frame_id=call('behavior_observe')['frame_id'],
            target_description='calibration floor',reason='Integration fixture only, not target found claim')
        j=execute('behavior_champion_vla',40,skill='putting_shoes_on_rack',
            interaction_goal='拿起当前地面的一只鞋，鞋离开地面并由夹爪支撑。')
        state=call('behavior_state')
        assert j['executed_steps']==40 and state['step']==48
    finally:
        if worker:base.finalize_worker(worker,socket,ledger)
        ledger.finish({'not_task_evaluation':True})
    evidence=base.read_json(out/'vla/summary.json')
    checks=evidence['qwen_checks'];assert len(checks)==2
    assert [x['observation_step']for x in checks]==[28,48]
    assert all(x['verdict']in ('继续','已完成') for x in checks)
    closed=base.read_json(out/'episode_closed.json');assert closed['execution_integrity'] and closed['videos']['passed']
    import numpy as np
    actions=np.load(out/'actions.npy')
    for d in (out/'vla').glob('job_*/chunk_*'):
        prefix=base.read_json(d/'execution_prefix.json');n=prefix['executed_prefix'];start=prefix['step_before']
        assert np.array_equal(actions[start:start+n],np.load(d/'predicted_actions.npy')[:n])
    base.save(out/'acceptance.json',dict(status='passed',not_task_evaluation=True,
        qwen_real_image_inference_verified=True,distinct_current_frame_steps=[28,48],
        physical_steps=48,raw_prefixes_equal_executed_actions=True,
        qwen_two_positive_return_observed=all(x['verdict']=='已完成'for x in checks),
        two_vote_interrupt_logic_verified_by_unit_tests=True,video_verification=closed['videos']))

if __name__=='__main__':main()
