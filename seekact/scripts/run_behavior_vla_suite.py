#!/usr/bin/env python3
"""Three fresh tool-augmented trials. Never resume physical partial execution."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_behavior_experiment as base
from report_behavior_vla_suite import write_report


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--attempt',default='tools_v1')
    p.add_argument('--retry-infrastructure-fresh-episode',action='store_true');args=p.parse_args()
    path=args.protocol.resolve();root=path.parent;protocol,tasks=base.protocol_tasks(path)
    assert protocol['vla_tools']['enabled']
    assert base.read_json(root/'models_ready.json')['status']=='verified'
    assert base.read_json(root/'worker_calibration_v1/acceptance.json')['status']=='passed'
    if protocol['vla_tools'].get('vla_only'):
        calibration=base.read_json(root/'worker_calibration_v1/acceptance.json')
        assert calibration.get('vla_only') and calibration.get('native_motion_rejected_without_physical_advance')
    if protocol['vla_tools'].get('closed_loop'):
        calibration=base.read_json(root/'worker_calibration_v1/acceptance.json')
        assert calibration.get('closed_loop') and calibration.get('repeated_policy_switching_verified')
    if protocol['vla_tools'].get('interaction_only'):
        assert base.read_json(root/'worker_calibration_v1/acceptance.json').get('gpt_assessment_gate_verified')
    if protocol['vla_tools'].get('qwen_monitor'):
        qwen=base.read_json(root/'qwen_worker_calibration_v1/acceptance.json')
        assert qwen['status']=='passed' and qwen['qwen_real_image_inference_verified']
    child=None;cancelled=False
    def stop(*_):
        nonlocal cancelled
        cancelled=True
        if child and child.poll()is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def status(state,task=None,**extra):
        base.save(root/'suite_status.json',dict(state=state,task=task,updated_at_utc=base.utc_now(),**extra));write_report(root)
    with (root/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for task in tasks:
            if cancelled:break
            current_path=path;current_task=task
            if task.get('champion_decision_required'):
                while not (root/'toy_decision.json').is_file() and not cancelled:
                    status('waiting_for_toy_skill_decision',task['task_name']);time.sleep(10)
                if cancelled:break
                decision=base.read_json(root/'toy_decision.json')
                if decision['choice']=='cross_task':
                    current_task=dict(task,champion_skills={'picking_up_toys':7,'collecting_childrens_toys':21},champion_cross_task_transfer=True)
                elif decision['choice']=='ap_only':
                    current_task=dict(task,champion_skills={},champion_cross_task_transfer=False)
                else:raise ValueError('Task replacement needs an explicit new official task protocol; no silent substitution')
                resolved=dict(protocol,tasks=[current_task if x['task_name']==task['task_name']else x for x in tasks],toy_decision=decision)
                current_path=root/'protocol_toy_resolved.json'
                if current_path.exists():assert base.read_json(current_path)==resolved
                else:base.save(current_path,resolved)
            previous=sorted((root/current_task['task_name']).glob('zero_shot_*'),key=lambda x:x.stat().st_mtime_ns)
            if previous:
                records=[base.read_optional(x/'trial_record.json')for x in previous]
                if any(x is None for x in records):raise RuntimeError('An unsealed attempt exists; verify its process before any new attempt')
                if records[-1]['evaluation_valid']:continue
                if not args.retry_infrastructure_fresh_episode:raise RuntimeError('Infrastructure attempt exists; repair and use a fresh attempt with explicit retry flag')
            destination=root/current_task['task_name']/('zero_shot_'+args.attempt)
            if destination.exists():raise FileExistsError('Never overwrite/resume '+str(destination))
            command=[str(base.AGENT_PYTHON),'-u',str(base.ROOT/'scripts/run_behavior_experiment.py'),
                '--protocol',str(current_path),'--task',current_task['task_name'],'--attempt',args.attempt,'--wait-for-runner']
            status('running',current_task['task_name'],attempt=args.attempt)
            with (root/(current_task['task_name']+'_'+args.attempt+'_supervisor.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=base.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONPATH=str(base.ROOT),RAX_ARTIFACT_ROOT=str(root)),start_new_session=True)
                base.save(root/'suite_child.json',{'pid':child.pid,'task':current_task['task_name'],'command':command})
                while child.poll()is None:
                    time.sleep(10);write_report(root)
                code=child.wait();child=None
            status('trial_sealed',current_task['task_name'],exit_code=code)
            if code:raise RuntimeError('Trial did not complete validly; preserve evidence and repair before fresh retry')
        status('cancelled'if cancelled else 'complete')
    return 130 if cancelled else 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'error':type(exc).__name__,'message':str(exc)}),flush=True);raise
