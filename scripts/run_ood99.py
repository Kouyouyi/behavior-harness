"""Sequential real-simulator suite. No fixture failure is scored as VLA failure."""
import argparse
import collections
import fcntl
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
PY='/home/cody/miniconda3/envs/behavior-2026/bin/python'

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def record(rows, result):
    prior=[r for r in rows if r['task']==result['task']]
    if prior:result['previous_attempts']=prior
    rows[:]=[r for r in rows if r['task']!=result['task']]+[result]

def retry_has_no_submitted_actions(row, output):
    if row['status']!='infrastructure_failed' or row.get('steps')!=0:return False
    evidence=read(output/row['run']/'acceptance.json')
    if not evidence or evidence.get('acknowledged_steps')!=0:return False
    summary=evidence.get('control_summary') or {}
    return summary.get('steps')==0 and all(r.get('ack_steps')==0 and r.get('unacknowledged_submissions')==0 for r in evidence.get('rounds',[]))

def transport_preflight():
    # No physics or policy action exists here; reconnecting cannot replay an action.
    expected='/cs/kyy/behavior_ap_experiment/runs/pi05_goal_horizon_v2_b8/checkpoint_024000'
    while True:
        try:
            with urllib.request.urlopen('http://127.0.0.1:18770/health',timeout=8) as r:health=json.load(r)
            if health.get('ready') is not True or health.get('checkpoint')!=expected:raise ValueError('Model identity mismatch')
            return health
        except (OSError,TimeoutError):
            try:
                subprocess.run(['runuser','-u','cody','--','ssh','-fN','-o','ControlMaster=no','-o','ControlPath=none','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-L','127.0.0.1:18770:127.0.0.1:18770','huoshan'],check=True,timeout=30,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            except (subprocess.SubprocessError,OSError):pass
            time.sleep(15)

def publish(output, rows, active=None):
    counts=dict(collections.Counter(r['status'] for r in rows))
    report={'total_tasks':99,'completed_records':len(rows),'active':active,'counts':counts,
            'checkpoint':'checkpoint_024000','max_steps':800,'instance_id':301,
            'objective':'find_and_approach','updated_unix':time.time(),'tasks':rows,
            'success_rate_note':'Only valid initialized OOD rollouts with complete infrastructure enter the capability denominator.'}
    tmp=output/'summary.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2));tmp.replace(output/'summary.json')
    cells=[]
    for r in rows:
        task=r['task'];link=html.escape(r.get('run',''),quote=True)
        cells.append(f'<tr><td>{html.escape(task)}</td><td>{r["status"]}</td><td>{r.get("steps",0)}</td><td>{html.escape(str(r.get("detail","")))}</td><td><a href="{link}/acceptance.json">证据</a> <a href="{link}/sim/videos/robot_cameras.mp4">视频</a></td></tr>')
    completed={r['task'] for r in rows}
    for spec in read(ROOT/'configs/ood99_manifest.json'):
        if spec['task_name'] in completed:continue
        status='running' if active and active['task']==spec['task_name'] else 'pending'
        cells.append(f'<tr><td>{spec["task_name"]}</td><td>{status}</td><td>—</td><td>目标：{spec["target_description"]}</td><td><a href="../../configs/ood99/{spec["task_name"]}.json">场景配置</a></td></tr>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>99任务主动感知测试</title><style>body{font:15px system-ui;margin:30px;color:#17212b}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}a{color:#1468a0}</style><h1>99任务主动感知真实仿真</h1><p>目标：找到并接近；每任务一个公开测试实例、一个扰动种子；不代表完整原任务完成。</p><p>当前：'+html.escape(str(active))+'</p><p>'+html.escape(json.dumps(counts,ensure_ascii=False))+'</p><table><tr><th>任务</th><th>状态</th><th>真实步数</th><th>说明</th><th>产物</th></tr>'+''.join(cells)+'</table>')
    subprocess.run([PY,str(ROOT/'scripts/summarize_ood99.py'),str(output)],check=True,stdout=subprocess.DEVNULL)

def execute(command, log, timeout):
    with log.open('w') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try: return proc.wait(timeout=timeout),False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=220)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            return proc.returncode,True

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',nargs='*');p.add_argument('--attempt',default='v2');p.add_argument('--wait-for-gpu',action='store_true');p.add_argument('--retry-zero-step-infrastructure',action='store_true');p.add_argument('--retry-infrastructure-fresh-episode',action='store_true',help='Explicitly reset failed infrastructure trials into new attempt directories; never replay a partial episode');args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'outputs/ood99_gpu.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX if args.wait_for_gpu else fcntl.LOCK_EX|fcntl.LOCK_NB)
    specs=read(ROOT/'configs/ood99_manifest.json')
    specs.sort(key=lambda s:s['task_name']!='picking_up_trash')
    if args.tasks: specs=[s for s in specs if s['task_name'] in args.tasks]
    old=read(out/'summary.json');rows=old['tasks'] if old else []
    done={r['task'] for r in rows if not ((args.retry_zero_step_infrastructure and retry_has_no_submitted_actions(r,out)) or (args.retry_infrastructure_fresh_episode and r['status'] in ('infrastructure_failed','initialization_infrastructure_failed')))}
    for spec in specs:
        if (out/'STOP_AFTER_CURRENT').exists():break
        task=spec['task_name']
        if task in done:continue
        try: transport_preflight()
        except Exception as exc:
            (out/'queue_stop.json').write_text(json.dumps({'reason':'model_transport_unavailable','message':str(exc),'remaining_tasks_not_executed':True}))
            publish(out,rows,{'task':task,'phase':'blocked_model_transport'})
            return
        config=ROOT/'configs/ood99'/f'{task}.json'
        pre=out/task/('preflight_'+args.attempt);run=out/task/('rollout_'+args.attempt)
        for candidate in sorted((out/task).glob('preflight_*'),reverse=True):
            previous=read(candidate/'sim/initial_occlusion_private.json')
            geometry=read(candidate/'sim/ood_geometry_private.json')
            if not (candidate/'error.json').exists() and read(candidate/'sim/result.json') and read(candidate/'scenario_spec.json')==spec and geometry and geometry.get('valid') and previous and previous.get('equivalent_target_evidence') and not any(r['target_visible'] for r in previous['equivalent_target_evidence']) and (candidate/'sim/initial_fingerprint_private.json').exists():
                pre=candidate;break
        publish(out,rows,{'task':task,'phase':'initialization_and_visibility'})
        if not pre.exists():
            code,timeout=execute(['bash','scripts/run_sim.sh','--output',str(pre),'--scenario-spec',str(config),'--initialize-only','--initial-visibility-audit','--max-steps','800'],out/(task+'_preflight_'+args.attempt+'.log'),1200)
        else: code,timeout=0,False
        evidence=read(pre/'sim/initial_occlusion_private.json')
        error=read(pre/'error.json')
        geometry=read(pre/'sim/ood_geometry_private.json')
        valid=code==0 and not timeout and not error and read(pre/'sim/result.json') and geometry and geometry.get('valid') and evidence and evidence.get('equivalent_target_evidence') and not any(r['target_visible'] for r in evidence['equivalent_target_evidence']) and (pre/'sim/initial_fingerprint_private.json').exists()
        if not valid:
            execute([PY,'scripts/audit_rollout.py',str(pre)],out/(task+'_preflight_audit_'+args.attempt+'.log'),180)
            message=(error or {}).get('message','')
            scene_invalid=any(x in message for x in ('No novel','OOD rejected','placement failed','original relation','Cannot open setup','Cannot close setup','starts already successful'))
            record(rows,{'task':task,'status':'scene_invalid' if scene_invalid else 'initialization_infrastructure_failed','run':str(pre.relative_to(out)),'detail':error or {'exit':code,'timeout':timeout,'has_visibility':bool(evidence)},'steps':0})
            publish(out,rows)
            if code in (245,-11,139):
                (out/'queue_stop.json').write_text(json.dumps({'reason':'native_renderer_crash_requires_repair','task':task,'remaining_tasks_not_executed':True}))
                break
            continue
        publish(out,rows,{'task':task,'phase':'planner_vla_closed_loop'})
        if run.exists():
            record(rows,{'task':task,'status':'interrupted_run_not_replayed','run':str(run.relative_to(out)),'detail':'Existing run retained; explicit new attempt required.'})
            publish(out,rows);continue
        code,timeout=execute(['bash','scripts/run_sim.sh','--output',str(run),'--scenario-spec',str(config),'--preflight-dir',str(pre),'--checkpoint-step','24000','--max-steps','800','--max-rounds','100','--planner-max-chunks','4','--opening-commit-chunks','12'],out/(task+'_rollout_'+args.attempt+'.log'),None)
        execute([PY,'scripts/audit_rollout.py',str(run)],out/(task+'_audit.log'),180)
        a=read(run/'acceptance.json') or {};summary=a.get('control_summary') or {}
        audit_issues=[]
        if any(r.get('raw_prefix_exact') is False or r.get('unacknowledged_submissions',0) for r in a.get('rounds',[])):audit_issues.append('Action receipt/raw-prefix audit failed')
        if not a.get('videos') or any(not v.get('fully_decoded') for v in a.get('videos',[])):audit_issues.append('Video missing or incomplete decode')
        transport=bool(a.get('error') or a.get('launcher_failure') or code or timeout or not summary or not (run/'preflight_match.json').exists() or audit_issues)
        status='infrastructure_failed' if transport else ('success' if a.get('find_and_approach_success') else 'not_successful_within_budget')
        record(rows,{'task':task,'status':status,'run':str(run.relative_to(out)), 'steps':a.get('acknowledged_steps',0),'unacknowledged_submissions':sum(r.get('unacknowledged_submissions',0) for r in a.get('rounds',[])),'detail':a.get('error') or audit_issues or summary.get('reason'), 'timeout':timeout,'process_returncode':code})
        publish(out,rows)
    publish(out,rows)

if __name__=='__main__':main()
