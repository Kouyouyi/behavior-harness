"""Search, preserve state, then execute the original task with public weights."""
import argparse
import fcntl
import html
import json
from pathlib import Path
import subprocess
import time
from run_ood99 import execute,read,PY,ROOT

def publish(out,rows,active=None):
    success=sum(bool(x.get('original_task_success')) for x in rows)
    data={'objective':'search_then_original_task','total_tasks':99,'completed_records':len(rows),'original_task_successes':success,'active':active,'tasks':rows,'note':'Search success and original BDDL task success are separate. Model waiting is not task failure.'}
    (out/'two_stage_results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    cells=[]
    for r in rows:
        links=[]
        for stage in ('search','completion'):
            run=r.get(stage+'_run')
            if run:
                links.append(f'<a href="{html.escape(run)}/acceptance.json">{stage}验收</a> <a href="{html.escape(run)}/sim/videos/robot_cameras.mp4">{stage}视频</a>')
        cells.append('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in ('task','search_success','original_task_success','reason'))+'<td>'+' · '.join(links)+'</td></tr>')
    (out/'two_stage_results.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>99任务两阶段验收</title><h1>搜索接近 → 原任务执行</h1><p>已记录 '+str(len(rows))+'/99，原任务成功 '+str(success)+'。等待模型不计失败。当前：'+html.escape(str(active))+'</p><table><tr><th>任务</th><th>搜索成功</th><th>原任务成功</th><th>原因</th><th>证据</th></tr>'+''.join(cells)+'</table>')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tasks',nargs='*');p.add_argument('--attempt',required=True);p.add_argument('--expert-max-steps',type=int,default=10000);args=p.parse_args()
    ready=read(ROOT/'configs/task_executor_readiness.json') or {}
    if not ready.get('remote_inference_verified') or not ready.get('cross_process_state_restore_verified'):
        raise RuntimeError('Task executor real inference and cross-process restoration must be verified first')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'outputs/two_stage_suite.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(out/'two_stage_results.json') or {};rows=old.get('tasks',[]);done={x['task'] for x in rows}
    specs=read(ROOT/'configs/ood99_manifest.json')
    for spec in specs:
        task=spec['task_name']
        if task in done or args.tasks and task not in args.tasks:continue
        publish(out,rows,{'task':task,'phase':'search'})
        execute([PY,str(ROOT/'scripts/run_ood99.py'),'--output',str(out),'--attempt',args.attempt,'--tasks',task,'--wait-for-gpu'],out/(task+'_search_driver.log'),None)
        search=next((r for r in (read(out/'summary.json') or {}).get('tasks',[]) if r['task']==task),None)
        if not search:raise RuntimeError('Search runner ended without a record')
        row={'task':task,'search_success':search['status']=='success','search_run':search['run'],'original_task_success':False,'reason':search['status']}
        if row['search_success']:
            handoff=out/search['run']/'task_handoff'
            if not (handoff/'world_state.npy').exists():raise RuntimeError('Successful search has no saved handoff')
            pre=json.loads((out/search['run']/'preflight_match.json').read_text())['preflight']
            target=out/task/('completion_'+args.attempt)
            if target.exists():raise RuntimeError('Never replay an existing task execution; choose new attempt')
            publish(out,rows,{'task':task,'phase':'original_task_execution'})
            with (ROOT/'outputs/ood99_gpu.lock').open('w') as gpu:
                fcntl.flock(gpu,fcntl.LOCK_EX)
                code,_=execute(['bash','scripts/run_sim.sh','--output',str(target),'--scenario-spec',str(ROOT/'configs/ood99'/f'{task}.json'),'--preflight-dir',pre,'--resume-handoff',str(handoff),'--max-steps',str(args.expert_max_steps)],out/(task+'_completion.log'),None)
            execute([PY,'scripts/audit_rollout.py',str(target)],out/(task+'_completion_audit.log'),180)
            a=read(target/'acceptance.json') or {}
            integrity=bool(a.get('videos')) and all(v['fully_decoded'] for v in a.get('videos',[])) and all(r.get('raw_prefix_exact') is not False and not r.get('unacknowledged_submissions') for r in a.get('rounds',[]))
            row.update(completion_run=str(target.relative_to(out)),original_task_success=bool(code==0 and not a.get('error') and integrity and a.get('original_task_success')),reason=a.get('error') or (a.get('control_summary') or {}).get('reason'),execution_integrity_passed=integrity)
        rows.append(row);publish(out,rows)

if __name__=='__main__':main()
