"""Fixed ten-trial OOD radio protocol, with separately audited two stages."""
import argparse,re
import fcntl,json,os,subprocess,time,urllib.request
from radio10_report import render_report
from pathlib import Path
from run_ood99 import ROOT,PY,read,execute,transport_preflight
OUT=ROOT/'outputs/radio10_20261019'

def publish(rows,active):
    data={'planned_trials':10,'completed_records':len(rows),'search_successes':sum(r.get('search_success',False) for r in rows),'original_task_successes':sum(r.get('original_task_success',False) for r in rows),'active':active,'trials':rows,'note':'Fixed cases; infrastructure failures remain unevaluated; 10/10 requires all ten valid successes.'}
    tmp=OUT/'summary.tmp';tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(OUT/'summary.json')
    render_report(OUT,data)

def audit(run,log):
    execute([PY,'scripts/audit_rollout.py',str(run)],log,180)
    a=read(run/'acceptance.json') or {}
    valid=bool(a.get('control_summary')) and bool(a.get('videos')) and not a.get('error') and not a.get('launcher_failure') and all(v['fully_decoded'] for v in a['videos']) and all(r.get('raw_prefix_exact') is not False and not r.get('unacknowledged_submissions') for r in a.get('rounds',[]))
    return a,valid

def record(rows,row):
    prior=[r for r in rows if r['trial']==row['trial']]
    if prior:row['previous_attempts']=prior
    rows[:]=[r for r in rows if r['trial']!=row['trial']]+[row]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',default='v1')
    parser.add_argument('--retry-infrastructure-fresh-episode',action='store_true')
    parser.add_argument('--wait-for-runner',action='store_true')
    options=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',options.attempt):parser.error('Invalid attempt name')
    if options.retry_infrastructure_fresh_episode and options.attempt=='v1':parser.error('Fresh retry requires a new attempt name')
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX if options.wait_for_runner else fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read(OUT/'protocol.json');rows=(read(OUT/'summary.json') or {}).get('trials',[])
    retryable={'infrastructure_failed','scene_or_initialization_invalid'}
    done={r['trial'] for r in rows if not (options.retry_infrastructure_fresh_episode and r['status'] in retryable)}
    os.environ['TASK_VLA_URL']='http://127.0.0.1:18772'
    for case in protocol['trials']:
        name=case['trial']
        if name in done:continue
        base=OUT/name;base.mkdir(exist_ok=True)
        args=['--ood',case['ood'],'--seed',str(case['seed']),'--initial-yaw-offset-deg',str(case['yaw'])]
        if case.get('container'):args+=['--container-name',case['container']]
        pre=base/('preflight_'+options.attempt);search=base/('search_'+options.attempt);completion=base/('completion_'+options.attempt)
        if options.retry_infrastructure_fresh_episode and any(p.exists() for p in (pre,search,completion)):raise RuntimeError('Fresh retry directories must not exist')
        row={'trial':name,'attempt':options.attempt,'condition':case['ood'],'seed':case['seed'],'search_success':False,'original_task_success':False}
        with (ROOT/'outputs/ood99_gpu.lock').open('w') as gpu:
            fcntl.flock(gpu,fcntl.LOCK_EX)
            publish(rows,{'trial':name,'phase':'initial_visibility'})
            if not pre.exists():
                execute(['bash','scripts/run_sim.sh','--output',str(pre),*args,'--initialize-only','--initial-visibility-audit','--max-steps','800'],OUT/(name+'_preflight_'+options.attempt+'.log'),1200)
            e=read(pre/'sim/initial_occlusion_private.json')
            if not e or e.get('target_visible') or (pre/'error.json').exists() or not (pre/'sim/result.json').exists():
                row.update(status='scene_or_initialization_invalid',reason=read(pre/'error.json') or 'Missing evidence or target initially visible');record(rows,row);publish(rows,None);continue
            transport_preflight()
            publish(rows,{'trial':name,'phase':'search'})
            if search.exists():raise RuntimeError('Existing physical attempt cannot be replayed: '+str(search))
            rc,_=execute(['bash','scripts/run_sim.sh','--output',str(search),*args,'--preflight-dir',str(pre),'--checkpoint-step','24000','--max-steps','800','--max-rounds','100','--planner-max-chunks','4','--opening-commit-chunks','12'],OUT/(name+'_search_'+options.attempt+'.log'),None)
            a,valid=audit(search,OUT/(name+'_search_audit_'+options.attempt+'.log'))
            row.update(search_run=str(search),search_success=bool(rc==0 and valid and a.get('find_and_approach_success')))
            if not valid or rc:row.update(status='infrastructure_failed',reason=a.get('error') or 'Search integrity failed')
            elif not row['search_success']:row.update(status='search_not_successful',reason=a['control_summary'].get('reason'))
            else:
                publish(rows,{'trial':name,'phase':'waiting_radio_inference'})
                while True:
                    try:
                        with urllib.request.urlopen(os.environ['TASK_VLA_URL']+'/health',timeout=10) as response:health=json.load(response)
                        if health.get('repository')!='BEHAVIOR-official-radio':raise ValueError('Wrong radio model identity')
                        proof=subprocess.run(['runuser','-u','cody','--','ssh','huoshan','cat /cs/kyy/behavior_ap_experiment/reports/radio_vla_inference_verified.json'],capture_output=True,text=True,timeout=30)
                        if proof.returncode==0 and json.loads(proof.stdout).get('remote_inference_verified'):break
                    except (OSError,subprocess.TimeoutExpired):pass
                    time.sleep(30)
                publish(rows,{'trial':name,'phase':'turn_on_radio'})
                if completion.exists():raise RuntimeError('Existing physical completion cannot be replayed')
                rc,_=execute(['bash','scripts/run_sim.sh','--output',str(completion),*args,'--preflight-dir',str(pre),'--resume-handoff',str(search/'task_handoff'),'--max-steps','10000'],OUT/(name+'_completion_'+options.attempt+'.log'),None)
                a,valid=audit(completion,OUT/(name+'_completion_audit_'+options.attempt+'.log'))
                success=bool(rc==0 and valid and a.get('original_task_success'))
                row.update(completion_run=str(completion),original_task_success=success,status='success' if success else ('task_not_successful' if valid and rc==0 else 'infrastructure_failed'),reason=a.get('error') or (a.get('control_summary') or {}).get('reason'))
            record(rows,row);publish(rows,None)
if __name__=='__main__':main()
