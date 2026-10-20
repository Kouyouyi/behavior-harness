"""Two fresh-start conditions, identical scene/seed and 10000-step total budget."""
import fcntl,json,os,subprocess,time,urllib.request,sys
from pathlib import Path
from audit_rollout import audit
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/os.environ.get('CHAMPION_COMPARISON_OUTPUT','outputs/champion_start_comparison_20261020')
PYTHON='/home/cody/miniconda3/envs/behavior-2026/bin/python'
COMMON=['--ood','radio_on_tv_cabinet_facing_room','--seed','20261020','--initial-yaw-offset-deg','180']
URL='http://127.0.0.1:18773'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def publish(active,rows):
 data={'active':active,'conditions':rows,'planned_conditions':2,'total_budget_each':10000}
 tmp=OUT/'summary.tmp';tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(OUT/'summary.json')
def run(name,args):
 p=OUT/name
 if p.exists():raise RuntimeError('Refuse physical replay: '+str(p))
 with (OUT/(name+'.log')).open('w') as log:
  rc=subprocess.run(['bash','scripts/run_sim.sh','--output',str(p),*COMMON,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
 a=audit(p)
 rows=a.get('rounds',[]);summary=a.get('control_summary') or {};videos=a.get('videos',[])
 integrity=bool(rows and all(not r['unacknowledged_submissions'] and (r.get('raw_prefix_exact') is True or r.get('executed_wrapper_prefix_exact') is True or (r['ack_steps']==0 and (r.get('command') or {}).get('status')=='reached' and r.get('raw_prefix_exact') is None)) for r in rows) and a['acknowledged_steps']==summary.get('steps'))
 vok=bool(len(videos)>=2 and all(v['fully_decoded'] and v['probe'] and all(s.get('codec_name')=='h264' and s.get('pix_fmt')=='yuv420p' for s in v['probe']['streams'] if s.get('codec_type')=='video') for v in videos))
 valid=rc==0 and integrity and vok and not a['error'] and not a['launcher_failure']
 return a,bool(valid)
def health():
 with urllib.request.urlopen(URL+'/health',timeout=10) as r:return json.load(r)
def restart():
 try:previous=health().get('boot_id')
 except Exception:previous=None
 cmd=['ssh','huoshan','python3 /cs/kyy/behavior_ap_experiment/scripts/restart_champion_comparison.py']
 if os.geteuid()==0:cmd=['runuser','-u','cody','--',*cmd]
 subprocess.run(cmd,check=True)
 while True:
  try:
   h=health()
   if h.get('ready') and h.get('boot_id')!=previous:
    assert h['checkpoint']=='checkpoint_2' and h['revision']=='89545bc1b7aa7f2e687bc0032d091f132d715d4e'
    return h
  except (OSError,ValueError):pass
  time.sleep(10)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (OUT/'summary.json').exists():raise RuntimeError('Existing comparison: choose fresh output instead of replay')
 protocol={'scene':'radio_on_tv_cabinet_facing_room','seed':20261020,'yaw':180,'initial_target_invisible_required':True,'conditions':['fresh harness AP checkpoint024000 then champion checkpoint2','fresh champion checkpoint2 directly'],'total_steps_each':10000,'harness_search_budget':800,'completion_budget':'10000 minus actual search steps','success':'official BDDL toggled_on','old_search_handoff_reused':False,'champion_state':'fresh service and reset stage/history/RNG for each condition','one_case_only':True}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 rows=[]
 with (ROOT/'outputs/ood99_gpu.lock').open('w') as gpu:
  fcntl.flock(gpu,fcntl.LOCK_EX)
  publish('fresh initial visibility preflight',rows)
  pre=OUT/'preflight_v1'
  with (OUT/'preflight_v1.log').open('w') as log:
   rc=subprocess.run(['bash','scripts/run_sim.sh','--output',str(pre),*COMMON,'--initialize-only','--initial-visibility-audit','--max-steps','10000'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
  visibility=read(pre/'sim/initial_occlusion_private.json')
  if rc or not visibility or visibility.get('target_visible') or (pre/'error.json').exists():raise RuntimeError('Invalid fresh initial scene')
  for condition in ['harness','direct']:
   for attempt in [1,2]:
    row={'condition':condition,'attempt':attempt,'search_success':None,'original_task_success':False,'valid_evaluation':False}
    prefix=condition+'_v'+str(attempt)
    if condition=='harness':
     publish(prefix+' search',rows)
     a,valid=run(prefix+'_search',['--preflight-dir',str(pre),'--checkpoint-step','24000','--max-steps','800','--max-rounds','100','--planner-max-chunks','4','--opening-commit-chunks','12'])
     row.update(search_run=prefix+'_search',search_success=a['find_and_approach_success'],search_steps=a['acknowledged_steps'])
     if not valid or not a['find_and_approach_success']:
      row.update(status='search_not_successful' if valid else 'infrastructure_failed',valid_evaluation=valid,reason=a['error'] or (a['control_summary'] or {}).get('reason'));rows.append(row);publish(None,rows)
      if valid:break
      continue
     remaining=10000-a['acknowledged_steps'];extra=['--resume-handoff',str(OUT/(prefix+'_search')/'task_handoff')]
    else:remaining=10000;extra=[]
    publish(prefix+' champion model reset',rows);h=restart();row['champion_boot_id']=h['boot_id']
    publish(prefix+' champion execution',rows)
    a,valid=run(prefix+'_champion',['--preflight-dir',str(pre),'--champion-executor','--max-steps',str(remaining),*extra])
    if condition=='harness':
     r=a.get('handoff_restore') or {};valid=bool(valid and r.get('full_state_checked') and r.get('physics_steps_added')==0)
    success=bool(valid and a['original_task_success'])
    row.update(completion_run=prefix+'_champion',completion_steps=a['acknowledged_steps'],total_steps=a['acknowledged_steps']+row.get('search_steps',0),original_task_success=success,valid_evaluation=valid,status='success' if success else ('task_not_successful' if valid else 'infrastructure_failed'),reason=a['error'] or (a['control_summary'] or {}).get('reason'))
    rows.append(row);publish(None,rows)
    if valid:break
  publish('complete',rows)
  links=[]
  for row in rows:
   for key in ['search_run','completion_run']:
    if key in row:
     for video in sorted((OUT/row[key]/'sim/videos').glob('*.mp4')):links.append(f'<li><a href="{video.relative_to(OUT)}">{row["condition"]} {key} {video.name}</a></li>')
  text=json.dumps(rows,ensure_ascii=False,indent=2)
  (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>从起点开始：Harness 与冠军直接控制</h1><p>相同场景、种子和每组10000步总预算；单个场景对照，不代表总体成功率。</p><pre>'+text+'</pre><ul>'+''.join(links)+'</ul>')
if __name__=='__main__':main()
