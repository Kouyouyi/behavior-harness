"""Fresh-start paired search OOD evaluation; no retry of valid capability failures."""
import fcntl,json,os,subprocess,time,urllib.request,traceback,html
from pathlib import Path
import run_champion_start_comparison as base
R=base.ROOT;OUT=R/'outputs/champion_search_suite_20261020';OUT.mkdir(exist_ok=True)
M=json.loads((R/'configs/champion_search_suite/manifest.json').read_text())
os.environ['CHAMPION_URL']='http://127.0.0.1:18774'
def save(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def ssh(cmd):
 a=['ssh','huoshan',cmd]
 if os.geteuid()==0:a=['runuser','-u','cody','--',*a]
 return subprocess.run(a,capture_output=True,text=True)
def health():
 with urllib.request.urlopen(os.environ['CHAMPION_URL']+'/health',timeout=10) as r:return json.load(r)
def status(active):
 records=[json.loads(p.read_text()) for p in OUT.glob('*/result.json')]
 save(OUT/'summary.json',{'active':active,'screening':M['counts'],'planned_pairs':39,'results':records})
 links=[]
 for p in sorted(OUT.glob('*/**/videos/*.mp4')):links.append(f'<li><a href="{p.relative_to(OUT)}">{p.relative_to(OUT)}</a></li>')
 (OUT/'index.html').write_text('<meta charset="utf-8"><h1>搜索 OOD：冠军直接控制 vs Harness + 冠军</h1><pre>'+html.escape(json.dumps({'active':active,'screening':M['counts'],'results':records},ensure_ascii=False,indent=2))+'</pre><ul>'+''.join(links)+'</ul>')
def reset(row):
 cp=row['checkpoint']; marker='DOWNLOAD_VERIFIED.json' if cp=='checkpoint_2' else cp+'_VERIFIED.json'
 while ssh('test -f /cs/kyy/models/behavior-2025-champion/'+marker).returncode:
  status(row['task']+' waiting for verified '+cp);time.sleep(30)
 try:old=health()['boot_id']
 except Exception:old=None
 result=ssh('python3 /cs/kyy/behavior_ap_experiment/scripts/restart_champion_suite.py '+str(row['task_id']))
 if result.returncode:raise RuntimeError(result.stderr)
 while True:
  try:
   h=health()
   if h['ready'] and h['boot_id']!=old:
    assert h['task_id']==row['task_id'] and h['checkpoint']==cp and h['task_name']==row['task'];return h
  except (OSError,ValueError):pass
  time.sleep(10)
def available():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
 m=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
 return p.returncode==0 and not p.stdout.strip() and m.returncode==0 and int(m.stdout.splitlines()[0])>=20000
def videos(case,row):
 stages=[row[k] for k in ['search_run','completion_run'] if k in row]
 dest=case/(row['condition']+'_full')/'videos';dest.mkdir(parents=True,exist_ok=True)
 result=[]
 for name in ['robot_cameras','third_person']:
  sources=[case/s/'sim/videos'/(name+'.mp4') for s in stages]
  listing=dest/(name+'.txt');listing.write_text(''.join("file '"+str(p)+"'\n" for p in sources))
  target=dest/(name+'.mp4')
  subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(target)],check=True)
  subprocess.run(['ffmpeg','-v','error','-i',str(target),'-f','null','-'],check=True)
  result.append(str(target))
 return result
lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
save(OUT/'screening.json',M)
try:
 for task in sorted((r for r in M['tasks'] if r['status']=='eligible'),key=lambda r:(r['checkpoint']!='checkpoint_2',r['checkpoint'],r['task'])):
  case=OUT/task['task'];case.mkdir(exist_ok=True)
  if (case/'result.json').exists():continue
  base.OUT=case;base.COMMON=['--scenario-spec',task['spec'],'--goal',json.loads(Path(task['spec']).read_text())['goal']]
  rows=[]
  for condition in ['harness','direct']:
   for attempt in range(1,4):
    prefix=condition+'_v'+str(attempt)
    if (case/(prefix+'_record.json')).exists():
     r=json.loads((case/(prefix+'_record.json')).read_text());rows.append(r)
     if r['valid_evaluation']:break
     continue
    row={'condition':condition,'attempt':attempt,'task':task['task'],'valid_evaluation':False,'original_task_success':False}
    try:
     while not available():status(task['task']+' waiting for GPU');time.sleep(30)
     with (R/'outputs/ood99_gpu.lock').open('w') as gpu:
      fcntl.flock(gpu,fcntl.LOCK_EX)
      pre=case/(prefix+'_preflight')
      if pre.exists():raise RuntimeError('Existing incomplete attempt: never replay; inspect before fresh retry')
      status(task['task']+' '+prefix+' preflight')
      with (case/(prefix+'_preflight.log')).open('w') as log:
       rc=subprocess.run(['bash','scripts/run_sim.sh','--output',str(pre),*base.COMMON,'--initialize-only','--initial-visibility-audit','--max-steps','10000'],cwd=R,stdout=log,stderr=subprocess.STDOUT).returncode
      if rc or (pre/'error.json').exists():raise RuntimeError('preflight initialization or scene validity failed; see log')
      remaining=10000;extra=[]
      if condition=='harness':
       status(task['task']+' '+prefix+' search')
       name=prefix+'_search';a,valid=base.run(name,['--preflight-dir',str(pre),'--checkpoint-step','24000','--max-steps','800','--max-rounds','100','--planner-max-chunks','4'])
       row.update(search_run=name,search_steps=a['acknowledged_steps'],search_success=a['find_and_approach_success'],total_steps=a['acknowledged_steps'])
       if not valid:raise RuntimeError('search infrastructure/audit failure')
       if not a['find_and_approach_success']:
        row.update(valid_evaluation=True,status='search_not_successful',reason='search budget exhausted or planner stopped before strict approach success')
       else:remaining-=a['acknowledged_steps'];extra=['--resume-handoff',str(case/name/'task_handoff')]
      if not row['valid_evaluation']:
       status(task['task']+' '+prefix+' resetting champion');row['model_health']=reset(task)
       status(task['task']+' '+prefix+' champion execution')
       name=prefix+'_completion';a,valid=base.run(name,['--preflight-dir',str(pre),'--champion-executor','--max-steps',str(remaining),*extra])
       row.update(completion_run=name,completion_steps=a['acknowledged_steps'],total_steps=a['acknowledged_steps']+row.get('search_steps',0))
       if condition=='harness':
        h=a.get('handoff_restore') or {};valid=valid and h.get('full_state_checked') and h.get('physics_steps_added')==0
       if not valid:raise RuntimeError('completion infrastructure/audit failure')
       row.update(valid_evaluation=True,original_task_success=bool(a['original_task_success']),status='success' if a['original_task_success'] else 'task_not_successful',reason=(a['control_summary'] or {}).get('reason'))
      row['full_videos']=videos(case,row)
    except Exception as e:
     row.update(valid_evaluation=False,status='infrastructure_or_scene_failed',reason=str(e),traceback=traceback.format_exc())
    save(case/(prefix+'_record.json'),row);rows.append(row);status(task['task']+' '+prefix+' finished')
    if row['valid_evaluation']:break
  save(case/'result.json',{'task':task,'conditions':rows});status(task['task']+' pair complete')
 status('complete')
except BaseException:
 save(OUT/'runner_error.json',{'traceback':traceback.format_exc()});raise
