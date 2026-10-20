"""Finish first valid fresh search after current runner releases its lock; no action replay."""
import os,json,fcntl
from pathlib import Path
os.environ['CHAMPION_COMPARISON_OUTPUT']='outputs/champion_start_comparison_20261020_v3'
import run_champion_start_comparison as runner
OUT=runner.OUT
lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
original=runner.read(OUT/'summary.json')
(OUT/'summary_before_audit_repair.json').write_text(json.dumps(original,ensure_ascii=False,indent=2))
a=runner.read(OUT/'harness_v1_search/acceptance.json');s=a['control_summary']
assert a['find_and_approach_success'] and not a['error'] and not a['launcher_failure']
assert a['acknowledged_steps']==s['steps']==480
for r in a['rounds']:
 assert r['unacknowledged_submissions']==0
 if r['ack_steps']:assert r['raw_prefix_exact'] is True
 else:
  assert r['command']['status']=='reached' and r['raw_prefix_exact'] is None
  f=OUT/'harness_v1_search/control'/r['round']/'execution.jsonl'
  assert not f.exists() or not f.read_text().strip()
assert len(a['videos'])==2 and all(v['fully_decoded'] for v in a['videos'])
rows=original['conditions']
for row in rows:
 if row['condition']=='harness':row['superseded_audit_classification']=row['status'];row['note']='Zero-action reached round was incorrectly required to have an action-prefix check; original evidence retained.'
with (runner.ROOT/'outputs/ood99_gpu.lock').open('w') as gpu:
 fcntl.flock(gpu,fcntl.LOCK_EX)
 runner.publish('harness first valid search continuation after audit repair',rows)
 health=runner.restart()
 name='harness_v1_champion_auditrepair'
 b,valid=runner.run(name,['--preflight-dir',str(OUT/'preflight_v1'),'--champion-executor','--max-steps',str(10000-s['steps']),'--resume-handoff',str(OUT/'harness_v1_search/task_handoff')])
 restore=b.get('handoff_restore') or {};valid=bool(valid and restore.get('full_state_checked') and restore.get('physics_steps_added')==0)
 row={'condition':'harness','attempt':1,'search_run':'harness_v1_search','search_steps':s['steps'],'search_success':True,'completion_run':name,'completion_steps':b['acknowledged_steps'],'total_steps':s['steps']+b['acknowledged_steps'],'valid_evaluation':valid,'original_task_success':bool(valid and b['original_task_success']),'status':'success' if valid and b['original_task_success'] else ('task_not_successful' if valid else 'infrastructure_failed'),'reason':b['error'] or (b['control_summary'] or {}).get('reason'),'champion_boot_id':health['boot_id'],'previous_audit_records':[r for r in rows if r['condition']=='harness']}
 final=[row]+[r for r in rows if r['condition']=='direct'];runner.publish('complete',final)
 links=[]
 for r in final:
  for k in ['search_run','completion_run']:
   if k in r:
    for v in sorted((OUT/r[k]/'sim/videos').glob('*.mp4')):links.append(f'<li><a href="{v.relative_to(OUT)}">{r["condition"]} {k} {v.name}</a></li>')
 (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>从起点对照</h1><p>相同起点与10000步总预算。首次搜索的无动作终止回合被误判，已修复；使用首次有效搜索交接状态继续，从未重放动作。</p><pre>'+json.dumps(final,ensure_ascii=False,indent=2)+'</pre><ul>'+''.join(links)+'</ul>')
 (OUT/'audit_repair_complete.json').write_text(json.dumps({'completed':True,'valid_evaluation':valid}))
