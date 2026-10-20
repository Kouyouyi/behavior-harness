import ast,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];C=R/'configs/champion_search_suite'
node=next(n for n in ast.parse((C/'eval_utils_2025_source.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TASK_NAMES_TO_INDICES' for t in n.targets))
ids=ast.literal_eval(node.value);mapping=json.loads((C/'task_checkpoint_mapping.json').read_text())['checkpoints']
rows=[]
for p in sorted((R/'configs/ood99').glob('*.json')):
 s=json.loads(p.read_text());t=s['task_name'];inside=any(a['relation']=='inside' for v in s['placements'] for a in v['ancestors'])
 row=dict(task=t,source_spec=str(p),task_id=ids.get(t),container_ancestry=inside)
 if inside:row.update(status='excluded_container_demonstration_unverified',reason='No verified matching container opening demonstration; conservatively excluded, not proof of absent data.')
 elif t not in ids:row.update(status='unsupported_champion_task',reason='Published champion has 50 fixed task embeddings; task is absent.')
 else:
  cp=next(k for k,v in mapping.items() if ids[t] in v['tasks']);dest=C/(t+'.json');dest.write_text(json.dumps(s,ensure_ascii=False,indent=2));row.update(status='eligible',checkpoint=cp,spec=str(dest))
 rows.append(row)
from collections import Counter
out={'counts':dict(Counter(r['status'] for r in rows)),'tasks':rows,'protocol':{'total_steps_each':10000,'search_budget':800,'harness':'AP checkpoint024000 then task-mapped champion','direct':'same task-mapped champion from fresh start','original_success':'official BDDL','retry':'infrastructure only, fresh attempt, never partial action replay','scope':'99 existing per-task candidates, container ancestors conservatively excluded'}}
(C/'manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(out['counts'])
