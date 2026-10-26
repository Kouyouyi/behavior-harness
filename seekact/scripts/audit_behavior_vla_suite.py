#!/usr/bin/env python3
"""Read-only final episode audit; never operates the simulator."""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def audit(root):
 current=root/'protocol_current.json'
 active={t['task_name'] for t in read(current)['tasks']} if current.exists()else None
 rows=[]
 for recpath in sorted(root.glob('*/zero_shot_*/trial_record.json')):
  if active is not None and recpath.parent.parent.name not in active:continue
  d=recpath.parent;r=read(recpath);events=[json.loads(x) for x in (d/'execution.jsonl').read_text().splitlines()]
  sub=[x for x in events if x['event']=='submitted'];acks=[x for x in events if x['event']=='ack']
  actions=np.load(d/'actions.npy');errors=[];counts=Counter();chunks=Counter();seen=set();cache={}
  if [x['sequence'] for x in sub]!=list(range(r['steps'])):errors.append('submission_sequence')
  if [x['sequence'] for x in acks]!=list(range(r['steps'])):errors.append('ack_sequence')
  if [x['step_after'] for x in acks]!=list(range(1,r['steps']+1)):errors.append('ack_steps')
  if len(actions)!=r['steps'] or not np.array_equal(actions,np.asarray([x['action'] for x in sub],dtype=np.float32)):errors.append('saved_actions')
  for x in sub:
   o=x.get('owner',{});c=o.get('controller','native_tools');counts[c]+=1
   if c not in ('ap_vla','champion_vla'):continue
   key=(o['job_id'],o['chunk']);offset=o['offset'];identity=key+(offset,)
   if identity in seen:errors.append('duplicate_prefix_offset')
   seen.add(identity)
   if key not in cache:
    p=d/'vla'/key[0]/f'chunk_{key[1]:04d}'
    a=np.load(p/'predicted_actions.npy');response=np.asarray(read(p/'response.json')['actions'],dtype=np.float32)
    if not np.array_equal(a,response):errors.append('raw_model_output_changed')
    prefix=read(p/'execution_prefix.json');expected=8 if c=='ap_vla' else 20
    if r.get('vla_tools',{}).get('closed_loop'):
     job=read(p/'job.json');expected=min(expected,job['step_budget']-job['executed_steps'])
    identity=read(p/'request_identity.json');request=read(p/'request.json');observation=read(p/'observation/observation.json')
    if sha(p/'request.json')!=identity['sha256']:errors.append('request_bytes_changed')
    if identity['observation_step']!=prefix['step_before'] or observation['step']!=prefix['step_before']:errors.append('stale_observation')
    if request['state']!=observation['state']:errors.append('request_state_changed')
    if prefix['step_after']-prefix['step_before']!=prefix['executed_prefix']:errors.append('prefix_steps')
    if prefix['planned_prefix']!=expected or not prefix['all_actions_acknowledged']:errors.append('prefix_contract')
    cache[key]=(a,prefix);chunks[c]+=1
   a,prefix=cache[key]
   if offset>=prefix['executed_prefix'] or not np.array_equal(np.asarray(x['action'],dtype=np.float32),a[offset]):errors.append('model_prefix_changed')
  for key,(a,prefix) in cache.items():
   if sum(1 for ident in seen if ident[:2]==key)!=prefix['executed_prefix']:errors.append('prefix_count')
  strict=bool(r.get('vla_tools',{}).get('vla_only'))
  handoffs=r['episode_closed'].get('vla_evidence',{}).get('handoffs',[])
  if strict:
   if sum(counts.get(k,0)for k in ('ap_vla','champion_vla'))!=r['steps']:errors.append('non_vla_motion_in_strict_trial')
   last_ap_after=0
   for x in sub:
    controller=x.get('owner',{}).get('controller')
    if controller=='ap_vla':last_ap_after=x['step_before']+1
    elif controller=='champion_vla' and not any(last_ap_after>0 and last_ap_after<=h['step']<=x['step_before']for h in handoffs):
     errors.append('champion_without_fresh_post_ap_handoff')
  decisions=r['episode_closed'].get('vla_evidence',{}).get('decisions',[])
  if r.get('vla_tools',{}).get('closed_loop'):
   jobs=r['episode_closed']['vla_evidence']['completed_jobs']
   if len(decisions)!=len(jobs):errors.append('missing_scheduling_decision')
   for decision,job in zip(decisions,jobs):
    if decision['job_id']!=job['job_id'] or decision['step']!=job['start_step'] or job['executed_steps']>decision['step_budget']:errors.append('scheduling_budget_or_identity')
    observation=read(d/'observations'/decision['frame_id']/'agent_observation.json')
    if observation['step']!=decision['step']:errors.append('stale_planner_decision_frame')
    persisted=read(d/'vla'/('decision_'+job['job_id'].split('_')[1]+'.json'))
    if persisted!=decision:errors.append('scheduler_evidence_changed')
  if r.get('vla_tools',{}).get('interaction_only'):
   assessments=r['episode_closed']['vla_evidence'].get('interaction_assessments',[])
   jobs=r['episode_closed']['vla_evidence']['completed_jobs']
   for index,job in enumerate(jobs):
    if job['kind']!='champion':continue
    end=job['start_step']+job['executed_steps']
    matching=[a for a in assessments if a['step']==end]
    if index<len(jobs)-1:
     if len(matching)!=1:errors.append('missing_gpt_interaction_assessment');continue
     a=matching[0]
     if a['outcome'] in ('completed','need_ap') and jobs[index+1]['kind']!='ap':errors.append('completed_interaction_not_returned_to_ap')
     if a['outcome']=='unsafe':errors.append('motion_after_unsafe_interaction')
    for a in matching:
     if a.get('source')=='qwen_two_consecutive_current_frames':
      checks=a.get('checks',[])
      if len(checks)!=2 or any(c.get('verdict')!='已完成'for c in checks):errors.append('invalid_qwen_completion_votes')
      elif not checks[0]['observation_step']<checks[1]['observation_step']==a['step']:errors.append('stale_qwen_completion_frames')
      for check in checks:
       q=Path(check['evidence_directory'])
       if not q.resolve().is_relative_to((d/'vla').resolve()):raise ValueError('Qwen evidence outside trial')
       saved=read(q/'verdict.json');obs=read(q/'observation/observation.json')
       if saved['verdict']!=check['verdict'] or obs['step']!=check['observation_step'] or sha(q/'current_frame.jpg')!=check['frame_sha256']:errors.append('qwen_evidence_changed')
       req=read(q/'request.json')
       if len(req['messages'])!=1 or len(req['messages'][0]['content'])!=2:errors.append('qwen_input_contract')
     else:
      frame=read(d/'observations'/a['frame_id']/'agent_observation.json')
      if frame['step']!=a['step']:errors.append('stale_interaction_assessment')
  if not r['episode_closed']['execution_integrity']:errors.append('worker_integrity')
  videos={}
  for role,v in r['videos']['videos'].items():
   p=Path(v['path']);videos[role]={'sha256':sha(p),'bytes':p.stat().st_size,'frames':v['actual_frames'],'full_decode_passed':v['full_decode_passed']}
   if not v['full_decode_passed'] or v['actual_frames']!=r['steps']:errors.append('video_integrity')
  rows.append({'task':r['task_name'] if 'task_name'in r else d.parent.name,'attempt':d.name,'status':r['status'],'evaluation_valid':r['evaluation_valid'],'official_success':r['official_success'],'steps':r['steps'],'wall_seconds':r['supervisor_timing']['total_seconds'],'controller_steps':dict(counts),'model_chunks':dict(chunks),'scheduling_decisions':len(decisions),'policy_switches':sum(a['kind']!=b['kind']for a,b in zip(decisions,decisions[1:])),'learned_jobs':r['episode_closed']['vla_evidence']['jobs_started'],'vla_only':strict,'interaction_assessments':r['episode_closed'].get('vla_evidence',{}).get('interaction_assessments',[]),'visual_handoffs':handoffs,'passed':not errors,'errors':sorted(set(errors)),'videos':videos,'record_sha256':sha(recpath)})
 result={'scope':'Sealed current trials only; no simulation or task success inferred from integrity','sealed_trials':len(rows),'all_sealed_audits_passed':bool(rows)and all(x['passed']for x in rows),'trials':rows}
 (root/'execution_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);audit(p.parse_args().output)
