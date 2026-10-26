#!/usr/bin/env python3
"""Own only this study's calibration/suite; never resume partial physical actions."""
import argparse,json,os,signal,subprocess,sys,datetime
from pathlib import Path
import run_behavior_experiment as base

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--closed-loop',action='store_true');p.add_argument('--interaction-only',action='store_true');a=p.parse_args();root=a.output.resolve();task_count=len(base.read_json(root/'protocol.json')['tasks']);child=None;phase=None;cancel=False
 def stop(*_):
  nonlocal cancel
  cancel=True
  if child and child.poll()is None:child.send_signal(signal.SIGINT if phase=='calibration' else signal.SIGTERM)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def status(state,**kw):base.save(root/'study_status.json',dict(state=state,updated_at_utc=base.utc_now(),**kw))
 def run(command,label):
  nonlocal child,phase
  phase=label
  with(root/(label+'_supervisor.log')).open('x')as log:
   child=subprocess.Popen(command,cwd=base.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(base.ROOT),RAX_ARTIFACT_ROOT=str(root)),start_new_session=True)
   base.save(root/'study_child.json',{'phase':label,'pid':child.pid,'command':command});code=child.wait();child=None
  if code:raise RuntimeError(label+' did not complete normally: '+str(code))
 status('calibration')
 run([str(base.SIM_PYTHON),str(base.ROOT/'scripts/smoke_behavior_vla_worker.py'),'--output',str(root/'worker_calibration_v1'),'--vla-only']+(['--closed-loop']if a.closed_loop else [])+(['--interaction-only']if a.interaction_only else []),'calibration')
 if cancel:status('cancelled');return
 c=base.read_json(root/'worker_calibration_v1/acceptance.json');assert c['vla_only']and c['native_motion_rejected_without_physical_advance']
 status('running_fresh_trials',planned_tasks=task_count)
 run([str(base.AGENT_PYTHON),'-u',str(base.ROOT/'scripts/run_behavior_vla_suite.py'),'--protocol',str(root/'protocol.json'),'--attempt','interaction_v1'if a.interaction_only else 'closed_loop_v1'if a.closed_loop else 'vla_only_v1'],'suite')
 if cancel:status('cancelled');return
 status('auditing')
 run([str(base.SIM_PYTHON),str(base.ROOT/'scripts/audit_behavior_vla_suite.py'),'--output',str(root)],'action_audit')
 audit=base.read_json(root/'execution_audit.json');assert audit['sealed_trials']==task_count and audit['all_sealed_audits_passed']
 assert all(x['evaluation_valid'] and not x['controller_steps'].get('native_tools',0)for x in audit['trials'])
 videos=[]
 for trial in audit['trials']:
  for role,info in trial['videos'].items():
   path=root/trial['task']/trial['attempt']/'sim/videos'/(role+'.mp4')
   check=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:v:0','-f','null','-'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
   videos.append({'task':trial['task'],'role':role,'path':str(path.relative_to(root)),'sha256':info['sha256'],'full_decode_passed':check.returncode==0 and not check.stderr,'error':check.stderr[:1500]})
 evidence={'count':len(videos),'all_passed':len(videos)==2*task_count and all(v['full_decode_passed']for v in videos),'videos':videos};base.save(root/'independent_video_verification.json',evidence);assert evidence['all_passed']
 result={'status':'complete','control_mode':'Repeated visual closed-loop AP/champion scheduling'if a.closed_loop else 'VLA-only AP search then matching champion execution','tasks_evaluated':task_count,'valid_trials':task_count,'official_successes':sum(t['official_success']for t in audit['trials']),'official_failures':sum(not t['official_success']and t['status']!='task_timeout'for t in audit['trials']),'wall_timeouts':sum(t['status']=='task_timeout'for t in audit['trials']),'native_motion_steps':0,'trials':audit['trials'],'not_statistical_success_rate':True,'visual_handoff_is_planner_assessment_not_official_success':True,'all_videos_independently_decoded':True}
 base.save(root/'final_acceptance.json',result)
 from report_behavior_vla_suite import write_report
 write_report(root);status('complete',official_successes=result['official_successes'])

if __name__=='__main__':
 try:main()
 except Exception as e:
  print(json.dumps({'infrastructure_error':str(e)}),flush=True);raise
