#!/usr/bin/env python3
"""Wait for Qwen, prove current-frame inference and new worker path, run three fresh trials."""
import argparse,fcntl,json,os,signal,subprocess,threading,time
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler
import run_behavior_experiment as base
from services.behavior_ap.qwen_monitor import QwenMonitor
from report_behavior_vla_suite import write_report

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    root=a.output.resolve();protocol_path=root/'protocol.json';protocol=base.read_json(protocol_path)
    os.environ['SEEKACT_AP_CHECKPOINT']=protocol['vla_tools']['ap_checkpoint']
    os.environ['SEEKACT_AP_URL']=protocol['vla_tools']['ap_url']
    os.environ['SEEKACT_CHAMPION_URL']=protocol['vla_tools']['champion_url']
    from services.behavior_ap.wallboard import ensure_wallboard
    ensure_wallboard(root)
    child=None;stop=threading.Event()
    def cancel(*_):
        stop.set()
        if child and child.poll()is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    def status(state,**kw):
        base.save(root/'study_status.json',dict(state=state,updated_at_utc=base.utc_now(),**kw));write_report(root)
    def run(command,label):
        nonlocal child
        status(label)
        with(root/(label+'_supervisor.log')).open('x')as log:
            child=subprocess.Popen(command,cwd=base.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=str(base.ROOT),RAX_ARTIFACT_ROOT=str(root)),start_new_session=True)
            base.save(root/'study_child.json',dict(phase=label,pid=child.pid,command=command))
            code=child.wait();child=None
        if stop.is_set():raise KeyboardInterrupt
        if code:raise RuntimeError(label+' failed; preserve evidence and use a fresh attempt after repair')
    with(root/'study.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        options=protocol['vla_tools']['qwen_monitor'];opener=build_opener(ProxyHandler({}))
        status('waiting_for_qwen_service',physical_trials_started=0)
        while not stop.is_set():
            try:
                req=Request(options['base_url']+'/models',headers={'Authorization':'Bearer '+Path(options['api_key_file']).read_text().strip()})
                with opener.open(req,timeout=10)as response:models=json.load(response)['data']
                ids=[m['id']for m in models]
                selected=options.get('model')
                candidates=([selected] if selected in ids else []) if selected else (ids if len(ids)==1 else [m for m in ids if 'qwen' in m.lower() and any(s in m.lower()for s in ('vl','vision','omni'))])
                if len(candidates)!=1:raise ValueError('Cannot uniquely select visual Qwen model')
                options['model']=candidates[0]
                base.save(protocol_path,protocol);base.save(root/'protocol_current.json',protocol)
                break
            except Exception as exc:
                status('waiting_for_qwen_service',error_type=type(exc).__name__,physical_trials_started=0)
                stop.wait(30)
        if stop.is_set():status('cancelled');return
        status('qwen_image_preflight')
        # Synthetic RGB smoke checks endpoint compatibility only, not task completion.
        # Real fresh robot images are checked by the monitored-worker calibration below.
        from PIL import Image
        d=root/'qwen_image_preflight';d.mkdir(exist_ok=False)
        paths=[]
        for role in ('head','left_wrist','right_wrist'):
            image=d/(role+'.png');Image.new('RGB',(224,224),(128,128,128)).save(image);paths.append(str(image))
        QwenMonitor(options).check(dict(step=0,image_paths=paths),'拿起一只鞋，鞋离地并受夹爪支撑。',d,stop)
        health={}
        for kind,url in [('ap',protocol['vla_tools']['ap_url']),('champion',protocol['vla_tools']['champion_url'])]:
            with opener.open(url+'/health',timeout=10)as response:health[kind]=json.load(response)
            assert health[kind]['ready']
        base.save(root/'models_ready.json',dict(status='verified',current_health=health,
            qwen_model=options['model'],qwen_current_frame_inference_verified=True))
        run([str(base.SIM_PYTHON),str(base.ROOT/'scripts/smoke_behavior_vla_worker.py'),'--output',str(root/'worker_calibration_v1'),'--vla-only','--closed-loop','--interaction-only'],'baseline_calibration')
        run([str(base.SIM_PYTHON),str(base.ROOT/'scripts/smoke_behavior_qwen_worker.py'),'--output',str(root/'qwen_worker_calibration_v1'),'--protocol',str(protocol_path)],'qwen_worker_calibration')
        run([str(base.AGENT_PYTHON),'-u',str(base.ROOT/'scripts/run_behavior_vla_suite.py'),'--protocol',str(protocol_path),'--attempt','qwen_interaction_v1'],'suite')
        run([str(base.SIM_PYTHON),str(base.ROOT/'scripts/audit_behavior_vla_suite.py'),'--output',str(root)],'audit')
        audit=base.read_json(root/'execution_audit.json');assert audit['sealed_trials']==3 and audit['all_sealed_audits_passed']
        videos=[]
        for trial in audit['trials']:
            for role,info in trial['videos'].items():
                path=root/trial['task']/trial['attempt']/'sim/videos'/(role+'.mp4')
                check=subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:v:0','-f','null','-'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
                assert check.returncode==0 and not check.stderr
                videos.append(dict(path=str(path),sha256=info['sha256'],full_decode_passed=True))
        assert len(videos)==6
        base.save(root/'final_acceptance.json',dict(status='complete',tasks_evaluated=3,
            official_successes=sum(x['official_success']for x in audit['trials']),trials=audit['trials'],
            videos=videos,native_motion_steps=0,total_step_budget=None,wall_limit_seconds=None))
        status('complete')

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:pass
    except Exception as exc:
        import sys
        root=Path(sys.argv[sys.argv.index('--output')+1])
        base.save(root/'study_error.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
