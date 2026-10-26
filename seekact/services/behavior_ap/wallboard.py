"""SeekAct live wallboard. Reads saved artifacts only; never connects to robot RPC."""
from __future__ import annotations
import argparse,json,os,secrets,subprocess,sys,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import build_opener,ProxyHandler

STATIC=Path(__file__).with_name('wallboard_static')

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def tail(path,limit=16000):
    try:
        with Path(path).open('rb')as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-limit));raw=stream.read(limit)
        if size>limit:raw=raw.partition(b'\n')[2]
        return raw.decode(errors='replace')
    except OSError:return ''

def jsonlines(path):
    result=[]
    for line in tail(path).splitlines():
        try:result.append(json.loads(line))
        except ValueError:pass
    return result

class Artifacts:
    def __init__(self,pointer):self.pointer=Path(pointer)
    def root(self):
        value=read(self.pointer,{})
        root=Path(value['output']).resolve()
        if not(root/'protocol.json').is_file():raise ValueError('Study protocol not ready')
        return root
    def trial(self,root):
        suite=read(root/'suite_status.json',{});child=read(root/'suite_child.json',{})
        task=suite.get('task')or child.get('task')
        if task:
            choices=sorted((root/task).glob('zero_shot_*'),key=lambda p:p.stat().st_mtime_ns)
            if choices:return choices[-1]
        study=read(root/'study_status.json',{})
        folder={'baseline_calibration':'worker_calibration_v1','qwen_worker_calibration':'qwen_worker_calibration_v1'}.get(study.get('state'))
        if folder and(root/folder).is_dir():return root/folder
        return None
    def frame(self,trial):
        value=read(trial/'live_frame.json')
        if value:return value
        # Compatibility for pre-wallboard runs only. New workers publish a direct pointer.
        choices=list((trial/'observations').glob('*/observation.json'))
        choices+=list((trial/'vla').glob('job_*/qwen_*/observation/observation.json'))
        choices+=list((trial/'vla').glob('job_*/chunk_*/observation/observation.json'))
        if not choices:return None
        return read(max(choices,key=lambda p:p.stat().st_mtime_ns))
    def image(self,role):
        if role not in ('head','left_wrist','right_wrist'):raise ValueError('Invalid camera')
        root=self.root();trial=self.trial(root)
        if not trial:raise FileNotFoundError
        frame=self.frame(trial)
        if not frame:raise FileNotFoundError
        paths=frame.get('image_paths',[])
        p=next((Path(v)for v in paths if Path(v).name==role+'.png'),None)
        if not p or not p.resolve().is_relative_to(trial.resolve())or not p.is_file():raise FileNotFoundError
        return p.read_bytes()
    def snapshot(self):
        root=self.root();protocol=read(root/'protocol.json',{})
        study=read(root/'study_status.json',{});suite=read(root/'suite_status.json',{})
        result=dict(system='SeekAct',output_name=root.name,read_only=True,physics_commands=0,
            generated_at_ns=time.time_ns(),study=study,suite=suite,
            limits=dict(max_steps=protocol.get('max_steps'),wall_limit_seconds=protocol.get('wall_limit_seconds')),
            qwen_model=protocol.get('vla_tools',{}).get('qwen_monitor',{}).get('model'),tasks=[])
        for task in protocol.get('tasks',[]):
            candidates=sorted((root/task['task_name']).glob('zero_shot_*/trial_record.json'),key=lambda p:p.stat().st_mtime_ns)
            rec=read(candidates[-1],{})if candidates else{}
            result['tasks'].append(dict(name=task['task_name'],task_id=task['task_index'],
                status=rec.get('status','running'if suite.get('task')==task['task_name']else'pending'),
                official_success=rec.get('official_success'),steps=rec.get('steps')))
        trial=self.trial(root)
        if not trial:return dict(result,active=None)
        progress=read(trial/'sim/progress.json',{});closed=read(trial/'episode_closed.json',{})
        live=read(trial/'vla/live_state.json',{})
        events=jsonlines(trial/'execution.jsonl')
        acknowledged=[e for e in events if e.get('event')=='ack']
        submitted=[e for e in events if e.get('event')=='submitted']
        latest=submitted[-1]if submitted else{}
        qwen=read(trial/'vla/qwen_checks.json',[])
        frame=self.frame(trial)
        messages=tail(trial/'agent/agent_messages.md',14000)
        if not live:
            latest_jobs=sorted((trial/'vla').glob('job_*'))
            if latest_jobs:
                job=latest_jobs[-1]
                live={'job':read(job/'result.json')or read(job/'chunk_0000/job.json',{})}
        checks=(live.get('recent_qwen_checks')or qwen)[-8:]
        if not checks:
            files=sorted((trial/'vla').glob('job_*/qwen_*/verdict.json'))[-8:]
            checks=[read(p)for p in files if read(p)]
        steps=max(progress.get('step',0),closed.get('steps',0),acknowledged[-1].get('step_after',0)if acknowledged else 0)
        job=live.get('job')or{}
        if not live.get('recent_decisions'):
            live['recent_decisions']=[read(p)for p in sorted((trial/'vla').glob('decision_*.json'))[-3:]if read(p)]
        if job and not job.get('done'):
            job=dict(job,executed_steps=max(job.get('executed_steps',0),steps-job.get('start_step',steps)))
        owner=latest.get('owner',{}).get('controller','none')
        started=read(trial/'trial_started.json',{}).get('started_at_utc')
        result['active']=dict(task=trial.parent.name if trial.name.startswith('zero_shot_')else'校准（不计正式结果）',
            attempt=trial.name,steps=steps,official_success=closed.get('official_success',progress.get('success',False)),
            done=closed.get('status')=='closed',termination_reason=closed.get('termination_reason',progress.get('termination_reason')),
            controller=owner,control_status=job.get('status','waiting_for_initialization'),job=job,
            phase=live.get('phase'),active_interaction_goal=live.get('active_interaction_goal')or job.get('args',{}).get('interaction_goal'),
            qwen_completion_streak=live.get('qwen_completion_streak',0),qwen_checks=checks,
            decisions=live.get('recent_decisions',[]),messages=messages,
            frame_step=frame.get('step')if frame else None,
            frame_age_seconds=max(0,(time.time_ns()-frame.get('captured_ns',time.time_ns()))/1e9)if frame else None,
            frame_available=bool(frame),started_at_utc=started,
            last_submitted_sequence=latest.get('sequence'),last_acknowledged_sequence=acknowledged[-1].get('sequence')if acknowledged else None)
        return result

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,status,body,kind):
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:self.wfile.write(body)
        except(BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        if self.headers.get('Host')not in self.server.allowed_hosts:return self.send(403,b'Forbidden','text/plain')
        route=urlsplit(self.path).path
        try:
            if route=='/health':
                value={'role':'seekact_readonly_wallboard','ready':True};return self.send(200,json.dumps(value).encode(),'application/json')
            if route=='/api/state':return self.send(200,json.dumps(self.server.artifacts.snapshot(),ensure_ascii=False).encode(),'application/json; charset=utf-8')
            if route.startswith('/camera/')and route.endswith('.png'):
                return self.send(200,self.server.artifacts.image(route[len('/camera/'):-4]),'image/png')
            assets={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
            if route in assets:
                name,kind=assets[route];return self.send(200,(STATIC/name).read_bytes(),kind)
            self.send(404,b'Not found','text/plain')
        except(FileNotFoundError,ValueError,KeyError):self.send(404,b'Artifact not ready','text/plain')
    def do_POST(self):self.send(405,b'Read-only dashboard','text/plain')

def make_server(pointer,port=18887):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True;actual=server.server_address[1]
    server.allowed_hosts={f'127.0.0.1:{actual}',f'localhost:{actual}'}
    server.artifacts=Artifacts(pointer)
    return server

def ensure_wallboard(output,port=18887):
    root=Path(output).resolve();pointer=root.parent/'seekact_wallboard_current.json'
    service=root.parent/'seekact_wallboard_service.json'
    existing=read(service,{})
    alive=False
    if existing.get('pid'):
        try:
            args=Path(f'/proc/{existing["pid"]}/cmdline').read_bytes().decode().split('\0')
            alive='services.behavior_ap.wallboard'in args and str(pointer)in args
        except OSError:pass
    url=f'http://127.0.0.1:{port}/'
    if alive:
        with build_opener(ProxyHandler({})).open(url+'health',timeout=3)as response:
            if json.load(response).get('role')!='seekact_readonly_wallboard':raise RuntimeError('Unexpected dashboard on port')
    tmp=pointer.with_suffix('.tmp');tmp.write_text(json.dumps({'output':str(root)}));tmp.replace(pointer)
    if not alive:
        log=(root.parent/'seekact_wallboard_server.log').open('a')
        command=[sys.executable,'-m','services.behavior_ap.wallboard','--pointer',str(pointer),'--port',str(port)]
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close()
        for _ in range(30):
            if process.poll()is not None:raise RuntimeError('Wallboard exited; inspect seekact_wallboard_server.log')
            try:
                with build_opener(ProxyHandler({})).open(url+'health',timeout=1)as response:
                    if json.load(response).get('role')=='seekact_readonly_wallboard':break
            except OSError:time.sleep(.1)
        else:raise RuntimeError('Wallboard not ready')
        existing={'pid':process.pid,'url':url,'pointer':str(pointer),'read_only':True}
        service.write_text(json.dumps(existing,indent=2)+'\n')
    (root/'wallboard.json').write_text(json.dumps(dict(existing,url=url),indent=2)+'\n')
    return url

def main():
    p=argparse.ArgumentParser();p.add_argument('--pointer',required=True,type=Path);p.add_argument('--port',type=int,default=18887);a=p.parse_args()
    make_server(a.pointer,a.port).serve_forever()

if __name__=='__main__':main()
