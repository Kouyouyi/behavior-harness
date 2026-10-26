"""Real Codex + HTTP + restricted MCP, entirely offline robot fixture (no SDKs)."""
import argparse
import base64
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.request import Request,urlopen
import uuid

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'local_tests/unit'))
from test_g1_ui_agent import FixtureBackend
from harness.codex_cli.ui_agent import Worker
from harness.codex_cli.launcher import binary
from services.controller.config import Configuration
from services.controller.frontend import Frontend
from services.controller.session import Session
from services.console.runtime import Console
from services.console.server import make_server
from services.dashboard.server import COOKIE
from services.g1_sonic.vla import atomic


class RobotFixture(FixtureBackend):
    def __init__(self,config):
        super().__init__(config);self.initialized=False;self.held=False;self.jobs={}
    def status(self):
        return dict(fixture=True,physical_robot_connected=False,phase='ready',fault=None,
                    vla=dict(initialized=self.initialized,active_job=None),motion_available=True)
    def _call(self,name,args):
        self.calls.append((name,args))
        if name=='g1_status':return self.status(),[]
        if name=='g1_observe':
            file=ROOT/'docs/g1/evidence/E115'/('held-confirmation.png' if self.held else 'before.png')
            value=dict(frame_id=uuid.uuid4().hex,fixture=True,source='recorded_test_image',
                note='Isolated fake robot acceptance. Treat this replay as the fixture observation only; all actions are simulated, never hardware.')
            return value,[dict(type='image',mimeType='image/png',data=base64.b64encode(file.read_bytes()).decode())]
        if name=='g1_vla_status':return dict(phase='ready',vla=dict(initialized=self.initialized,active_job=None,tasks={'mickey':'pick up the mickey on the table'}),fixture=True),[]
        if name=='g1_vla':
            ident=uuid.uuid4().hex
            if args['action']=='prepare':self.initialized=True
            if args['action']=='run':
                assert self.initialized and args['steps']==750
                self.held=True
            self.jobs[ident]=dict(job_id=ident,action=args['action'],completed=True,fixture=True,
                reason='initial_pose_complete' if args['action']=='prepare' else 'step_budget_complete',
                task_success_verified=False)
            return dict(job_id=ident,status='submitted',fixture=True),[]
        if name=='g1_vla_result':return self.jobs[args['job_id']],[]
        raise ValueError('Not implemented in this offline fixture: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
    run=a.run.resolve();run.mkdir(parents=True,exist_ok=False)
    atomic(run/'ankle_status.json',dict(session_id='offline-fixture',phase='ready'))
    config=Configuration(root=run/'session',task='ui_agent_offline',robot='g1_sonic',agent_provider='codex',
        task_description='这是本地隔离假机器人验收：图像来自历史回放，MCP动作只更新内存，无机器人SDK、DDS或身体发布器。用户明确授权完成假机器人抓取流程。应观察、prepare、run 750步、读取结果、观察并公开说明是离线验证。',
        backend_options=dict(control_directory=str(run)))
    path=run/'operator.json';path.write_text(json.dumps(config.to_dict(),ensure_ascii=False,indent=2))
    backend=RobotFixture(config);front=Frontend(Session(config),backend)
    address='/tmp/rax-ui-smoke-'+uuid.uuid4().hex[:12]+'.sock'
    listener=socket.socket(socket.AF_UNIX);listener.bind(address);listener.listen(8)
    stop=threading.Event();thread=threading.Thread(target=front.serve_socket,args=(listener,stop),daemon=True);thread.start()
    console=Console(mode='readonly',message_directory=run,agent_events_file=run/'agent-events.jsonl')
    http,token=make_server(port=0,console=console)
    web=threading.Thread(target=http.serve_forever,daemon=True);web.start()
    worker=Worker(config,path,address,binary());worker.start()
    def send(text,ident):
        data=json.dumps(dict(text=text,request_id=ident)).encode()
        request=Request(f'http://127.0.0.1:{http.server_port}/api/v1/messages',data=data,
            headers={'Content-Type':'application/json','Cookie':COOKIE+'='+token},method='POST')
        assert json.loads(urlopen(request,timeout=5).read())['accepted']
    try:
        for index,text in enumerate(['请在这个隔离假机器人测试里抓取桌上的米老鼠，完整调用准备与15秒 VLA，再观察回放结果。不要停在准备完成。',
                                     '请只检查当前状态，告诉我刚才是否是离线验证，不要再次抓取。'],1):
            send(text,'smoke-'+str(index));deadline=time.monotonic()+250
            while time.monotonic()<deadline:
                worker.tick()
                if len(list((run/'ui-agent-turns').glob('*/result.json')))>=index:break
                time.sleep(.25)
            else:raise RuntimeError('Smoke deadline')
            print('turn finished',index,flush=True)
        calls=backend.calls;run_calls=[args for name,args in calls if name=='g1_vla' and args['action']=='run']
        assert len(run_calls)==1 and run_calls[0]['steps']==750,calls
        assert any(name=='g1_vla' and args['action']=='prepare' for name,args in calls)
        assert len([x for x in calls if x[0]=='g1_observe'])>=2
        assert not worker.box.snapshot()['pending']
        report=dict(passed=True,real_codex=True,physical_robot_connected=False,robot_commands_sent=0,
            http_submissions=2,automatic_agent_turns=2,vla_run_steps=[x['steps'] for x in run_calls],calls=calls,
            messages=worker.box.snapshot()['messages'])
        atomic(run/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('calls','messages')}))
    finally:
        worker.close();http.shutdown();http.server_close();stop.set();thread.join(3)
        listener.close();front.close();Path(address).unlink(missing_ok=True)


if __name__=='__main__':main()
