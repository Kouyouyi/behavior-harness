"""UI wakeup lifecycle and restricted real MCP transport, with no robot I/O."""
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen

import pytest

from harness.codex_cli import ui_agent
from harness.codex_cli.ui_bridge import check, serve, ALLOWED
from services.g1_sonic.messages import Inbox
from services.g1_sonic.vla import atomic
from services.console.runtime import Console
from services.console.server import make_server
from services.controller.config import Configuration
from services.controller.frontend import Frontend
from services.controller.session import Session
from services.g1_sonic.backend import G1SonicBackend


def config(tmp_path):
    run=tmp_path/'run';run.mkdir()
    atomic(run/'ankle_status.json',dict(session_id='fixture',phase='ready'))
    c=Configuration(root=run/'session',task='ui_test',robot='g1_sonic',agent_provider='codex',backend_options={'control_directory':str(run)})
    path=run/'operator.json';path.write_text(json.dumps(c.to_dict()))
    return run,c,path


def test_claim_retry_crash_and_session_change(tmp_path):
    run,c,p=config(tmp_path);box=Inbox(run)
    box.send('抓取桌上米老鼠','id');assert box.claim('batch')==[dict(id='id',text='抓取桌上米老鼠',session='fixture')]
    box.send('抓取桌上米老鼠','id');assert not box.claim('second')
    Inbox(run).recover_dispatches();assert not box.snapshot()['pending']
    assert box.snapshot()['messages'][0]['status']=='failed'
    assert not box.claim('third')
    box.send('向左移动','new');atomic(run/'ankle_status.json',dict(session_id='different'))
    assert not box.claim('changed')
    assert box.snapshot()['messages'][-1]['status']=='expired'


def test_dispatcher_heartbeat_is_not_agent_receipt(tmp_path):
    run,c,p=config(tmp_path);box=Inbox(run);console=Console(mode='readonly',message_directory=run)
    box.send('检查状态','one');atomic(run/'ui-agent-status.json',dict(phase='idle',updated_at=time.time()))
    s=console.snapshot();assert s['capabilities']['agent_autowake']
    assert not s['capabilities']['agent_connected']
    assert box.snapshot()['messages'][0]['status']=='queued'
    atomic(run/'ui-agent-status.json',dict(phase='idle',updated_at=time.time()-6))
    assert not console.snapshot()['capabilities']['agent_autowake']


def test_proxy_blocks_changed_session_inactive_worker_and_long_vla(tmp_path):
    run,c,p=config(tmp_path)
    atomic(run/'ui-agent-status.json',dict(phase='running',batch='batch',updated_at=time.time()))
    check(run,'fixture','batch','g1_vla',dict(action='run',steps=750))
    with pytest.raises(ValueError,match='750'):check(run,'fixture','batch','g1_vla',dict(action='run',steps=751))
    with pytest.raises(ValueError,match='session changed'):check(run,'old','batch','g1_hand',{})
    with pytest.raises(ValueError,match='not enabled'):check(run,'fixture','batch','exec_command',{})
    atomic(run/'ui-agent-status.json',dict(phase='running',batch='batch',updated_at=time.time()-6))
    with pytest.raises(ValueError,match='no longer active'):check(run,'fixture','batch','g1_move',{})


def test_exec_command_uses_shared_proxy_and_no_privilege_bypass(tmp_path):
    run,c,p=config(tmp_path)
    cmd=ui_agent.command(c,p,sys.executable,'/codex','/socket',run,'fixture','batch')
    assert 'exec' in cmd and '--json' in cmd and '--ephemeral' in cmd
    assert '--dangerously-bypass-approvals-and-sandbox' not in cmd
    assert any('harness.codex_cli.ui_bridge' in x for x in cmd)
    assert not any('services.controller.frontend' in x for x in cmd)
    assert 'read-only' in cmd and cmd[-1]=='-'
    assert any('g1_task_step' in value for value in cmd if 'enabled_tools=' in value)
    assert any('g1_replay_route' in value for value in cmd if 'enabled_tools=' in value)
    assert 'mcp_servers.rax_robot.tool_timeout_sec=210' in cmd


def test_transient_reconnect_does_not_poison_successful_turn(tmp_path):
    run,c,p=config(tmp_path)
    worker=ui_agent.Worker(c,p,'/unused','/unused')
    worker.batch='fixture'
    path=tmp_path/'events.jsonl'
    path.write_text(json.dumps(dict(type='error',message='Reconnecting... 2/5 (routing timed out)'))+'\n')
    worker.output=path.open('r+')
    try:
        worker.drain()
        assert not worker.failed
        assert '正在恢复' in (run/'agent-events.jsonl').read_text()
    finally:
        worker.output.close()
        worker.reader.close()


class FixtureBackend(G1SonicBackend):
    """Only message read/ack and a recorded fake VLA command; cannot touch hardware."""
    def __init__(self,c):super().__init__(c);self.calls=[]
    def status(self):return dict(fixture=True)
    def _call(self,name,args):
        self.calls.append((name,args))
        if name=='g1_observe':return dict(frame_id='a'*32,fixture=True),[]
        if name=='g1_vla':return dict(job_id='b'*32,status='submitted',fixture=True),[]
        if name=='g1_vla_result':return dict(completed=True,task_success_verified=False,fixture=True),[]
        raise ValueError('Fixture tool not implemented: '+name)


FAKE_CODEX='''import json,sys,subprocess,time
args=sys.argv[1:];prompt=sys.stdin.read()
settings=[args[i+1] for i,x in enumerate(args) if x=='-c']
bridge=json.loads(next(x.split('=',1)[1] for x in settings if x.startswith('mcp_servers.rax_robot.args=')))
p=subprocess.Popen([sys.executable,*bridge],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
def request(i,method,params):
 p.stdin.write(json.dumps(dict(jsonrpc='2.0',id=i,method=method,params=params))+'\\n');p.stdin.flush()
 return json.loads(p.stdout.readline())['result']
def call(i,name,**args):
 r=request(i,'tools/call',dict(name=name,arguments=args));assert not r.get('isError'),r
 return json.loads(r['content'][0]['text'])
request(1,'initialize',{})
messages=call(2,'g1_operator_messages',action='read')['pending']
for i,m in enumerate(messages):call(10+i,'g1_operator_messages',action='ack',message_id=m['id'],reply='离线测试：观察后调用假 VLA')
call(100,'g1_observe')
job=call(101,'g1_vla',action='run',task='mickey',steps=750,evidence='Offline fixture only')
call(102,'g1_vla_result',job_id=job['job_id'])
print(json.dumps(dict(type='item.completed',item=dict(type='agent_message',text='离线工具链完成，没有机器人动作。'))),flush=True)
print(json.dumps(dict(type='item.completed',item=dict(type='reasoning',text='PRIVATE_REASONING_MUST_NOT_APPEAR'))),flush=True)
p.stdin.close();p.wait(timeout=3)
'''


@pytest.fixture
def services(tmp_path):
    run,c,path=config(tmp_path);backend=FixtureBackend(c);front=Frontend(Session(c),backend)
    # Keep the AF_UNIX pathname below the kernel limit.
    address=str(tmp_path/'mcp.sock');listener=socket.socket(socket.AF_UNIX);listener.bind(address);listener.listen(8)
    stop=threading.Event()
    thread=threading.Thread(target=front.serve_socket,args=(listener,stop),daemon=True);thread.start()
    console=Console(mode='readonly',message_directory=run,agent_events_file=run/'agent-events.jsonl')
    server,token=make_server(port=0,console=console,token='test-only')
    http=threading.Thread(target=server.serve_forever,daemon=True);http.start()
    fake=tmp_path/'fake_codex.py';fake.write_text('#!'+sys.executable+'\n'+FAKE_CODEX);fake.chmod(0o700)
    yield run,c,path,address,backend,server,console,fake
    server.shutdown();server.server_close()
    stop.set();thread.join(timeout=3);listener.close();front.close()


def wait_worker(worker, predicate):
    until=time.monotonic()+12
    while time.monotonic()<until:
        worker.tick()
        if predicate():return
        time.sleep(.03)
    raise AssertionError('worker did not finish')


def test_http_wakes_tool_chain_twice_without_manual_codex_prompt(services):
    run,c,path,address,backend,server,console,fake=services
    worker=ui_agent.Worker(c,path,address,str(fake));worker.start()
    def post(ident):
        body=json.dumps(dict(text='抓取桌上的米老鼠',request_id=ident)).encode()
        req=Request(f'http://127.0.0.1:{server.server_port}/api/v1/messages',data=body,
                    headers={'Content-Type':'application/json','Cookie':'operator_token=test-only'},method='POST')
        # Authentication cookie name comes from the actual HTTP boundary.
        from services.dashboard.server import COOKIE
        req.add_header('Cookie',COOKIE+'=test-only')
        return json.loads(urlopen(req,timeout=3).read())
    try:
        assert post('first')['accepted'];post('first')
        wait_worker(worker,lambda:len(list((run/'ui-agent-turns').glob('*/result.json')))==1)
        assert len([x for x in backend.calls if x[0]=='g1_vla'])==1
        assert console.snapshot()['conversation']['output']['text']=='离线工具链完成，没有机器人动作。'
        assert 'PRIVATE_REASONING' not in (run/'agent-events.jsonl').read_text()
        assert not Inbox(run).snapshot()['pending']
        post('second')
        wait_worker(worker,lambda:len(list((run/'ui-agent-turns').glob('*/result.json')))==2)
        assert len([x for x in backend.calls if x[0]=='g1_vla'])==2
        assert not (run/'vla_requests').exists() and not (run/'move_requests').exists()
    finally:worker.close()


def test_two_workers_cannot_launch_parallel_agents(tmp_path):
    run,c,p=config(tmp_path);a=ui_agent.Worker(c,p,'unused','unused');b=ui_agent.Worker(c,p,'unused','unused')
    a.start()
    try:
        with pytest.raises(BlockingIOError):b.start()
    finally:
        a.close()
        if b.lock:b.lock.close()


def test_start_failure_reported_once_not_retried(services):
    run,c,path,address,backend,server,console,fake=services
    worker=ui_agent.Worker(c,path,address,'/missing/codex');worker.start();box=Inbox(run)
    try:
        box.send('检查状态','one');worker.tick();worker.tick()
        assert box.snapshot()['messages'][0]['status']=='failed'
        assert len(list((run/'ui-agent-turns').glob('*/result.json')))==1
        assert worker.process is None and not backend.calls
    finally:worker.close()


def test_owner_change_between_claim_and_launch_cannot_replay(services):
    run,c,path,address,backend,server,console,fake=services
    worker=ui_agent.Worker(c,path,address,str(fake));worker.start();box=Inbox(run)
    try:
        box.send('松开右手','old');worker.batch='oldbatch';messages=box.claim(worker.batch)
        atomic(run/'ankle_status.json',dict(session_id='new-owner'))
        with pytest.raises(RuntimeError,match='Owner changed'):worker.launch(messages)
        assert worker.process is None and not backend.calls
    finally:worker.close()


def test_queued_intervention_does_not_start_second_agent_while_running(services):
    run,c,path,address,backend,server,console,fake=services
    # Use a slow child before reading messages to ensure the intervention arrives mid-turn.
    fake.write_text('#!'+sys.executable+'\nimport time;time.sleep(.4)\n'+FAKE_CODEX)
    worker=ui_agent.Worker(c,path,address,str(fake));worker.start();box=Inbox(run)
    try:
        box.send('抓取','first');worker.tick();pid=worker.process.pid
        box.send('请检查最新目标位置','intervention');worker.tick()
        assert worker.process.pid==pid
        wait_worker(worker,lambda:len(list((run/'ui-agent-turns').glob('*/result.json')))==1)
        worker.tick();assert worker.process is None
        assert not box.snapshot()['pending']
        assert len(list((run/'ui-agent-turns').iterdir()))==1
    finally:worker.close()


def test_shutdown_cancels_only_own_agent_and_never_creates_native_stop(services):
    run,c,path,address,backend,server,console,fake=services
    fake.write_text('#!'+sys.executable+'\nimport sys,time\nsys.stdin.read();time.sleep(30)\n')
    worker=ui_agent.Worker(c,path,address,str(fake));worker.start();box=Inbox(run)
    box.send('等待测试','one');worker.tick();process=worker.process
    worker.close()
    assert process.poll() is not None and not box.dispatcher()['online']
    assert box.snapshot()['messages'][0]['status']=='failed'
    assert not (run/'STOP_O').exists() and not (run/'native-exit.json').exists()
    assert not backend.calls


def test_new_instruction_receipt_does_not_hide_later_agent_response(tmp_path):
    run,c,p=config(tmp_path);box=Inbox(run);now=time.time();events=run/'agent-events.jsonl'
    rows=[dict(id='task',task_id='turn',kind='task',time=now-2,status='running',details=dict(instruction='抓取')),
          dict(id='user',task_id='turn',kind='message',time=now-2,role='user',text='抓取')]
    box.send('暂停抓取','new');box.snapshot(read=True);box.acknowledge('new','收到，准备暂停')
    rows.append(dict(id='final',task_id='turn',kind='message',time=now+1,role='assistant',text='已暂停并确认队列清空；底层控制保持。'))
    events.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    console=Console(mode='readonly',message_directory=run,agent_events_file=events)
    view=console.snapshot()['conversation']
    assert view['user']['text']=='暂停抓取'
    assert view['output']['text']=='已暂停并确认队列清空；底层控制保持。'
