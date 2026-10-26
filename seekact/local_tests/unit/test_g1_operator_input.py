"""UI to real MCP inbox, including duplicate sends and action ordering; no hardware."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from services.console.runtime import Console
from services.g1_sonic.messages import Inbox
from services.g1_sonic.backend import G1SonicBackend
from services.controller.config import Configuration
from services.controller.frontend import Frontend
from services.controller.session import Session


def test_ui_to_mcp_and_receipt(tmp_path):
    console = Console(mode='readonly', message_directory=tmp_path)
    config = Configuration(root=tmp_path/'session', task='messages', robot='g1_sonic',
                           backend_options={'control_directory':str(tmp_path)})
    front = Frontend(Session(config), G1SonicBackend(config))
    def call(args):
        result = front.call('g1_operator_messages', args)
        assert not result.get('isError'), result
        return json.loads(result['content'][0]['text'])
    try:
        assert console.snapshot()['capabilities']['chat']
        assert not console.snapshot()['capabilities']['agent_connected']
        reply = console.message('松开右手', 'retry-safe-id')
        assert reply['accepted'] and reply['action']=='queued'
        assert '等待 Agent 读取' in console.snapshot()['conversation']['output']['text']
        messages = call({'action':'read'})['pending']
        assert messages[0]['text']=='松开右手'
        assert console.snapshot()['capabilities']['agent_connected']
        assert '已读取' in console.snapshot()['conversation']['output']['text']
        response = call({'action':'ack','message_id':messages[0]['id'],'reply':'已读取；先确认物体有支撑，尚未执行松手。'})
        assert not response['action_executed']
        assert not call({'action':'read'})['pending']
        assert console.snapshot()['conversation']['output']['text']==response['reply']
        assert not (tmp_path/'vla_requests').exists()
    finally:
        front.close()


def test_retry_deduplicates_and_survives_restart(tmp_path):
    box=Inbox(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: box.send('右手松开','same'), range(8)))
    assert len(Inbox(tmp_path).snapshot()['pending'])==1
    with pytest.raises(ValueError):box.send('左手松开','same')
    with pytest.raises(ValueError):box.acknowledge('same','Cannot acknowledge before read')


@pytest.mark.parametrize('change',['time','session'])
def test_stale_messages_are_visible_but_not_actionable(tmp_path,change):
    now=[1000.];box=Inbox(tmp_path,clock=lambda:now[0])
    (tmp_path/'ankle_status.json').write_text(json.dumps({'session_id':'one'}))
    box.send('松手','one')
    if change=='time':now[0]+=121
    else:(tmp_path/'ankle_status.json').write_text(json.dumps({'session_id':'two'}))
    value=box.snapshot(read=True)
    assert not value['pending'] and value['messages'][0]['status']=='expired'
    with pytest.raises(ValueError):box.acknowledge('one','Too late')


def test_pending_message_blocks_new_motion_but_not_pause(tmp_path,monkeypatch):
    config=Configuration(root=tmp_path/'session',task='messages',robot='g1_sonic',backend_options={'control_directory':str(tmp_path)})
    backend=G1SonicBackend(config);box=Inbox(tmp_path);box.send('暂停，先不要松手','id')
    called=[]
    monkeypatch.setattr(backend,'_call',lambda name,args:(called.append(name) or {},[]))
    try:
        with pytest.raises(RuntimeError,match='Pending UI'):
            backend.call('g1_hand',{'side':'right','action':'open','evidence':'visible object'})
        assert not called
        backend.call('g1_vla',{'action':'pause'})
        assert called==['g1_vla']
        backend.call('g1_move',{'action':'stop'})
        backend.call('g1_move',{'action':'status'})
        backend.call('g1_move',{'action':'result','job_id':'a'*32})
        assert called==['g1_vla','g1_move','g1_move','g1_move']
        with pytest.raises(RuntimeError,match='Pending UI'):
            backend.call('g1_move',dict(action='run',direction='backward',frame_id='b'*32,evidence='Rear path checked'))
        assert box.snapshot()['pending'][0]['status']=='received'
        box.acknowledge('id','保持暂停')
        assert not box.snapshot()['pending']
    finally:backend.close()
