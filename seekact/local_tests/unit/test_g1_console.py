"""Console contracts and isolation. No model/SDK/hardware services are started."""
import json
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from services.console.adapters import BOOT_ID, Camera, JOINTS, agent_events, telemetry
from services.console.runtime import Console
from services.console.server import make_server
from services.console.telemetry import sample_message
from services.console.thermal import ThermalDisplay


class Clock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_demo_pause_resume_and_evidence(tmp_path):
    clock = Clock()
    console = Console(clock=clock, storage=tmp_path)
    task = console.submit('把桌上的米老鼠放到沙发上', 'one')
    clock.advance(9)
    assert console.snapshot()['task']['stage_index'] == 1
    paused = console.action(task['id'], 'pause')
    clock.advance(100)
    assert console.snapshot()['task']['elapsed_s'] == paused['elapsed_s']
    console.action(task['id'], 'resume')
    clock.advance(33)
    value = console.snapshot()
    assert value['task']['status'] == 'completed'
    assert value['task']['elapsed_s'] == 42
    assert value['task']['progress'] == 100
    assert value['task']['simulated_verified'] is True
    assert value['task']['verified'] is False
    assert value['capabilities']['physical_motion'] is False
    assert value['camera']['source'] == 'recorded_sample'
    assert len([e for e in value['events'] if e['kind'] == 'tool']) == 6
    restored = Console(storage=tmp_path)
    assert restored.snapshot()['task'] is None
    assert restored.snapshot()['history'][-1]['status'] == 'completed'


def test_submission_deduplicates_concurrent_requests():
    console = Console()
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = list(pool.map(lambda _: console.submit('把米老鼠放到沙发上', 'same'), range(6)))
    assert len({t['id'] for t in tasks}) == 1
    assert len([e for e in console.events if e['kind'] == 'plan']) == 1
    with pytest.raises(RuntimeError, match='request_id'):
        console.submit('米老鼠搬到沙发', 'same')
    with pytest.raises(RuntimeError, match='已有任务'):
        console.submit('米老鼠到沙发', 'new')


def test_cancel_is_terminal_and_repeatable():
    clock = Clock()
    console = Console(clock=clock)
    task = console.submit('米老鼠到沙发', 'one')
    console.action(task['id'], 'cancel')
    console.action(task['id'], 'cancel')
    clock.advance(100)
    assert console.snapshot()['task']['status'] == 'cancelled'
    assert len(console.tasks) == 1
    with pytest.raises(RuntimeError):
        console.action(task['id'], 'resume')
    assert console.submit('米老鼠到沙发', 'two')['id'] != task['id']


def test_readonly_never_accepts_tasks_or_chat(tmp_path):
    console = Console(mode='readonly', storage=tmp_path)
    with pytest.raises(RuntimeError, match='只提供观测'):
        console.submit('米老鼠到沙发', 'one')
    with pytest.raises(RuntimeError):
        console.action('unknown', 'pause')
    with pytest.raises(RuntimeError):
        console.message('开始')
    value = console.snapshot()
    assert not any(value['capabilities'].values())
    assert value['robot']['status'] == 'offline'
    assert value['camera']['url'] is None
    assert all(j['winding_c'] is None for j in value['telemetry']['joints'])
    assert not (tmp_path / 'tasks.jsonl').exists()


@pytest.mark.parametrize('instruction', ['', '打开窗户', 'x' * 1001, None, {'text': '米老鼠到沙发'}])
def test_unsupported_or_invalid_task_is_not_success(instruction):
    console = Console()
    with pytest.raises(ValueError):
        console.submit(instruction, 'one')
    assert console.task is None


def test_shutdown_records_interruption_and_no_replay(tmp_path):
    console = Console(storage=tmp_path)
    console.submit('米老鼠到沙发', 'one')
    console.close()
    restored = Console(storage=tmp_path)
    assert restored.tasks[-1]['status'] == 'interrupted'
    assert restored.task is None


def test_event_cursor_and_export():
    console = Console()
    cursor = console.event_page()['cursor']
    console.submit('米老鼠到沙发', 'one')
    assert all(e['seq'] > cursor for e in console.event_page(cursor)['events'])
    value = console.export()
    assert '模拟' in value['export_note']
    assert value['schema_version'] == 1
    json.dumps(value, allow_nan=False)


def make_sample(now=1000):
    return dict(source='live_lowstate', sampled_at=now, host_boot_id=BOOT_ID,
                motors=[dict(hardware_index=j['hardware_index'], temperature_case_winding_c=[35, 48],
                             motorstate=0) for j in JOINTS])


def test_telemetry_tuple_order_stale_and_boot_identity(tmp_path):
    path = tmp_path / 'telemetry.json'
    sample = make_sample()
    path.write_text(json.dumps(sample))
    fresh = telemetry(path, now=1001)
    assert fresh['fresh'] and len(fresh['joints']) == 29
    assert fresh['joints'][0]['case_c'] == 35 and fresh['joints'][0]['winding_c'] == 48
    assert not telemetry(path, now=1003)['fresh']
    assert not telemetry(path, now=999)['fresh']
    sample['host_boot_id'] = 'another-boot'
    path.write_text(json.dumps(sample))
    assert not telemetry(path, now=1001)['fresh']


def test_legacy_temperature_name_not_mislabelled(tmp_path):
    path = tmp_path / 'temperatures.jsonl'
    sample = {'monotonic': 1000, 'motors': [{'hardware_index': 0, 'temperature_winding_driver_c': [60, 130], 'motorstate': 512}]}
    path.write_text(json.dumps({'sample': sample}) + '\n')
    value = telemetry(path, now=1001)
    assert value['joints'][0]['case_c'] == 60 and value['joints'][0]['winding_c'] == 130
    assert not value['fresh']  # Legacy samples have no boot identity/wall timestamp.
    assert value['fault_count'] == 1


def test_invalid_and_partial_telemetry(tmp_path):
    path = tmp_path / 'telemetry.json'
    path.write_text('{broken')
    assert not telemetry(path)['fresh']
    sample = make_sample()
    sample['motors'][0]['temperature_case_winding_c'] = [None, 20]
    path.write_text(json.dumps(sample))
    assert not telemetry(path, now=1000)['fresh']
    assert telemetry(path, now=1000)['joints'][0]['winding_c'] is None
    sample['motors'] = None
    path.write_text(json.dumps(sample))
    assert not telemetry(path, now=1000)['fresh']


def test_collector_only_samples_and_labels():
    msg = SimpleNamespace(motor_state=[SimpleNamespace(temperature=[30, 50], motorstate=0, tau_est=1.5) for _ in range(29)])
    sample = sample_message(msg)
    assert sample['motors'][0]['temperature_case_winding_c'] == [30, 50]
    assert sample['host_boot_id'] == BOOT_ID and len(sample['motors']) == 29


def test_camera_freshness_and_path_boundary(tmp_path):
    import time
    camera = Camera(tmp_path)
    assert camera.snapshot()['url'] is None
    frame_id = 'a' * 32
    (tmp_path / frame_id).mkdir()
    (tmp_path / frame_id / 'rgb.png').write_bytes(b'png')
    meta = dict(frame_id=frame_id, source='live_camera', host_boot_id=BOOT_ID,
                received_monotonic=time.monotonic(), published_unix=time.time())
    (tmp_path / 'latest.json').write_text(json.dumps(meta))
    assert camera.snapshot()['fresh']
    assert camera.frame(frame_id) == b'png'
    meta['received_monotonic'] -= 5
    (tmp_path / 'latest.json').write_text(json.dumps(meta))
    assert not camera.snapshot()['fresh']
    with pytest.raises(KeyError):
        camera.frame('../secret')
    (tmp_path / frame_id / 'rgb.png').unlink()
    (tmp_path / frame_id / 'rgb.png').symlink_to('/etc/passwd')
    with pytest.raises(ValueError):
        camera.frame(frame_id)


def test_app_server_adapter_excludes_private_reasoning(tmp_path):
    path = tmp_path / 'agent.jsonl'
    rows = [dict(method='item/completed', params={'item': {'type': 'reasoning', 'content': 'PRIVATE'}}),
            dict(method='item/completed', params={'item': {'id': 'one', 'type': 'mcpToolCall', 'tool': 'g1_observe', 'status': 'completed', 'result': {'frame_id': 'frame'}}}),
            dict(method='item/completed', params={'item': {'id': 'two', 'type': 'agentMessage', 'text': '已观察目标'}})]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    events = agent_events(path)
    assert len(events) == 2 and events[0]['tool'] == 'g1_observe'
    assert 'PRIVATE' not in json.dumps(events)


def test_conversation_progresses_and_freezes_with_task():
    clock = Clock()
    console = Console(clock=clock)
    task = console.submit('把米老鼠放到沙发上', 'conversation')
    assert console.snapshot()['conversation']['phase'] == 'thinking'
    assert console.snapshot()['conversation']['tool'] is None
    clock.advance(2)
    assert console.snapshot()['conversation']['phase'] == 'tool'
    clock.advance(1.5)
    output = console.snapshot()['conversation']['output']
    assert output['streaming'] and output['text']
    clock.advance(.3)
    later = console.snapshot()['conversation']['output']
    assert later['id'] == output['id'] and later['text'].startswith(output['text'])
    assert len(later['text']) > len(output['text'])
    console.action(task['id'], 'pause')
    paused = console.snapshot()['conversation']
    clock.advance(30)
    assert console.snapshot()['conversation'] == paused
    assert paused['phase'] == 'paused' and not paused['output']['streaming']
    console.action(task['id'], 'resume')
    assert console.snapshot()['conversation']['output']['text'] == later['text']
    clock.advance(42)
    complete = console.snapshot()['conversation']
    assert complete['phase'] == 'completed' and not complete['output']['streaming']


@pytest.mark.parametrize('autoplay', [False, True])
def test_chat_reply_stays_visible_and_manual_input_stops_autoplay(autoplay):
    clock = Clock()
    console = Console(clock=clock, demo_autoplay=autoplay)
    console.tick()
    result = console.message('检查一下窗户')
    assert result['accepted'] and result['source'] == 'demo'
    assert not console.demo_autoplay
    view = console.snapshot()['conversation']
    assert view['user']['text'] == '检查一下窗户'
    assert view['phase'] == 'replied' and view['output']['text'] == result['reply']
    assert view['tool'] is None and not view['output']['streaming']
    clock.advance(60)
    console.tick()
    assert console.snapshot()['conversation']['output'] == view['output']
    console.message('好的，请等待下一条指令')
    view = console.snapshot()['conversation']
    assert view['user']['text'] == '好的，请等待下一条指令'
    assert [m['role'] for m in view['history']] == ['user', 'assistant']
    assert view['history'][0]['text'] == '检查一下窗户'
    assert console.snapshot()['capabilities']['physical_motion'] is False


def test_example_input_starts_interactive_demo_and_preserves_original_instruction():
    clock = Clock()
    console = Console(clock=clock, demo_autoplay=True)
    console.tick()
    old_id = console.task['id']
    instruction = '把桌上的米老师玩偶放到沙发上'
    result = console.message(instruction)
    assert result['action'] == 'demo_task_started'
    assert result['task']['id'] != old_id
    assert console.tasks[-1]['status'] == 'cancelled'
    value = console.snapshot()
    assert value['task']['instruction'] == instruction
    assert value['conversation']['user']['text'] == instruction
    assert '米老师' in value['conversation']['plan_note']
    assert len(value['conversation']['plan']) == 6
    assert not value['presentation']['autoplay']
    clock.advance(42)
    value = console.snapshot()
    assert value['task']['simulated_verified'] and not value['task']['verified']
    assert all(step['status'] == 'completed' for step in value['conversation']['plan'])
    assert len([e for e in value['events'] if e['kind'] == 'tool_result']) == 6


def test_demo_tool_feedback_is_once_only_and_precedes_next_step():
    clock = Clock()
    console = Console(clock=clock)
    console.message('把桌上的米老鼠玩偶放到沙发上')
    clock.advance(2)
    assert console.snapshot()['conversation']['tool']['status'] == 'running'
    clock.advance(1.5)
    value = console.snapshot()
    assert value['task']['stage_index'] == 0
    assert value['conversation']['tool']['status'] == 'completed'
    assert value['conversation']['tool']['result']['simulated'] is True
    assert value['conversation']['plan'][0]['status'] == 'completed'
    console.snapshot()
    assert len([e for e in console.events if e['kind'] == 'tool_result']) == 1
    clock.advance(4)
    value = console.snapshot()
    assert value['task']['stage_index'] == 1
    assert value['conversation']['plan'][1]['status'] == 'running'
    assert len([e for e in console.events if e['kind'] == 'tool_result']) == 1


def test_readonly_plan_uses_agent_steps_instead_of_demo_fixture(tmp_path):
    path = tmp_path / 'plan.jsonl'
    rows = [dict(kind='task', task_id='window', status='running', details={'instruction': '检查窗户'}),
            dict(kind='plan', task_id='window', text='按本次指令检查', details={'steps': [
                {'id': 'look', 'label': '查看窗户', 'status': 'completed'},
                {'id': 'report', 'label': '报告结果', 'status': 'running'}]})]
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    value = Console(mode='readonly', agent_events_file=path).snapshot()['conversation']
    assert [s['label'] for s in value['plan']] == ['查看窗户', '报告结果']
    assert value['plan'][1]['status'] == 'running'


def test_arbitrary_task_and_cumulative_agent_output(tmp_path):
    clock = Clock()
    path = tmp_path / 'agent.jsonl'
    console = Console(mode='readonly', clock=clock, agent_events_file=path)
    rows = [dict(id='task-update', task_id='inspection-1', kind='task', time=clock(), status='running',
                 details=dict(instruction='检查窗户是否关闭，然后向我报告', current_step='查看窗户状态', elapsed_s=12)),
            dict(id='tool-1', task_id='inspection-1', kind='tool', tool='inspect_window', status='running'),
            dict(id='answer-1', task_id='inspection-1', kind='message', role='assistant', time=clock(),
                 status='streaming', text='我正在检查')]
    def publish():
        path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    publish()
    value = console.snapshot()['conversation']
    assert value['task']['instruction'] == '检查窗户是否关闭，然后向我报告'
    assert value['task']['current_step'] == '查看窗户状态'
    assert value['task']['elapsed_s'] == 12
    assert 'progress' not in value['task'] and 'stage_index' not in value['task']
    assert value['tool']['tool'] == 'inspect_window'
    assert value['output']['streaming'] and value['output']['id'] == 'answer-1'
    rows.append(dict(rows[-1], text='我正在检查窗户，稍后报告结果。', status='completed'))
    publish()
    updated = console.snapshot()['conversation']
    assert updated['output']['text'].startswith(value['output']['text'])
    assert updated['output']['id'] == value['output']['id']
    assert not updated['output']['streaming']
    assert console.snapshot()['task'] is None  # Observation does not create a demo task.
    # A streaming turn can exceed the bounded event tail without losing its task.
    rows.extend(dict(rows[-1], text=f'持续输出 {i}') for i in range(125))
    publish()
    assert console.snapshot()['conversation']['task']['instruction'] == '检查窗户是否关闭，然后向我报告'
    rows.append(dict(id='new-task', task_id='new-task', kind='task', status='waiting',
                     details=dict(instruction='请向我挥手')))
    publish()
    updated = console.snapshot()['conversation']
    assert updated['task']['instruction'] == '请向我挥手'
    assert updated['tool'] is None and updated['task']['elapsed_s'] is None
    assert updated['output']['text'] == '等待 Agent 的公开输出。'


def test_agent_stream_indicator_expires_and_malformed_task_is_safe(tmp_path):
    clock = Clock()
    path = tmp_path / 'agent.jsonl'
    path.write_text(json.dumps(dict(kind='task', details='invalid', status='nonsense')) + '\n' +
                    json.dumps(dict(kind='message', text='公开输出', status='streaming', time=clock(),
                                    reasoning='PRIVATE')) + '\n')
    console = Console(mode='readonly', clock=clock, agent_events_file=path)
    value = console.snapshot()
    assert value['conversation']['task']['status'] == 'unknown'
    assert 'PRIVATE' not in json.dumps(value)
    path.write_text(json.dumps(dict(kind='message', id='one', text='公开输出', status='streaming', time=clock())) + '\n')
    assert console.snapshot()['conversation']['output']['streaming']
    clock.advance(6)
    assert not console.snapshot()['conversation']['output']['streaming']


@pytest.fixture
def http_console(tmp_path):
    console = Console(storage=tmp_path)
    server, token = make_server(port=0, console=console, token='console-test')
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f'http://127.0.0.1:{server.server_port}', token, console
    server.shutdown()
    server.server_close()
    console.close()


def call_http(base, path, data=None, token='console-test', content_type='application/json'):
    request = Request(base + path, data=json.dumps(data).encode() if data is not None else None,
                      headers={'X-Operator-Token': token, 'Content-Type': content_type})
    try:
        with urlopen(request, timeout=3) as response:
            return response.status, response.read(), response.headers
    except HTTPError as error:
        return error.code, error.read(), error.headers


def test_http_auth_assets_and_csp(http_console):
    base, _, _ = http_console
    assert call_http(base, '/api/v1/state', token='')[0] == 401
    assert call_http(base, '/assets/observation.png', token='')[0] == 401
    for path in ('/', '/assets/console.js', '/assets/composer.js', '/assets/console.css', '/assets/observation.png'):
        status, body, headers = call_http(base, path)
        assert status == 200 and body
        assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
    assert call_http(base, '/api/v1/events?after=-1')[0] == 400
    assert call_http(base, '/api/v1/camera/frame?frame_id=../../etc/passwd')[0] == 404
    status, body, _ = call_http(base, '/api/v1/schema')
    assert status == 200 and json.loads(body)['openapi'] == '3.1.0'


def test_http_task_lifecycle_and_export(http_console):
    base, _, _ = http_console
    data = {'instruction': '把米老鼠放到沙发上', 'request_id': 'http'}
    assert call_http(base, '/api/v1/tasks', data, content_type='text/plain')[0] == 400
    status, body, _ = call_http(base, '/api/v1/tasks', data)
    assert status == 201
    task_id = json.loads(body)['task']['id']
    path = f'/api/v1/tasks/{task_id}/actions'
    assert json.loads(call_http(base, path, {'action': 'pause'})[1])['task']['status'] == 'paused'
    assert json.loads(call_http(base, path, {'action': 'resume'})[1])['task']['status'] == 'running'
    assert json.loads(call_http(base, path, {'action': 'cancel'})[1])['task']['status'] == 'cancelled'
    assert len(json.loads(call_http(base, '/api/v1/history')[1])['tasks']) == 1
    status, body, headers = call_http(base, '/api/v1/export')
    assert status == 200 and 'attachment' in headers['Content-Disposition']
    assert json.loads(body)['mode'] == 'demo'


def test_http_readonly_motion_is_refused(http_console):
    base, _, console = http_console
    console.mode = 'readonly'
    status, _, _ = call_http(base, '/api/v1/tasks', {'instruction': '米老鼠到沙发', 'request_id': 'x'})
    assert status == 409 and console.task is None


def test_http_live_inbox_authenticated_retry_and_receipt(http_console,tmp_path):
    from services.g1_sonic.messages import Inbox
    base, _, console = http_console
    console.mode='readonly';console.inbox=Inbox(tmp_path/'inbox')
    payload={'text':'消息通路测试，不执行动作','request_id':'http-retry'}
    assert call_http(base,'/api/v1/messages',payload,token='wrong')[0]==401
    assert not console.inbox.snapshot()['messages']
    assert call_http(base,'/api/v1/messages',{'text':'missing id'})[0]==400
    for _ in range(2):
        status,body,_=call_http(base,'/api/v1/messages',payload)
        assert status==200 and json.loads(body)['source']=='operator_inbox'
    state=json.loads(call_http(base,'/api/v1/state')[1])
    assert state['capabilities']['chat'] and not state['capabilities']['physical_motion']
    assert len(state['operator_inbox']['pending'])==1
    assert state['operator_inbox']['pending'][0]['text']==payload['text']
    assert not (tmp_path/'inbox/vla_requests').exists()


def test_network_exposure_is_not_silently_enabled():
    with pytest.raises(ValueError, match='loopback'):
        make_server('0.0.0.0')


def display_sample(case=35, winding=45, *, index=4, fresh=True, fault=0):
    return {'joints': [{'id': index, 'name': 'left_ankle_pitch_joint' if index == 4 else 'left_elbow_joint',
                        'case_c': case, 'winding_c': winding, 'fresh': fresh, 'motorstate': fault}]}


@pytest.mark.parametrize('case,winding,expected', [(84.9, 40, 'normal'), (85, 40, 'warning'),
    (40, 89.9, 'warning'), (90, 40, 'critical'), (40, 90, 'critical'), (40, 130, 'critical')])
def test_wallboard_native_thresholds_use_both_sensors(case, winding, expected):
    value = ThermalDisplay().apply(display_sample(case, winding), 'native90')
    assert value['joints'][0]['level'] == expected
    assert value['joints'][0]['max_c'] == max(case, winding)


def test_wallboard_ankle_hysteresis_and_stale_samples():
    display = ThermalDisplay()
    assert display.apply(display_sample(winding=90), 'native90')['joints'][0]['level'] == 'critical'
    retained = display.apply(display_sample(winding=87), 'native90')['joints'][0]
    assert retained['level'] == 'critical' and retained['reason'] == 'alarm_latched'
    assert display.apply(display_sample(winding=30, fresh=False), 'native90')['joints'][0]['level'] == 'unknown'
    assert display.apply(display_sample(winding=85), 'native90')['joints'][0]['level'] == 'critical'
    assert display.apply(display_sample(winding=84.9), 'native90')['joints'][0]['level'] == 'normal'
    assert display.apply(display_sample(winding=87), 'native90')['joints'][0]['level'] == 'warning'


def test_wallboard_non_ankle_matches_terminal_colors_without_latching():
    display = ThermalDisplay()
    assert display.apply(display_sample(winding=90, index=18), 'native90')['joints'][0]['level'] == 'critical'
    assert display.apply(display_sample(winding=87, index=18), 'native90')['joints'][0]['level'] == 'warning'


def test_wallboard_strict91_boundaries_and_rule_change():
    display = ThermalDisplay()
    display.apply(display_sample(winding=93), 'native90')
    for temperature, level in [(90, 'warning'), (91, 'warning'), (91.1, 'critical'), (84.9, 'normal')]:
        value = display.apply(display_sample(winding=temperature), 'strict91')
        assert value['joints'][0]['level'] == level
        assert not value['joints'][0]['alarm_latched']


def test_wallboard_fault_red_but_stale_never_looks_current():
    display = ThermalDisplay()
    value = display.apply(display_sample(fault=512), 'native90')['joints'][0]
    assert value['level'] == 'critical' and value['reason'] == 'motor_fault'
    value = display.apply(display_sample(fault=512, fresh=False), 'native90')['joints'][0]
    assert value['level'] == 'unknown' and value['reason'] == 'stale'


def test_wallboard_autoplay_and_no_live_autoplay():
    clock = Clock()
    console = Console(clock=clock, demo_autoplay=True)
    console.tick()
    task_id = console.task['id']
    clock.advance(42)
    console.tick()
    assert console.task['status'] == 'completed'
    clock.advance(7)
    console.tick()
    assert console.task['id'] == task_id
    clock.advance(1)
    console.tick()
    assert console.task['id'] != task_id
    console.action(console.task['id'], 'cancel')
    clock.advance(100)
    console.tick()
    assert console.task['status'] == 'cancelled'
    readonly = Console(mode='readonly', demo_autoplay=True)
    readonly.tick()
    assert readonly.task is None


def test_wallboard_owner_rule_and_all_29_classified(tmp_path):
    import time
    path = tmp_path / 'owner.json'
    path.write_text(json.dumps({'updated_monotonic': time.monotonic(), 'thermal_rule': 'strict91'}))
    value = Console(owner_file=path).snapshot()['telemetry']
    assert value['thermal_rule'] == 'strict91' and value['rule_source'] == 'owner'
    assert len(value['joints']) == 29 and sum(value['counts'].values()) == 29
    assert all(j['level'] in ('normal', 'warning', 'critical', 'unknown') for j in value['joints'])
