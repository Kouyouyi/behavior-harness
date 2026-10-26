"""Thread-safe presentation runtime with a deterministic, explicitly simulated task.

No model invocation, robot command, arming or motion publisher exists in this module.
Live mode optionally queues operator messages for MCP agents, without executing them.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import json
import math
from pathlib import Path
import threading
import time
import uuid

from .adapters import Camera, JOINTS, agent_events, joint_group, read_json, tail_json, telemetry
from .thermal import ThermalDisplay
from .conversation import project
from .scenario import matches, RESULTS, RESULT_AFTER

STAGES = [
    ('observe', '识别目标', '确认桌面上的米老鼠', 'g1_observe', '模拟视觉反馈：已定位桌面玩偶，可进入接近阶段。'),
    ('approach', '走近桌面', '调整到可操作的位置', 'navigate_to_object', '演示移动阶段结束，进入抓取步骤。'),
    ('grasp', '抓取确认', '伸手、握持并检查反馈', 'grasp_object', '演示抓取反馈已返回，继续检查抬升状态。'),
    ('carry', '移动至沙发', '保持握持并检查路径', 'carry_to_location', '演示搬运阶段结束，准备放置。'),
    ('place', '放置物体', '放下、松手并撤离', 'place_object', '演示放置阶段结束，准备最终检查。'),
    ('verify', '结果验证', '确认米老鼠留在沙发上', 'verify_placement', '演示验收通过。实机结果仍需现场图像与反馈验证。'),
]
STEP_SECONDS = 7.0


class Console:
    def __init__(self, *, mode='demo', storage=None, camera_directory=None, telemetry_file=None,
                 owner_file=None, agent_events_file=None, clock=time.time,
                 thermal_rule='native90', demo_autoplay=False, message_directory=None):
        if mode not in ('demo', 'readonly'):
            raise ValueError('mode must be demo or readonly')
        from services.g1_sonic.messages import Inbox
        self.inbox = Inbox(message_directory) if message_directory and mode == "readonly" else None
        self.mode = mode
        if thermal_rule not in ('native90', 'strict91'):
            raise ValueError('invalid thermal rule')
        self.thermal_rule = thermal_rule
        self.thermal_display = ThermalDisplay()
        self.demo_autoplay = demo_autoplay and mode == 'demo'
        self.clock = clock
        self.camera = Camera(camera_directory)
        self.telemetry_file = telemetry_file
        self.owner_file = owner_file
        self.agent_events_file = agent_events_file
        self.storage = Path(storage) if storage else None
        self.lock = threading.RLock()
        self.events = []
        self.messages = []
        self._observed_task = None
        self.tasks = []
        self.task = None
        self.sequence = 0
        self.idempotency = OrderedDict()
        self.temperatures = []
        self.started_at = clock()
        self._stop = threading.Event()
        self._worker = None
        if self.storage:
            self.storage.mkdir(parents=True, exist_ok=True)
            self.events = tail_json(self.storage / 'events.jsonl', 200)
            self.sequence = max((e.get('seq', 0) for e in self.events), default=0)
            self.tasks = tail_json(self.storage / 'tasks.jsonl', 50)
        self._message('assistant', 'G1 展示已就绪。任务执行、观测反馈与关节状态将在大屏同步展示。'
                      if mode == 'demo' else '实机消息收件箱已启用。消息会交给执行中的 Agent 读取；空闲时等待处理，不会自动启动机器人。'
                      if self.inbox else '已进入实机观测模式。相机、遥测与 Agent 记录接入后会在这里更新；当前未连接任务执行服务。')
        self._event('system', '控制台已就绪', '演示模式 · 不连接机器人' if mode == 'demo' else '实机观测 · 不发送运动指令')

    def start(self):
        if self._worker:
            return
        def work():
            while not self._stop.wait(.25):
                with self.lock:
                    self.tick()
        self._worker = threading.Thread(target=work, name='g1-console-demo', daemon=True)
        self._worker.start()

    def tick(self):
        with self.lock:
            if self.demo_autoplay and (self.task is None or
                    (self.task['status'] == 'completed' and self.clock() - self.task['ended_at'] >= 8)):
                self.submit('把桌上的米老鼠放到沙发上', 'autoplay-' + uuid.uuid4().hex)
            self._advance()

    def close(self):
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=2)
        with self.lock:
            if self.task and self.task['status'] in ('running', 'paused'):
                self._finish('interrupted', '控制台退出，演示任务结束；不会在重启后重放。')

    def _append(self, name, value):
        if self.storage:
            with (self.storage / name).open('a') as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')

    def _event(self, kind, title, text='', **extra):
        self.sequence += 1
        event = dict(id=f'event-{self.sequence}', seq=self.sequence, time=self.clock(), kind=kind,
                     title=title, text=text, source=self.mode, task_id=self.task['id'] if self.task else None, **extra)
        self.events.append(event)
        self.events = self.events[-200:]
        self._append('events.jsonl', event)
        return event

    def _message(self, role, text, **extra):
        message = dict(id=uuid.uuid4().hex, role=role, text=text, time=self.clock(), **extra)
        self.messages.append(message)
        self.messages = self.messages[-80:]
        return message

    def _demo_only(self):
        if self.mode != 'demo':
            raise RuntimeError('实机任务执行服务尚未接入；当前只提供观测，不会提交机器人动作。')

    def _start_stage(self, index):
        stage = STAGES[index]
        arguments = {'object': 'mickey'}
        if index == 0:
            arguments['camera'] = 'head'
        elif index == 1:
            arguments['target'] = 'table'
        elif index in (3, 5):
            arguments['destination'] = 'sofa'
        elif index == 4:
            arguments['surface'] = 'sofa_cushion'
        self._event('tool', stage[1], stage[2], tool=stage[3], status='running',
                    details={'simulated': True, 'stage': stage[0], 'task_id': self.task['id'],
                             'arguments': arguments,
                             'implementation': 'existing_interface' if index == 0 else 'proposed_skill'})

    def submit(self, text, request_id):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
            raise ValueError('请输入 1–1000 字的任务')
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
            raise ValueError('request_id is required (1–100 characters)')
        text = text.strip()
        with self.lock:
            self._demo_only()
            if request_id in self.idempotency:
                old_text, old_task = self.idempotency[request_id]
                if old_text != text:
                    raise RuntimeError('同一 request_id 不能用于不同任务')
                return deepcopy(old_task)
            if self.task and self.task['status'] in ('running', 'paused'):
                raise RuntimeError('已有任务正在进行，请先完成或取消当前任务')
            if not matches(text):
                raise ValueError('当前演示支持「把桌上的米老鼠放到沙发上」。其他任务需要接入真实 Agent 后执行。')
            now = self.clock()
            self.task = dict(id=uuid.uuid4().hex, instruction=text, status='running', stage_index=0,
                             started_at=now, ended_at=None, elapsed_s=0, stage_started_at=now,
                             paused_at=None, pause_total=0, progress=0, source='demo',
                             verified=False, simulated_verified=False, stage_result_index=-1,
                             interpretation='演示样例将「米老师玩偶」对应为米老鼠玩偶；保留用户输入原文。' if '米老师' in text else '目标：米老鼠玩偶 · 桌面 → 沙发')
            self._message('user', text)
            self._message('assistant', '收到。我会按「识别 → 走近 → 抓取 → 搬运 → 放置 → 验证」展示任务流程。当前为模拟执行。')
            self._event('plan', '演示任务已拆解', self.task['interpretation'], status='completed',
                        details={'simulated': True, 'steps': [dict(id=s[0], label=s[1]) for s in STAGES]})
            self._start_stage(0)
            initial = deepcopy(self.task)
            self.idempotency[request_id] = (text, initial)
            if len(self.idempotency) > 200:
                self.idempotency.popitem(last=False)
            return initial

    def _finish(self, status, message):
        self.task['status'] = status
        self.task['ended_at'] = self.clock()
        self.task['simulated_verified'] = status == 'completed'
        if status == 'completed':
            self.task['progress'] = 100
        record = deepcopy(self.task)
        self.tasks.append(record)
        self.tasks = self.tasks[-50:]
        self._append('tasks.jsonl', record)
        self._event('verification' if status == 'completed' else 'system',
                    '演示任务完成' if status == 'completed' else '任务已结束', message, status=status)
        self._message('assistant', message)

    def _advance(self):
        if not self.task or self.task['status'] != 'running':
            return
        now = self.clock()
        elapsed = max(0, now - self.task['started_at'] - self.task['pause_total'])
        self.task['elapsed_s'] = round(min(elapsed, len(STAGES) * STEP_SECONDS), 1)
        self.task['progress'] = min(99, int(elapsed / (len(STAGES) * STEP_SECONDS) * 100))
        while self.task['status'] == 'running' and now - self.task['stage_started_at'] >= STEP_SECONDS:
            index = self.task['stage_index']
            self._complete_stage(index)
            if index + 1 == len(STAGES):
                self._finish('completed', '演示流程已完成：米老鼠从桌面搬运到沙发。此结果为模拟，不代表真机搬运已完成。')
                break
            self.task['stage_index'] += 1
            self.task['stage_started_at'] += STEP_SECONDS
            self._start_stage(index + 1)
        if self.task['status'] == 'running' and now - self.task['stage_started_at'] >= RESULT_AFTER:
            self._complete_stage(self.task['stage_index'])

    def _complete_stage(self, index):
        if self.task['stage_result_index'] >= index:
            return
        self.task['stage_result_index'] = index
        stage = STAGES[index]
        self._event('tool_result', stage[1] + ' · 模拟返回', stage[4], tool=stage[3], status='completed',
                    details={'simulated': True, 'stage': stage[0], 'result': RESULTS[index]})
        self._event('observation', stage[1] + ' · 演示反馈', stage[4], status='completed',
                    details={'simulated': True, 'physical_verification': False})
        self._message('assistant', stage[4])

    def action(self, task_id, action):
        with self.lock:
            self._demo_only()
            self._advance()
            if not self.task or self.task['id'] != task_id:
                raise KeyError('任务不存在')
            status = self.task['status']
            if action == 'pause' and status == 'paused':
                return deepcopy(self.task)
            if action == 'resume' and status == 'running':
                return deepcopy(self.task)
            if action == 'cancel' and status == 'cancelled':
                return deepcopy(self.task)
            if action == 'pause' and status == 'running':
                self.task.update(status='paused', paused_at=self.clock())
                self._event('system', '演示已暂停', '计时与步骤已冻结，继续后从当前步骤恢复。', status='paused')
            elif action == 'resume' and status == 'paused':
                duration = self.clock() - self.task['paused_at']
                self.task['pause_total'] += duration
                self.task['stage_started_at'] += duration
                self.task.update(status='running', paused_at=None)
                self._event('system', '演示已继续', '恢复当前任务步骤。', status='running')
            elif action == 'cancel' and status in ('running', 'paused'):
                self._finish('cancelled', '演示任务已取消。可以重新下达任务。')
            else:
                raise RuntimeError('当前任务状态不支持此操作')
            return deepcopy(self.task)

    def message(self, text, request_id=None):
        if self.inbox:
            return self.inbox.send(text, request_id)
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
            raise ValueError('请输入 1–1000 字的消息')
        with self.lock:
            self._demo_only()
            self._advance()
            # Manual conversation takes over the screen; do not replace it with another autoplay task.
            self.demo_autoplay = False
            if matches(text.strip()):
                if self.task and self.task['status'] in ('running', 'paused'):
                    self._finish('cancelled', '用户重新提交演示指令，结束上一轮模拟任务。')
                task = self.submit(text.strip(), 'message-' + uuid.uuid4().hex)
                return {'accepted': True, 'source': 'demo', 'action': 'demo_task_started',
                        'task': task, 'reply': '已开始搬运交互演示；工具调用和执行反馈均为模拟。'}
            user = self._message('user', text.strip(), channel='chat')
            if self.task and self.task['status'] in ('running', 'paused'):
                reply = f'已收到你的消息。当前演示正在「{STAGES[self.task["stage_index"]][1]}」。真实 Agent 尚未接入，这条消息已保存在演示会话中，不会触发机器人动作。'
            else:
                reply = '已收到你的消息。当前为演示会话，真实 Agent 尚未接入；自定义指令会显示在这里，但不会触发机器人动作。'
            self._message('assistant', reply, channel='chat', reply_to=user['id'])
            return {'accepted': True, 'source': 'demo', 'reply': reply}

    def _demo_telemetry(self):
        tick = self.clock() - self.started_at
        joints = []
        for joint in JOINTS:
            index = joint['hardware_index']
            base = 37 + (index * 7 % 15)
            winding = round(base + math.sin(tick / 9 + index) * 1.1, 1)
            # Explicitly simulated yellow/red examples for unattended presentations.
            # No data here is an actual measurement or a robot protection decision.
            if index == 4:
                winding = round(90 + 3 * math.sin(tick / 12), 1)
            elif index == 10:
                winding = round(86.5 + math.sin(tick / 15), 1)
            joints.append(dict(id=index, name=joint['name'], group=joint_group(joint['name']),
                               case_c=round(winding - 7.4, 1), winding_c=winding, motorstate=0, fresh=True))
        return dict(source='demo', fresh=True, sampled_at=self.clock(), age_s=0, joints=joints, fault_count=0)

    def snapshot(self):
        with self.lock:
            self._advance()
            now = self.clock()
            thermal = self._demo_telemetry() if self.mode == 'demo' else telemetry(self.telemetry_file)
            valid = [max(j['case_c'], j['winding_c']) for j in thermal['joints']
                     if j['fresh'] and j['case_c'] is not None and j['winding_c'] is not None]
            if not self.temperatures or now - self.temperatures[-1]['time'] >= 1:
                self.temperatures.append(dict(time=now, value=max(valid) if valid else None))
                self.temperatures = self.temperatures[-60:]
            owner = read_json(self.owner_file)
            # Owner has no boot id in its legacy schema. File freshness plus source clock,
            # reported separately; never use this display status to authorize commands.
            try:
                owner_age = now - Path(self.owner_file).stat().st_mtime
                mono_age = time.monotonic() - owner['updated_monotonic']
                owner_fresh = 0 <= owner_age <= 2 and 0 <= mono_age <= 2
            except (OSError, TypeError, KeyError):
                owner_fresh = False
            rule = owner.get('thermal_rule') if owner_fresh else None
            rule = rule if rule in ('native90', 'strict91') else self.thermal_rule
            thermal['history'] = self.temperatures
            thermal['rule_source'] = 'owner' if owner_fresh and owner.get('thermal_rule') == rule else 'configured'
            self.thermal_display.apply(thermal, rule)
            camera = (dict(source='recorded_sample', fresh=False, url='/assets/observation.png',
                           frame_id='E050-ed08c869', sampled_at=None, age_s=None,
                           note='历史实拍样本 · 非直播，目标框为演示标注')
                      if self.mode == 'demo' else self.camera.snapshot())
            events = self.events[-60:] if self.mode == 'demo' else agent_events(self.agent_events_file)
            conversation_events = events
            if self.mode == 'readonly':
                observed = next((event for event in reversed(events) if event.get('kind') == 'task'), None)
                if observed is not None:
                    self._observed_task = (observed if isinstance(observed.get('task_id'), str)
                                           and observed['task_id'] and isinstance(observed.get('details'), dict)
                                           else None)
                elif self._observed_task is not None:
                    # A long streamed response must not evict its task from the header.
                    # This is the last observed record, never a live execution authority.
                    conversation_events = [self._observed_task, *events]
            conversation = project(self.mode, self.task, self.messages, conversation_events, now, STAGES)
            inbox_state = self.inbox.snapshot() if self.inbox else None
            if self.inbox:
                incoming = self.inbox.events()
                if incoming:
                    # Keep the current task/tools visible while exposing operator receipts separately.
                    conversation['history'] = incoming[-18:-2]
                    if incoming[-1]['time'] >= (conversation.get('user') or {}).get('time', 0):
                        conversation['user'] = incoming[-2]
                        output_id=conversation['output']['id']
                        public=next((e for e in reversed(events) if e.get('id')==output_id),{})
                        # A later agent reply must not be hidden by an earlier
                        # receipt for an intervention sent during the same turn.
                        if public.get('time',0) < incoming[-2]['time']:
                            conversation['output'] = dict(id=incoming[-1]['id']+':'+incoming[-1]['text'],
                                                          text=incoming[-1]['text'], streaming=False)

            return deepcopy(dict(schema_version=1, server_time=now, mode=self.mode,
                presentation={'aspect_ratio': '16:9', 'width': 1920, 'height': 1080,
                              'autoplay': self.demo_autoplay},
                capabilities={'submit_task': self.mode == 'demo', 'task_control': self.mode == 'demo',
                              'chat': self.mode == 'demo' or self.inbox is not None, 'physical_motion': False,
                              'agent_connected': bool(inbox_state and inbox_state['agent_connected']),
                              'agent_autowake': bool(inbox_state and inbox_state['dispatcher']['online'])},
                operator_inbox=inbox_state,
                robot={'name': 'Unitree G1', 'joint_count': 29,
                       'status': 'demo' if self.mode == 'demo' else 'observing' if thermal['fresh'] else 'offline',
                       'owner_phase': owner.get('phase') if owner_fresh else None,
                       'owner_fresh': owner_fresh, 'fault': owner.get('fault') if owner_fresh else None},
                camera=camera, telemetry=thermal, task=self.task,
                stages=[dict(id=s[0], label=s[1], description=s[2]) for s in STAGES],
                messages=self.messages, events=events, conversation=conversation,
                event_cursor=self.sequence, history=self.tasks[-20:]))

    def event_page(self, after=0):
        with self.lock:
            return deepcopy({'events': [e for e in self.events if e['seq'] > after], 'cursor': self.sequence,
                             'reset_required': bool(self.events and after and after < self.events[0]['seq'] - 1)})

    def export(self):
        value = self.snapshot()
        value['export_note'] = '演示任务、温度及判断均为模拟；历史照片不代表实时观测。' if self.mode == 'demo' else '实机只读观测快照，未授权或提交运动。'
        return value
