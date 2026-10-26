"""Wake one bounded Codex turn per UI batch, with public events and no native control.

The operator starts this service alongside the existing frontend/UI. It stays idle
between messages, does not replay claimed work after a crash, and never starts or
stops the native controller or body owner.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .launcher import ROOT, binary, prepare, settings, inherited_mcp, pause_active
from .ui_bridge import emit, ALLOWED
from services.controller.config import Configuration
from services.g1_sonic.messages import Inbox
from services.g1_sonic.vla import atomic
from auto_research_agent.api.runtime import Context

INSTRUCTIONS='''你是正在运行的 G1 UI 任务 Agent。以下 UI 消息就是本轮用户任务，已经提交授权，不要等待用户在另一个终端重复说“开始”。
先读取 g1_operator_messages，结合最新指令、g1_status 和 g1_observe 的真实图像判断。用 ack 公开说明执行意图或阻碍，再调用对应工具完成任务；ack 不是动作完成。
如果用户要求抓取桌上米老鼠，现场可达且 owner 就绪时，按需要 prepare、等待准备完成、观察，然后调用 g1_vla(run,task=mickey,steps=750,evidence=实际观察)。不能停在准备完成就结束。
每段 VLA 最多15秒；等待结果期间和每段边界读取新消息，再观察是否抓起。新消息优先改变计划。需要移动时先停 VLA；移动停稳重新观察。准备会打开双手，持物时不得复位。手部工具支持 left/right/both 的 open/close。
完成用户任务、等待必要现场信息或遇到不可继续的问题时，给出公开结论并结束本轮。结束前再次读取消息，处理本轮内新到的指令。不要无限空轮询或重复失败动作。
只能使用配置的 RAX MCP。不能启动/重启/关闭原生或 owner，绝不发送 O、杀控制进程或把计时/故障当停机授权。控制未就绪/已被用户手动停止时说明需现场重新启动，不自行恢复。
需要问现场问题时直接用中文公开输出，用户下一条 UI 回复会自动开始下一轮。历史 ack 消息只作背景，不重新执行旧命令。
'''

FAST_TASK_INSTRUCTIONS = '''
用户要求像RAX一样replay式执行、从桌右侧绕行时，优先g1_replay_route：
preview查看当前起点和路线，起点区域和朝向适用且整条右侧通道（包括小幅转向修正）、
空手及目标处抬臂净空有当前依据后run一次，传start_ready=true。
Python连续执行8段绕桌动作、检查当前观测与停稳，段间不用你推理。
绕行允许10度以内的实测航向残差，5度仅为记录目标，不要因约6度偏差另行结束任务。
这个容差只用于已确认有净空的绕行，不代表抓取或贴近桌角也允许同样偏差。
超过绕行容差才有限修正；修正改善不足1度时停止重复弱指令并返回恢复检查点。
历史像素/深度匹配已删除，start_matches仅为旧参数兼容；不要额外索要画面匹配。
不要将这些已编译路线动作拆回逐段g1_task_step。run最长180秒，过程中代码监听UI新消息。
终点prepare后返回route_complete_review_current_grasp和当前图像；这不是让用户再确认开始。
你要在同一UI轮次检查当前目标与手的关系，可用localize(frame_id,selections=[{x,y}])
通过现有g1_pixel_to_position读取当前像素的相机坐标；仅允许最新未用于动作的帧。
外参未标定，相机坐标不是骨盆/世界坐标，不能据此直接算脚步，也不能把颜色当可达性。
当前可达且接近路径清楚就立即g1_task_step(grasp)，继续check/pause和两帧持物验收。
需调整则g1_task_step移动、重观测、重新prepare；不要重放整条路线或停在prepare就结束。
只有现场信息确实缺失才在UI询问。各步骤返回ok/measured/why，失败检查点不能硬闯。
replay_checkpoint_required不自动代表整轮结束。先处理新UI消息、owner/活动动作和当前图像；
若只是航向检查点且当前几何允许，在同轮用g1_task_step调整/继续。结合recovery中的
next_route_step和remaining_commands规划余下路线；这些是上下文，不是盲目照搬的授权。
不要重复刚失败的弱转向，不要从起点重放已走路段。真实阻碍、观测不足或预算耗尽才报告停止上层尝试。
米老鼠抓取优先使用 g1_task_step 编译步骤，目标3–5分钟，单轮最长10分钟。
这不是动作加速或放宽保护：固定移动参考0.3、移动/转向单段上限、VLA最多750步保持。
先read/ack当前UI消息，然后g1_task_step(action=observe)一次获得新图像和owner。
move/prepare内部自动等待完成并返回新图像，不要另行逐次轮询。
grasp内部提交750步，约2秒后带图返回检查点；正在运行用check，发现抓起用pause。
observe/check/pause只传action即可；不要为查询重新填写动作检查字段。
暂停/结束后用observe取得第二个不同源帧；两帧都确实持物才confirm_hold。
confirm_hold只记录你的图像判断，不会自动松手或复位。用户要抓起后默认保持。
每次动作传唯一request_id、当前frame_id、真实evidence和必要检查布尔值。
不要编造净空；双手空闲等当前明确确认不必反复问。确认只在姿态/区域仍适用时沿用。
未知项一次合并问清，不要每个步骤重复询问同一事实。收到新指令优先处理，不重放旧任务。
参考10月20日下午经验：先寻找目标、从可见通道绕桌对准合适抓取方向，最后接近、
再prepare/grasp，减少移动后重复prepare。最后那轮转向后右手抓起，但87度不能硬编码重放；
根据当前目标、桌角、手部几何和实测航向选择动作。单次失败改变可解释的姿态条件后才再试。
目标不可见先观察/寻找；不能直接按旧路线开走。prepare会打开双手，持物禁止。
每段只用一句话说明动作/关键结果，省去重复状态复述；遇到真实阻碍再解释。
3或5分钟未完成时说明当前阶段；10分钟结束本轮上层尝试并报告，不发送O、不关闭原生。
若接口不可用报告实际错误，不将旧日志或模拟测试当现场成功。
'''


def command(config, config_path, python, codex, socket, run, session, batch):
    # Reuse the operational instructions; only the MCP transport changes.
    prepare(config,config_path,python,codex)
    values=settings(config,config_path,python,inherited_mcp())
    args=['-m','harness.codex_cli.ui_bridge','--socket',str(socket),'--run',str(run),
          '--session',json.dumps(session),'--batch',batch]
    values=[v for v in values if not v.startswith('mcp_servers.rax_robot.args=')]
    values=[v for v in values if not v.startswith('mcp_servers.rax_robot.enabled_tools=')]
    values=[v for v in values if not v.startswith('mcp_servers.rax_robot.tool_timeout_sec=')]
    values+=['mcp_servers.rax_robot.args='+json.dumps(args),
             'mcp_servers.rax_robot.enabled_tools='+json.dumps(sorted(ALLOWED)),
             'mcp_servers.rax_robot.tool_timeout_sec=210']
    cmd=[codex]
    for value in values:cmd+=['-c',value]
    cmd+=['--ask-for-approval','never','exec','--json','--ephemeral','--color','never',
          '--sandbox','read-only','--cd',str(config.root/'agent-workspace')]
    if config.agent_model:cmd+=['--model',config.agent_model]
    return cmd+['-']


class Worker:
    def __init__(self, config, config_path, socket, codex, python=sys.executable, timeout=600):
        self.config=config;self.config_path=Path(config_path);self.socket=socket
        self.codex=codex;self.python=python;self.timeout=timeout
        self.run=Path(config.backend_options['control_directory']).resolve();self.box=Inbox(self.run)
        self.process=None;self.batch=None;self.session=None;self.output=None;self.error=None
        self.started=None;self.carry='';self.failed=False;self.public=False;self.stop_reason=None
        self.lock=None;self.launch_lock=None

    def start(self):
        self.config.root.mkdir(parents=True,exist_ok=True)
        self.lock=(self.run/'ui-agent.lock').open('a')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Excludes the interactive launcher for this MCP session as well.
        self.launch_lock=(self.config.root/'codex-launch.lock').open('a')
        fcntl.flock(self.launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.box.recover_dispatches();self.heartbeat('idle')

    def heartbeat(self, phase, **extra):
        atomic(self.run/'ui-agent-status.json',dict(updated_at=time.time(),pid=os.getpid(),phase=phase,
            batch=self.batch,agent_pid=self.process.pid if self.process else None,
            started_monotonic=self.started,
            session=self.session,automatic_native_stop=False,**extra))

    def launch(self, messages):
        self.session=messages[0]['session'];self.started=time.monotonic()
        self.instruction=messages[-1]['text']
        self.failed=False;self.public=False;self.stop_reason=None;self.carry=''
        folder=self.run/'ui-agent-turns'/self.batch;folder.mkdir(parents=True)
        if self.box.session()!=self.session:raise RuntimeError('Owner changed after UI message claim')
        with Context(self.socket) as connection:
            if connection.value('status').get('task')!=self.config.task:
                raise RuntimeError('MCP socket belongs to a different UI task/session')
        prompt=INSTRUCTIONS+FAST_TASK_INSTRUCTIONS+'\n本轮待处理指令：\n'+json.dumps(messages,ensure_ascii=False)+'\n最近消息背景：\n'+json.dumps(self.box.snapshot()['messages'][-12:],ensure_ascii=False)
        (folder/'prompt.txt').write_text(prompt)
        cmd=command(self.config,self.config_path,self.python,self.codex,self.socket,self.run,self.session,self.batch)
        (folder/'command.json').write_text(json.dumps(cmd,indent=2))
        emit(self.run,self.batch,'task',status='running',details=dict(instruction=messages[-1]['text'],current_step='Agent 已唤醒，正在观察与判断'))
        emit(self.run,self.batch,'message',role='user',text=messages[-1]['text'])
        self.heartbeat('running')
        self.output=(folder/'events.jsonl').open('w+')
        self.error=(folder/'stderr.log').open('w')
        try:
            self.process=subprocess.Popen(cmd,cwd=self.config.root/'agent-workspace',
                env=dict(os.environ,PYTHONPATH=str(ROOT)),stdin=subprocess.PIPE,
                stdout=self.output,stderr=self.error,text=True,start_new_session=True,
                pass_fds=(self.lock.fileno(),self.launch_lock.fileno()))
            self.process.stdin.write(prompt);self.process.stdin.close()
        except Exception:
            if self.process is not None:self.cancel('Agent 启动失败')
            raise
        self.heartbeat('running')

    def drain(self):
        if not self.output:return
        # Parent and child share the write offset; read through a separate handle.
        path=Path(self.output.name)
        if not hasattr(self,'reader') or self.reader is None:self.reader=path.open()
        while True:
            at=self.reader.tell();line=self.reader.readline()
            if not line or not line.endswith('\n'):
                self.reader.seek(at);break
            try:row=json.loads(line)
            except ValueError:continue
            item=row.get('item') or {}
            if row.get('type')=='item.completed' and item.get('type')=='agent_message':
                text=item.get('text','')
                if text:
                    emit(self.run,self.batch,'message',role='assistant',text=text);self.public=True
            elif row.get('type') == 'error' and str(row.get('message','')).startswith('Reconnecting'):
                emit(self.run,self.batch,'message',role='assistant',
                     text='Agent 推理连接正在恢复；尚未因此重放任何机器人动作。')
            elif row.get('type') in ('turn.failed','error'):
                self.failed=True
                emit(self.run,self.batch,'error',text='Agent 推理服务报错；本轮不自动重试，请查看运行日志。')
            # Never expose model reasoning items as public summaries.

    def cancel(self, reason):
        if self.process and self.process.poll() is None and not self.stop_reason:
            self.stop_reason=reason;self.cancel_at=time.monotonic()
            self.process.terminate()  # Only this service's Codex child, never native/owner.
        elif self.process and self.process.poll() is None and time.monotonic()-self.cancel_at>5:
            self.process.kill()

    def finish(self, reason):
        self.drain()
        outcome=pause_active(self.config) if self.process is not None and self.box.session()==self.session else dict(status='no_current_agent_process_or_owner_changed_no_action')
        self.box.finish_dispatch(self.batch,reason)
        emit(self.run,self.batch,'task',status='failed' if self.failed or self.stop_reason else 'waiting',
             details=dict(instruction=getattr(self,'instruction','UI 任务'),current_step=reason))
        if not self.public or self.failed or self.stop_reason:emit(self.run,self.batch,'message',role='assistant',text=reason)
        atomic(self.run/'ui-agent-turns'/self.batch/'result.json',dict(reason=reason,
            exit_code=self.process.returncode if self.process else None,upper_level_pause=outcome,automatic_native_stop=False))
        for obj in (self.output,self.error,getattr(self,'reader',None)):
            if obj:obj.close()
        self.output=None;self.error=None;self.reader=None;self.process=None;self.batch=None
        self.heartbeat('idle',last_result=reason)

    def tick(self):
        if self.process:
            self.heartbeat('running');self.drain()
            if self.stop_reason:self.cancel(self.stop_reason)
            elif self.box.session()!=self.session:self.cancel('owner 会话已更换，本轮已中止；未重放旧指令。')
            elif time.monotonic()-self.started>self.timeout:self.cancel('Agent 本轮处理超时，已结束推理并请求暂停上层动作；未关闭运动控制。')
            code=self.process.poll()
            if code is not None:
                self.failed=self.failed or code!=0
                self.finish(self.stop_reason or ('Agent 执行失败，未自动重试；请检查日志后重新提交。' if self.failed else '本轮 Agent 已返回；未处理的指令不会自动重放，请按当前状态重新提交。'))
            return
        self.heartbeat('idle')
        batch=uuid.uuid4().hex;messages=self.box.claim(batch)
        if not messages:return
        self.batch=batch
        try:self.launch(messages)
        except Exception as exc:
            self.failed=True
            if self.process is None or self.process.poll() is not None:self.finish('Agent 启动失败：'+str(exc))

    def close(self):
        if self.process and self.process.poll() is None:
            self.cancel('后台 Agent 服务关闭；仅暂停上层任务，原生控制保持不变。')
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        if self.batch:self.finish(self.stop_reason or '后台 Agent 服务关闭。')
        self.heartbeat('stopped')
        if self.launch_lock:self.launch_lock.close()
        if self.lock:self.lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--socket',required=True)
    p.add_argument('--codex-bin');p.add_argument('--turn-timeout',type=float,default=600)
    a=p.parse_args()
    if not 30<=a.turn_timeout<=600:p.error('turn timeout must be 30..600 seconds')
    config=Configuration.load(a.config)
    if config.robot!='g1_sonic' or config.agent_provider!='codex' or config.subagents:
        p.error('Requires g1_sonic, codex and subagents=0')
    worker=Worker(config,a.config.resolve(),a.socket,binary(a.codex_bin),timeout=a.turn_timeout)
    worker.start();stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while not stopping:worker.tick();time.sleep(.25)
    finally:worker.close()


if __name__=='__main__':main()
