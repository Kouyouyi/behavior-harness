"""Start the operator's G1 experiment. Never sends O or stops live native control."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import g1_manage_terminals as terminals
from scripts.g1_prepare_native import prepare
from scripts.g1_vla_trial_terminal import health_reason
from services.controller.config import Configuration
from services.g1_sonic.vla import DEFAULT_MODEL
from services.console.adapters import BOOT_ID

DATA_PYTHON = Path(sys.executable)
RTC_PYTHON = Path('/home/user/code/Isaac-GR00T-rtc/.venv/bin/python')
INFERENCE_PYTHON = Path('/home/user/GR00T-WholeBodyControl/.venv_inference/bin/python')
TELEMETRY_PYTHON = Path('/home/user/GR00T-WholeBodyControl/.venv_teleop/bin/python')
DEFAULT_BUILD = Path('/home/user/code/rax/worktrees/g1-sonic/runtime/g1/build-stream-timing-field-v2')
OWNER_MODULES = {'g1_sonic.ankle_executor', 'g1_sonic.executor', 'g1_sonic.commission'}
TASK = ('等待 UI 提交任务后再执行。观察当前场景，按需调用移动、VLA、左/右/双手开合。'
        '移动速度参考固定0.3，平移单位m/s。VLA每段最多15秒，段间观察并读取用户介入。'
        '准备会张开双手，持物时不能复位。会话租约不限时，不自行设置期限。'
        '禁止自动O或终止原生控制；停机由用户手动执行或明确授权当次停止。')


def read(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write(path, value):
    path = Path(path)
    temp = path.with_name('.'+path.name)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def module(process):
    argv = process.argv
    return argv[argv.index('-m')+1] if '-m' in argv and argv.index('-m')+1 < len(argv) else None


def option(process, flag):
    argv = process.argv
    return argv[argv.index(flag)+1] if flag in argv and argv.index(flag)+1 < len(argv) else None


def native_live(processes):
    return any(Path(p.argv[0]).name == 'g1_field' for p in processes.values())


def manual_exit(run, process):
    """A recorded manual O must postdate this exact process's start."""
    if not run or not Path(run).is_absolute():
        return False
    path = Path(run)/'native-exit.json'
    row = read(path)
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                if line.startswith('btime '))
    began = boot+int(process.start)/os.sysconf('SC_CLK_TCK')
    return (row.get('reason') == 'operator_keyboard_O' and row.get('exit_code') == 0
            and path.stat().st_mtime >= began)


def retirement_plan(processes, runs_root):
    if native_live(processes):
        raise RuntimeError('原生控制仍运行；不会重启或关闭它。请先按现场流程手动 O。')
    selected = {}
    for p in processes.values():
        if module(p) in OWNER_MODULES:
            run = option(p, '--run')
            if not run or Path(run).parent != runs_root or not manual_exit(run, p):
                raise RuntimeError(f'已有 owner PID {p.pid}，没有可核实的本轮手动 O 退出记录；不自动关闭。')
            selected[p.pid] = p
    for path in runs_root.glob('*/background-services.json'):
        run = path.parent
        for name, entry in read(path).items():
            if name not in ('frontend', 'ui', 'ui_agent') or not isinstance(entry, dict):
                continue
            p = processes.get(entry.get('pid'))
            if not p or list(p.argv) != entry.get('command'):
                continue
            expected = {'frontend': 'services.controller.frontend', 'ui_agent': 'harness.codex_cli.ui_agent',
                        'ui': 'services.console.server'}[name]
            legacy_ui = name == 'ui' and len(p.argv) == 2 and p.argv[1] == str(run/'console.py')
            if module(p) != expected and not legacy_ui:
                continue
            if not manual_exit(run, p):
                raise RuntimeError(f'旧 {name} PID {p.pid} 尚无对应的手动 O 退出记录；请先结束旧轮次。')
            if name == 'ui_agent' and read(run/'ui-agent-status.json').get('phase') == 'running':
                raise RuntimeError('旧 UI agent 仍在处理任务，请等待它返回后重试。')
            selected[p.pid] = p
    return list(selected.values())


def retire(processes):
    for p in processes:
        current = terminals.snapshot()
        if native_live(current):
            raise RuntimeError('检测到原生控制启动，已取消旧服务清理。')
        if current.get(p.pid) != p:
            raise RuntimeError('进程身份变化，取消清理。')
        fd = os.pidfd_open(p.pid)
        try:
            latest = terminals.snapshot()
            if native_live(latest) or latest.get(p.pid) != p:
                raise RuntimeError('进程身份变化，取消清理。')
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        finally:
            os.close(fd)
    wait_for('旧轮次服务退出', lambda: all(terminals.snapshot().get(p.pid) != p for p in processes), 15)


def wait_for(label, check, seconds):
    deadline = time.monotonic()+seconds
    notice = 0.
    while time.monotonic() < deadline:
        value = check()
        if value:
            print(label+'：就绪', flush=True)
            return value
        if time.monotonic() >= notice:
            print('等待 '+label+'…', flush=True)
            notice = time.monotonic()+10
        time.sleep(.25)
    raise RuntimeError(label+'等待超时；查看本轮日志。不会自动 O 或关闭 owner/native。')


def port_open(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.2):
            return True
    except OSError:
        return False


def camera_ready(path):
    row = read(Path(path)/'latest.json')
    ident = row.get('frame_id', '')
    return (row.get('host_boot_id') == BOOT_ID and row.get('source') == 'live_camera'
            and 0 <= time.monotonic()-row.get('received_monotonic', -1e9) < 2
            and bool(re.fullmatch('[a-f0-9]{32}', ident))
            and (Path(path)/ident/'rgb.png').is_file())


def telemetry_ready(path):
    row = read(path)
    return row.get('host_boot_id') == BOOT_ID and health_reason(row, time.monotonic()) is None


def existing_source(processes, name, flag):
    found = [p for p in processes.values() if module(p) == name]
    if len(found) > 1:
        raise RuntimeError('发现重复服务 '+name+'；请先处理重复进程。')
    if not found:
        return None
    value = option(found[0], flag)
    if not value or not Path(value).is_absolute():
        raise RuntimeError('已有服务输出路径不明确：'+name)
    if name.endswith('telemetry') and option(found[0], '--interface') not in (None, 'eno1'):
        raise RuntimeError('已有遥测使用不同网卡。')
    if name.endswith('camera_service') and option(found[0], '--robot') not in (None, '192.168.123.164'):
        raise RuntimeError('已有相机连接不同机器人。')
    return Path(value).resolve()


def preflight(args):
    from harness.codex_cli.launcher import binary
    paths = [DATA_PYTHON, RTC_PYTHON, INFERENCE_PYTHON, TELEMETRY_PYTHON,
             Path(DEFAULT_MODEL)/'model.safetensors.index.json', args.native_build/'g1_field',
             args.native_build/'manifest.json', Path('/home/user/code/CoMotion/env.local.sh')]
    missing = [str(p) for p in paths if not p.is_file()]
    codex = binary()
    if not args.prepare_only:
        missing += [name for name in ('gnome-terminal', 'firefox', 'flock', 'script') if not shutil.which(name)]
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            missing.append('桌面 DISPLAY/WAYLAND_DISPLAY')
    if missing:
        raise RuntimeError('缺少依赖：'+', '.join(missing))
    if not 1024 <= args.ui_port <= 65535 or args.ui_port in (5550, 5555, 17586, 17587):
        raise RuntimeError('UI 端口无效或与机器人服务冲突。')
    print('依赖检查通过；Codex: '+codex, flush=True)


def prepare_run(args):
    run = args.run
    prepare(run, native_build=args.native_build, vla=True, locomotion=True)
    config = Configuration(root=run/'session', task=run.name, robot='g1_sonic',
                           task_description=TASK, backend_options=read(run/'backend-options.json'),
                           agent_provider='codex', subagents=0)
    write(run/'operator-input.json', config.to_dict())
    address = '/tmp/rax-g1-'+uuid.uuid4().hex[:16]+'.sock'
    values = dict(G1_RUN=str(run), G1_CONFIG=str(run/'operator-input.json'),
                  G1_PY=str(DATA_PYTHON), RAX_MCP_SOCKET=address)
    (run/'experiment.env').write_text(''.join(f'export {k}={shlex.quote(v)}\n' for k, v in values.items()))
    for name, command in [
        ('owner', ['bash', str(run/'owner.sh')]),
        ('native', [str(DATA_PYTHON), str(ROOT/'scripts/g1_vla_trial_terminal.py'), '--run', str(run)])]:
        # Closing a failed launcher never sends a signal to these windows.
        (run/f'window-{name}.sh').write_text(
            '#!/usr/bin/env bash\nset -o pipefail\ncd '+shlex.quote(str(ROOT))+'\n'
            'export PYTHONPATH='+shlex.quote(str(ROOT))+'\n'
            +shlex.join(command)+' 2>&1 | tee -a '+shlex.quote(str(run/f'{name}-window.log'))+'\n'
            'printf "\\nProcess finished. Press Enter to close.\\n"\nread -r\n')
    version = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, text=True, capture_output=True)
    write(run/'source-version.json', dict(commit=version.stdout.strip(),dirty=bool(dirty.stdout.strip())))
    return address


class Startup:
    def __init__(self, args, address):
        self.args = args
        self.run = args.run
        self.address = address
        self.services = {}
        self.children = {}
        self.control_launched = False

    def start(self, name, command):
        command = [str(v) for v in command]
        with (self.run/(name+'-background.log')).open('ab') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=os.environ | {'PYTHONPATH': str(ROOT)},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[name] = process
        self.services[name] = dict(pid=process.pid, command=command)
        write(self.run/'background-services.json', self.services)
        return process

    def alive(self):
        for name, child in self.children.items():
            if child.poll() is not None:
                raise RuntimeError(f'{name} 已退出，见 {self.run}/{name}-background.log')

    def wait(self, label, check, seconds=60):
        def probe():
            self.alive()
            return check()
        return wait_for(label, probe, seconds)

    def ensure_model(self):
        if not port_open(5550):
            # A server still loading weights has not bound its port yet: don't start a second one.
            loading = [p for p in terminals.snapshot().values()
                       if any(Path(a).name == 'g1_vla_server.py' for a in p.argv[1:2])]
            if not loading:
                self.start('model', [RTC_PYTHON, ROOT/'scripts/g1_vla_server.py',
                                    '--cpu-threads', '1', '--rtc-delay-steps', '8'])
            self.wait('GR00T 模型服务', lambda: port_open(5550), 300)
        code = '''from services.g1_sonic.vla_runtime import dependencies,identity,request_interval
from services.g1_sonic.vla import DEFAULT_MODEL
import json
rtc=dependencies()
c=rtc.PolicyClient("127.0.0.1",5550,timeout_ms=3000)
try:
    assert c.call_endpoint("get_rax_identity",requires_input=False)==identity(DEFAULT_MODEL), "家庭模型身份不匹配"
    cfg=c.get_rtc_config()
    assert cfg["rtc_delay_steps"]==8 and request_interval(cfg)==30, "RTC配置不是delay=8/interval=30"
    print(json.dumps(dict(ready=True,rtc_config=cfg)))
finally:c.close()
'''
        result = subprocess.run([str(INFERENCE_PYTHON), '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=45)
        (self.run/'model-check.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError('模型身份/RTC 检查失败，见 model-check.log；不会替换已有模型服务。')
        write(self.run/'model-ready.json', json.loads(result.stdout.strip().splitlines()[-1]))
        print('家庭模型就绪：delay=8，request interval=30', flush=True)

    def sensors(self):
        processes = terminals.snapshot()
        camera = existing_source(processes, 'services.g1_sonic.camera_service', '--output')
        telemetry = existing_source(processes, 'services.console.telemetry', '--output')
        if camera:
            (self.run/'camera').symlink_to(camera)
        else:
            self.start('camera', [DATA_PYTHON, '-m', 'services.g1_sonic.camera_service',
                '--receiver-ip', self.args.receiver_ip, '--output', self.run/'camera', '--seconds', '0'])
        if telemetry:
            (self.run/'telemetry.json').symlink_to(telemetry)
        else:
            self.start('telemetry', [TELEMETRY_PYTHON, '-m', 'services.console.telemetry',
                '--interface', 'eno1', '--output', self.run/'telemetry.json'])
        write(self.run/'sensor-sources.json', dict(camera=str(camera or self.run/'camera'),
                                                  telemetry=str(telemetry or self.run/'telemetry.json')))
        self.wait('相机', lambda: camera_ready(self.run/'camera'))
        self.wait('29 电机实时遥测', lambda: telemetry_ready(self.run/'telemetry.json'))

    def mcp_ready(self):
        if not Path(self.address).exists():
            return False
        from auto_research_agent.api.runtime import Context
        previous = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(3)
            with Context(self.address) as ctx:
                return ctx.value('status').get('task') == self.run.name
        finally:
            socket.setdefaulttimeout(previous)

    def ui_ready(self):
        path = self.run/'ui-background.log'
        match = re.search(r'G1 Console: (http://127\.0\.0\.1:\d+/\?token=\S+)', path.read_text() if path.exists() else '')
        if not match:
            return False
        self.url = match.group(1)
        from services.dashboard.server import COOKIE
        req = urllib.request.Request(f'http://127.0.0.1:{self.args.ui_port}/api/v1/state',
                                     headers={'Cookie': COOKIE+'='+self.url.split('token=', 1)[1]})
        with urllib.request.urlopen(req, timeout=2) as response:
            return json.load(response)['capabilities'].get('chat')

    def owner_ready(self, initialized=False):
        if (self.run/'native-exit.json').exists():
            raise RuntimeError('原生已经退出；不会自动重启。')
        row = read(self.run/'ankle_status.json')
        if row.get('phase') == 'fault_idle':
            raise RuntimeError('owner 故障：'+str(row.get('fault'))+'；不会自动 O。')
        fresh = 0 <= time.monotonic()-row.get('updated_monotonic', -1e9) < .25
        if initialized:
            return fresh
        return fresh and row.get('phase') == 'ready' and row.get('lease_unlimited') is True

    def launch(self):
        if port_open(self.args.ui_port):
            raise RuntimeError('UI 端口已被其他服务占用；不会关闭未知进程。请用 --ui-port 指定空闲端口。')
        self.ensure_model()
        self.sensors()
        self.start('frontend', [DATA_PYTHON, '-m', 'services.controller.frontend',
            '--config', self.run/'operator-input.json', '--socket', self.address])
        self.wait('RAX MCP', self.mcp_ready)
        self.start('ui', [DATA_PYTHON, '-m', 'services.console.server', '--mode', 'readonly',
            '--port', self.args.ui_port, '--storage', self.run/'console',
            '--camera-directory', self.run/'camera', '--telemetry-file', self.run/'telemetry.json',
            '--owner-file', self.run/'ankle_status.json', '--agent-events-file', self.run/'agent-events.jsonl',
            '--message-directory', self.run])
        self.wait('Firefox UI 服务', self.ui_ready)
        from harness.codex_cli.ui_bridge import emit
        emit(self.run, self.run.name, 'message', role='assistant',
             text='模型、相机和遥测已就绪，正在启动站立。请等启动就绪提示后提交任务。')
        if terminals.active_control(terminals.snapshot()):
            raise RuntimeError('检测到另一套控制进程；取消启动，不关闭它。')
        # Sequential creation prevents the terminal cleaner from racing a new native window.
        self.control_launched = True
        subprocess.run(['gnome-terminal', '--window', '--title', 'G1 Owner — Unlimited',
                        '--', 'bash', str(self.run/'window-owner.sh')], check=True)
        self.wait('身体 owner 初始化', lambda: self.owner_ready(initialized=True), 90)
        subprocess.run(['gnome-terminal', '--window', '--title', 'G1 Native — operator O only',
                        '--', 'bash', str(self.run/'window-native.sh')], check=True)
        with (self.run/'firefox.log').open('ab') as log:
            subprocess.Popen(['firefox', '--new-tab', self.url], stdout=log, stderr=log,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        self.wait('站立稳定', self.owner_ready, 90)
        self.start('ui_agent', [DATA_PYTHON, '-m', 'harness.codex_cli.ui_agent',
            '--config', self.run/'operator-input.json', '--socket', self.address])
        self.wait('UI Agent', lambda: read(self.run/'ui-agent-status.json').get('phase') in ('idle', 'running'))
        emit(self.run, self.run.name, 'message', role='assistant',
             text='启动就绪：会话不限时。请在 UI 输入任务，Agent 将按需准备姿态、移动、调用15秒VLA或开合手；不会自动 O。')
        write(self.run/'startup-ready.json', dict(ready=True, session_lease_unlimited=True,
              automatic_native_stop=False, automatic_vla_prepare=False, socket=self.address))
        shutil.copyfile(self.run/'experiment.env', ROOT/'runtime/g1/current.env')
        print(f'启动就绪。实验目录：{self.run}\nUI：{self.url}\n停止由操作者在原生窗口按 O。', flush=True)

    def release_failed_ui(self):
        """Before any control launch only, release our own frontend/UI port for retry."""
        if self.control_launched:
            return
        for name in ('ui', 'frontend'):
            child = self.children.get(name)
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(name+' 未退出；保留进程，请检查日志。', file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', help='Read-only dependency/process check; start nothing')
    group.add_argument('--prepare-only', action='store_true', help='Freeze a new run; start no service or hardware')
    parser.add_argument('--run', type=Path, help='New directory directly under this checkout runtime/g1')
    parser.add_argument('--native-build', type=Path, default=DEFAULT_BUILD)
    parser.add_argument('--receiver-ip', default='192.168.123.76')
    parser.add_argument('--ui-port', type=int, default=8782)
    args = parser.parse_args(argv)
    runs = ROOT/'runtime/g1'
    args.run = (args.run or runs/('agent-'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    args.native_build = args.native_build.resolve()
    if args.run.parent != runs.resolve():
        parser.error('--run must be directly under '+str(runs))
    prepared = False
    startup = None
    try:
        preflight(args)
        if args.check:
            plan = retirement_plan(terminals.snapshot(), runs.resolve())
            print('检查完成；没有启动或关闭任何服务。已手动 O 后可清理的旧进程：'+str([p.pid for p in plan]))
            return 0
        lock = open(f'/tmp/rax-g1-start-{os.getuid()}.lock', 'a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.prepare_only:
                prepare_run(args)
                print('仅生成配置，未启动任何服务：'+str(args.run))
                return 0
            plan = retirement_plan(terminals.snapshot(), runs.resolve())
            address = prepare_run(args)
            prepared = True
            if plan:
                retire(plan)
            terminals.clean(runs.resolve(), args.run, apply=True)
            startup = Startup(args, address)
            startup.launch()
        finally:
            lock.close()
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        if startup is not None:
            startup.release_failed_ui()
        print(f'启动未完成：{exc}\n本轮目录：{args.run}\n未发送 O，也未关闭本轮 owner/native。', file=sys.stderr)
        if prepared:
            write(args.run/'startup-error.json', dict(error=str(exc), automatic_native_stop=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
