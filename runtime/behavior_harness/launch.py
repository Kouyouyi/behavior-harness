"""Launch an isolated episode with GPU admission and owned-process cleanup."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from .planner import validate_planner_transport
from .run import validate_planner_timeout

ROOT = Path(__file__).resolve().parent.parent
SUDO = Path('/home/cody/.codex/skills/environment/scripts/sudo_run.sh')
SIM_PYTHON = Path('/home/cody/miniconda3/envs/behavior-2026/bin/python')


def gpu_free_mib():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    return int(output.splitlines()[0].strip())


def wait_for_gpu(required, deadline, output):
    while True:
        free = gpu_free_mib()
        status = {'status': 'waiting_for_gpu' if free < required else 'gpu_available',
                  'free_mib': free, 'required_mib': required, 'checked_at': datetime.now().astimezone().isoformat()}
        (output / 'launch_status.json').write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps(status), flush=True)
        if free >= required:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f'GPU still has {free} MiB free; need {required} MiB')
        time.sleep(20)


def ensure_port_free(port):
    with socket.socket() as sock:
        # Match HTTPServer's allow_reuse_address behavior: a previous episode's
        # closed TCP connections may still be in TIME_WAIT. SO_REUSEADDR permits
        # those to be reused but does not share a port with an active listener.
        # Do not enable SO_REUSEPORT, which could mask an existing live service.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', port))
        except OSError as exc:
            raise RuntimeError(f'Port {port} is occupied; select another port instead of reusing an unknown server') from exc


def root_process(command, pid_path, log, cwd=ROOT):
    # Positional arguments keep paths out of shell syntax. The written PID is
    # the process that execs the requested runtime, not the sudo wrapper.
    return subprocess.Popen([str(SUDO), 'bash', '-c',
        'printf "%s\\n" "$$" > "$1"; cd "$2"; shift 2; exec "$@"',
        'behavior-harness-owned-process', str(pid_path), str(cwd), *command],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def terminate_owned(pid_file, needle):
    if not pid_file.exists():
        return
    pid = int(pid_file.read_text().strip())
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return
    if needle in command:
        subprocess.run([str(SUDO), 'kill', '-TERM', str(pid)], check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True, help='New episode directory; existing directories rejected')
    parser.add_argument('--mode', choices=['assisted', 'baseline', 'control_smoke'], default='assisted')
    parser.add_argument('--head-mode', choices=['physical_trunk', 'virtual_gimbal'])
    parser.add_argument('--language-context-enabled', action=argparse.BooleanOptionalAction, default=None,
                        help='Opt into a separate visual-language-assisted experiment; forbidden for baselines')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--model')
    parser.add_argument('--planner-transport', choices=['auto', 'chatgpt_https'], default=None,
                        help='Optional ChatGPT-signed-in HTTPS planner transport; assisted only')
    parser.add_argument('--planner-timeout', type=float, default=None,
                        help='Total wait for one planner CLI call, up to 600 seconds; config or default 240')
    parser.add_argument('--max-turns', type=int, default=80)
    parser.add_argument('--rpc-port', type=int, default=8766)
    parser.add_argument('--policy-port', type=int, default=8011)
    parser.add_argument('--min-free-mib', type=int, default=16000)
    parser.add_argument('--gpu-wait-seconds', type=int, default=0)
    parser.add_argument('--startup-timeout', type=int, default=600)
    args = parser.parse_args()
    def on_termination(_signum, _frame):
        raise KeyboardInterrupt('launcher termination requested')
    signal.signal(signal.SIGTERM, on_termination)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    if args.planner_transport is not None:
        config['planner_transport'] = args.planner_transport
    planner_transport = validate_planner_transport(config.get('planner_transport', 'auto'), mode=args.mode)
    if args.planner_timeout is not None:
        config['planner_timeout_s'] = args.planner_timeout
    planner_timeout = validate_planner_timeout(config.get('planner_timeout_s', 240))
    if args.language_context_enabled is not None:
        config['language_context_enabled'] = args.language_context_enabled
    context_enabled = config.get('language_context_enabled', False)
    if type(context_enabled) is not bool:
        raise ValueError('language_context_enabled must be a boolean')
    if context_enabled and args.mode != 'assisted':
        raise ValueError('Language context is only allowed in a separately labelled assisted episode')
    config.update(output_dir=str(output), run_mode='baseline' if args.mode == 'baseline' else 'assisted',
                  policy_port=args.policy_port, policy_kind='local' if args.mode == 'control_smoke' else 'websocket')
    if args.head_mode:
        config['head_mode'] = args.head_mode
    if args.max_steps:
        config['max_steps'] = args.max_steps
    config_path = output / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    (output / 'launcher.pid').write_text(str(os.getpid()) + '\n')
    policy = sim = runner = None
    policy_log = sim_log = None
    policy_process_needle = 'serve_b1k.py'
    ready = output / 'ready.json'
    gpu_deadline = time.monotonic() + args.gpu_wait_seconds
    try:
        wait_for_gpu(args.min_free_mib, gpu_deadline, output)
        ensure_port_free(args.rpc_port)
        if args.mode != 'control_smoke':
            checkpoint = config.get('policy_checkpoint')
            if checkpoint is None:
                if config.get('task_name') != 'turning_on_radio':
                    raise ValueError('A different task requires its own policy_checkpoint and policy configuration')
                checkpoint = '/root/download/models/pi05-BEHAVIOR-turning-on-radio/pi05_turn_on_the_radio'
            ensure_port_free(args.policy_port)
            policy_log = (output / 'policy.log').open('w')
            if context_enabled:
                policy_process_needle = 'behavior_harness.context_policy_service'
                policy = root_process(['env', 'JAX_PLATFORMS=cpu', 'XLA_PYTHON_CLIENT_PREALLOCATE=false',
                    'PYTHONUNBUFFERED=1', f'PYTHONPATH={ROOT}', '/root/openpi/.venv/bin/python',
                    '-m', policy_process_needle, '--robot', config.get('policy_robot', 'b1k/R1Pro'),
                    '--task', config.get('policy_task', 'b1k/' + config['task_name']),
                    '--repo-id', config.get('policy_repo_id', config['task_name']),
                    '--config', config.get('policy_config', 'pi05_b1k'), '--checkpoint', checkpoint,
                    '--audit', str(output / 'context_policy_audit.jsonl'), '--port', str(args.policy_port)],
                    output / 'policy.pid', policy_log, cwd='/root/openpi')
            else:
                policy = root_process(['env', 'JAX_PLATFORMS=cpu', 'XLA_PYTHON_CLIENT_PREALLOCATE=false',
                    'PYTHONUNBUFFERED=1', '/root/openpi/.venv/bin/python', '/root/openpi/scripts/b1k/serve_b1k.py',
                    '--robot', config.get('policy_robot', 'b1k/R1Pro'),
                    '--task', config.get('policy_task', 'b1k/' + config['task_name']),
                    '--repo-id', config.get('policy_repo_id', config['task_name']),
                    '--policy.config', config.get('policy_config', 'pi05_b1k'), '--policy.dir', checkpoint,
                    '--control_mode', 'receding_horizon', '--action_horizon', '16', '--port', str(args.policy_port)],
                    output / 'policy.pid', policy_log, cwd='/root/openpi')
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if policy.poll() is not None:
                    raise RuntimeError('Policy exited during startup; inspect policy.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.policy_port}/healthz', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError('Policy readiness timeout; inspect policy.log')
        # Loading the CPU policy can take tens of seconds. Recheck immediately
        # before Isaac starts, without granting a second GPU waiting budget.
        wait_for_gpu(args.min_free_mib, gpu_deadline, output)
        sim_log = (output / 'simulator.log').open('w')
        sim = root_process(['bash', str(ROOT / 'scripts/start_harness.sh'), str(config_path), str(args.rpc_port)],
                           output / 'simulator.pid', sim_log)
        deadline = time.monotonic() + args.startup_timeout
        while not ready.exists():
            if sim.poll() is not None:
                raise RuntimeError('Simulator exited during startup; inspect simulator.log and result.json')
            if time.monotonic() >= deadline:
                raise TimeoutError('Simulator readiness timeout; inspect simulator.log')
            time.sleep(2)
        (output / 'launch_status.json').write_text(json.dumps({'status': 'running', 'ready_file': str(ready)}) + '\n')
        if args.mode == 'control_smoke':
            command = [sys.executable, '-m', 'behavior_harness.smoke', '--ready', str(ready)]
        else:
            command = [sys.executable, '-m', 'behavior_harness.run', '--ready', str(ready),
                       '--mode', args.mode, '--max-turns', str(args.max_turns)]
            if args.model:
                command += ['--model', args.model]
            if planner_transport != 'auto':
                command += ['--planner-transport', planner_transport]
            if 'planner_timeout_s' in config:
                command += ['--planner-timeout', str(planner_timeout)]
        runner = subprocess.Popen(command, cwd=ROOT)
        returncode = runner.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        sim.wait(timeout=60)
        (output / 'launch_status.json').write_text(json.dumps({'status': 'finished', 'result_file': str(output / 'result.json')}) + '\n')
    except BaseException as exc:
        (output / 'launch_status.json').write_text(json.dumps({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}) + '\n')
        raise
    finally:
        if runner and runner.poll() is None:
            # Let the runner's Python finally block stop its separate Codex
            # process group and preserve the original inference/episode error.
            runner.send_signal(signal.SIGINT)
            try:
                runner.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print('Runner cleanup still pending; its PID is ' + str(runner.pid), flush=True)
        if ready.exists():
            try:
                from .client import HarnessClient
                HarnessClient(ready).stop()
            except Exception:
                pass
        if sim and sim.poll() is None and not (output / 'simulator.pid').exists():
            time.sleep(1)
        terminate_owned(output / 'simulator.pid', 'behavior_harness.service')
        if sim:
            try:
                sim.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print('Simulator shutdown still pending; inspect its exact PID. No forced kill was issued.', flush=True)
        if policy and policy.poll() is None and not (output / 'policy.pid').exists():
            time.sleep(1)
        terminate_owned(output / 'policy.pid', policy_process_needle)
        if policy:
            try:
                policy.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        for stream in (sim_log, policy_log):
            if stream:
                stream.close()


if __name__ == '__main__':
    main()
