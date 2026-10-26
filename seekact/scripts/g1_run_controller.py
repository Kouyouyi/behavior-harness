"""Run the repeatable RAX commissioning controller through the actual MCP service.

Requires an already commissioned operator reference executor. Starts/stops only
camera and MCP processes. It never starts/stops the native controller or executor.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'auto_research_agent')]
from services.controller.config import Configuration
from services.g1_sonic.reference import check_pulse
from api.runtime import Context


def wait_for(path, process, timeout=10):
    deadline = time.monotonic()+timeout
    while not path.exists():
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError('Service did not become ready; inspect logs')
        time.sleep(.05)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--executor', type=Path, required=True)
    p.add_argument('--receiver-ip', default='192.168.123.76')
    p.add_argument('--side', choices=['left', 'right'], default='right')
    p.add_argument('--delta-rad', type=float, default=.03)
    p.add_argument('--action',choices=['elbow_pulse','clear_arms','reach_return','grasp_step','walk_step'],default='elbow_pulse')
    p.add_argument('--delta-m',type=float,nargs=3)
    p.add_argument('--rotation-delta',type=float,nargs=3,default=[0.,0.,0.])
    a = p.parse_args()
    check_pulse(a.side, a.delta_rad)
    if a.action=='grasp_step':
        from services.g1_sonic.reach import validate_step
        validate_step(a.delta_m,a.rotation_delta)
    run, executor = a.run.resolve(), a.executor.resolve()
    run.mkdir(parents=True, exist_ok=False)
    options = dict(camera_directory=str(run/'camera'), trace_path=str(executor/'controller_trace.jsonl'),
                   control_directory=str(executor))
    config = Configuration(root=run/'session', task='g1_'+a.action, robot='g1_sonic', backend_options=options)
    config.root.mkdir()
    path = config.root/'operator-input.json'
    path.write_text(json.dumps(config.to_dict(), indent=2)+'\n')
    controller = ROOT/'examples/g1/controller.py'
    (run/'controller.py').write_text(controller.read_text())
    socket_path = Path('/tmp')/('rax-g1-controller-'+str(os.getpid())+'.sock')
    camera = frontend = None
    with (run/'camera.log').open('w') as camera_log, (run/'frontend.log').open('w') as frontend_log:
        try:
            camera = subprocess.Popen([sys.executable, '-m', 'services.g1_sonic.camera_service',
                '--receiver-ip', a.receiver_ip, '--output', str(run/'camera'), '--seconds', '75'],
                cwd=ROOT, stdout=camera_log, stderr=camera_log)
            wait_for(run/'camera/health.json', camera)
            frontend = subprocess.Popen([sys.executable, '-m', 'services.controller.frontend',
                '--config', str(path), '--socket', str(socket_path)], cwd=ROOT, stdout=frontend_log, stderr=frontend_log)
            wait_for(socket_path, frontend)
            spec = importlib.util.spec_from_file_location('g1_controller', controller)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            with Context(str(socket_path)) as ctx:
                result = module.main(ctx, a.side, a.delta_rad,a.action,a.delta_m,a.rotation_delta)
            (run/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result, indent=2))
        except BaseException as error:
            (run/'failure.json').write_text(json.dumps(dict(error=type(error).__name__, reason=str(error)), indent=2)+'\n')
            raise
        finally:
            for proc in (frontend, camera):
                if proc is not None:
                    proc.terminate(); proc.wait(timeout=10)


if __name__ == '__main__':
    main()
