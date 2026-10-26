"""Exercise the actual RAX MCP socket and Python Context with live RGB-D. No motion."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'auto_research_agent')]
from services.controller.config import Configuration
from api.runtime import Context


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--trace', type=Path)
    a = p.parse_args()
    a.run = a.run.resolve()
    a.run.mkdir(parents=True, exist_ok=True)
    options = dict(camera_directory=str(a.camera.resolve()))
    if a.trace:
        options['trace_path'] = str(a.trace.resolve())
    config = Configuration(root=a.run, task='g1_readonly_smoke', robot='g1_sonic', backend_options=options)
    path = a.run/'operator-input.json'
    path.write_text(json.dumps(config.to_dict(), indent=2)+'\n')
    # Unix socket paths must fit sockaddr_un; evidence remains in the run directory.
    socket_path = Path('/tmp')/('rax-g1-'+str(os.getpid())+'.sock')
    with (a.run/'frontend.log').open('w') as log:
        service = subprocess.Popen([sys.executable, '-m', 'services.controller.frontend', '--config', str(path),
                                     '--socket', str(socket_path)], cwd=ROOT, stdout=log, stderr=log)
        try:
            deadline = time.monotonic()+10
            while not socket_path.exists():
                if service.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Frontend did not start; inspect frontend.log')
                time.sleep(.05)
            with Context(str(socket_path)) as ctx:
                status = ctx.value('g1_status')
                observation = ctx.value('g1_observe')
                folder = a.run/'agent-workspace/runtime/frames'/observation['frame_id']
                valid = cv2.imread(str(folder/'valid.png'), cv2.IMREAD_GRAYSCALE)
                ys, xs = np.where(valid > 0)
                if not len(xs):
                    raise RuntimeError('No valid stereo depth in this observation')
                selected = np.argmin((xs-320)**2+(ys-180)**2)
                x, y = int(xs[selected]), int(ys[selected])
                point = ctx.value('g1_pixel_to_position', frame_id=observation['frame_id'], x=x, y=y, radius=0)
                rejected = None
                try:
                    ctx.value('g1_pixel_to_position', frame_id=observation['frame_id'], x=320, y=180, coordinate_frame='base')
                except RuntimeError as e:
                    rejected = str(e)
                assert rejected, 'Uncalibrated base coordinates unexpectedly accepted'
                report = dict(passed=True, actual_mcp_socket=True, source='live_camera',
                              robot_commands_sent=0, status=status, observation=observation,
                              point=point, selected_valid_pixel=[x, y], base_coordinates_rejection=rejected)
                (a.run/'result.json').write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps(dict(passed=True, frame_id=observation['frame_id'], point=point,
                                     body_feedback='body' in status, robot_commands_sent=0), indent=2))
        finally:
            service.terminate()
            service.wait(timeout=10)


if __name__ == '__main__':
    main()
