"""Real simulator control checks; this is never counted as VLA task success."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .client import HarnessClient
from .run import decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready', required=True)
    args = parser.parse_args()
    client = HarnessClient(args.ready)
    output = Path(args.ready).resolve().parent
    records = []
    try:
        obs = client.call('observe')['observation']
        # Physical motion checks use only proprioception, not scene object poses.
        before = obs['robot']
        row = client.call('execute', decision('move_base', obs, yaw_rps=.20, steps=30,
                          reasoning='control smoke: small physical yaw and zero-velocity braking'))
        records.append(row['execution'])
        obs = row['observation']
        yaw_delta = math.atan2(math.sin(obs['robot']['base_yaw_rad'] - before['base_yaw_rad']),
                               math.cos(obs['robot']['base_yaw_rad'] - before['base_yaw_rad']))
        if not .02 < yaw_delta < .35:
            raise RuntimeError(f'Physical base yaw is not verified: delta={yaw_delta}')
        before_trunk = obs['robot']['trunk_qpos']
        target = list(before_trunk)
        target[3] += .10 if target[3] + .10 < obs['robot']['trunk_limits'][3][1] else -.10
        row = client.call('execute', decision('look', obs, trunk_qpos=target, steps=30,
                          reasoning='control smoke: physical torso yaw changes the head view'))
        records.append(row['execution'])
        obs = row['observation']
        trunk_delta = obs['robot']['trunk_qpos'][3] - before_trunk[3]
        if abs(trunk_delta) < .025:
            raise RuntimeError(f'Physical trunk motion is not verified: delta={trunk_delta}')
        if obs['capabilities']['gimbal_enabled']:
            initial_path = Path(obs['cameras']['head']['path'])
            initial_hash = hashlib.sha256(initial_path.read_bytes()).hexdigest()
            row = client.call('execute', decision('look_head', obs, head_yaw_rad=.30,
                              head_pitch_rad=.15, steps=24,
                              reasoning='control smoke: virtual head changes the actual head RGB'))
            records.append(row['execution'])
            obs = row['observation']
            if any(abs(a-b) > 1e-5 for a, b in zip(obs['robot']['head_gimbal'], [.30,.15])):
                raise RuntimeError('Gimbal did not reach its commanded absolute angles')
            if hashlib.sha256(Path(obs['cameras']['head']['path']).read_bytes()).hexdigest() == initial_hash:
                raise RuntimeError('Head image unchanged after virtual gimbal motion')
        result = client.call('close', {'reason': 'control_smoke_complete'})
        report = {'status': 'passed', 'task_success_tested': False, 'records': records,
                  'base_yaw_delta_rad': yaw_delta, 'trunk_yaw_delta_rad': trunk_delta, 'result': result}
    except BaseException as exc:
        report = {'status': 'failed', 'error': str(exc), 'records': records}
        try:
            report['stop_receipt'] = client.stop()
        except Exception as stop_error:
            report['stop_error'] = str(stop_error)
        (output / 'control_smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        raise
    (output / 'control_smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'control_smoke': report['status'], 'steps': result['steps']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
