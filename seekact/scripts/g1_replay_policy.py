"""Replay recorded ONNX inputs on CPU; no simulation, transport or robot control.

Hybrid-input ablations characterize a frozen model locally. They are not predicted
physical trajectories or evidence that changing a live token/history is safe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executor', type=Path, required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    events = [json.loads(line) for line in (a.executor/'executor_events.jsonl').read_text().splitlines()]
    job = next(e for e in events if e['event'] == 'job_started' and e['id'] == a.job_id)
    samples = []
    wanted = iter([-1., 5., 10., 12.])
    next_time = next(wanted)
    with (a.executor/'controller_trace.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['monotonic']-job['monotonic'] >= next_time:
                if row['motion'] != 'streamed':
                    raise ValueError('Requested sample outside streamed reference interval')
                samples.append(row)
                try:
                    next_time = next(wanted)
                except StopIteration:
                    break
    if len(samples) != 4:
        raise ValueError('Insufficient recorded observations')
    assets = Path('/home/user/code/CoMotion/experiments/shared_control/assets/local_release_20261004')
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    models = {name: ort.InferenceSession(str(assets/f'model_{name}.onnx'), sess_options=opt,
                                       providers=['CPUExecutionProvider']) for name in ['encoder', 'decoder']}

    def infer(name, obs):
        return models[name].run(None, {'obs_dict': np.asarray(obs, dtype=np.float32)[None]})[0][0]

    anchor = np.asarray(samples[0]['policy_obs'], dtype=np.float32)
    anchor_action = infer('decoder', anchor)
    rows = []
    for row in samples:
        obs = np.asarray(row['policy_obs'], dtype=np.float32)
        decoded = infer('decoder', obs)
        baseline_token = obs.copy(); baseline_token[:64] = anchor[:64]
        baseline_history = anchor.copy(); baseline_history[:64] = obs[:64]
        arrays = dict(recorded_input_action=decoded, baseline_token_action=infer('decoder', baseline_token),
                      baseline_history_action=infer('decoder', baseline_history))
        rows.append(dict(elapsed_s=row['monotonic']-job['monotonic'],
            encoder_replay_max_abs_error=float(np.max(abs(infer('encoder', row['encoder_obs'])-row['token']))),
            decoder_replay_max_abs_error=float(np.max(abs(decoded-row['last_action']))),
            recorded_token_vs_policy_input_max_error=float(np.max(abs(obs[:64]-row['token']))),
            token_max_change_from_anchor=float(np.max(abs(obs[:64]-anchor[:64]))),
            actions_isaac_order={key: value.tolist() for key, value in arrays.items()},
            max_action_change_from_anchor={key: float(np.max(abs(value-anchor_action))) for key, value in arrays.items()}))
    report = dict(job_id=a.job_id, robot_commands_sent=False, runtime='ONNX Runtime CPU',
        onnxruntime_version=ort.__version__, source_executor=str(a.executor.resolve()),
        models_sha256={str(assets/f'model_{name}.onnx'): hashlib.sha256((assets/f'model_{name}.onnx').read_bytes()).hexdigest()
                       for name in models},
        limitations=['Four recorded input snapshots; no closed-loop physics or causal root-cause proof.',
                     'Hybrid inputs are diagnostic only; never injected into the live controller.',
                     'Action values are normalized decoder outputs in Isaac order, not joint angles.'],
        samples=rows)
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, a.output/'g1_replay_policy.py')
    (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps([{key: value for key, value in row.items() if key != 'actions_isaac_order'} for row in rows], indent=2))


if __name__ == '__main__':
    main()
