"""Summarize archived simulations and model FK of measured real-robot joints."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.g1_sonic.reference import pulse_envelope
from services.g1_sonic.state import NAMES, HARDWARE_TO_ISAAC
from services.g1_sonic.vendor import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    result = dict(simulations=[])
    elbow = NAMES.index('right_elbow_joint')
    for run in a.runs:
        metrics = json.loads((run/'result.json').read_text())
        rows = [json.loads(line) for line in (run/'samples.jsonl').read_text().splitlines()]
        t = np.array([r['elapsed_s'] for r in rows])
        ref = np.array([r['native_q'][elbow] for r in rows])
        q = np.array([r['q'][elbow] for r in rows])
        peak = float(np.max(ref-ref[0]))
        metrics.update(directory=str(run.resolve()), native_elbow_reference_peak_rad=peak,
            elbow_same_direction_response_rad=float(np.max(q-q[0])))
        if metrics['case'] == 'pulse':
            mask = t <= 11
            lags = np.arange(0, 4.001, .02)
            errors = [float(np.sqrt(np.mean((ref[mask]-ref[0]-.03*pulse_envelope(t[mask]-lag)[0])**2))) for lag in lags]
            j = int(np.argmin(errors))
            metrics.update(native_lag_fit_s=float(lags[j]), lag_fit_rmse_rad=errors[j])
        returned = [r['elapsed_s'] for r in rows if r['elapsed_s'] >= 7 and r['native_future_qdq_max_error'] < 1e-4]
        metrics['native_full_reference_return_first_elapsed_s'] = returned[0] if returned else None
        metrics['source_sha256'] = {name: sha(run/name) for name in ['manifest.json', 'result.json', 'samples.jsonl']}
        result['simulations'].append(metrics)
    scene = Path('/home/user/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml')
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    qids = [int(model.jnt_qposadr[model.joint(name).id]) for name in NAMES]
    wrist = model.body('right_wrist_yaw_link').id
    def fk(q):
        data.qpos[:] = model.qpos0; data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        data.qpos[qids] = q; mujoco.mj_forward(model, data)
        return data.xpos[wrist].copy(), Rotation.from_matrix(data.xmat[wrist].reshape(3, 3))
    real = ROOT/'runtime/g1/E007-reference'
    events = [json.loads(line) for line in (real/'executor_events.jsonl').read_text().splitlines()]
    start = next(e for e in events if e['event'] == 'job_started')['monotonic']
    report = json.loads((ROOT/'runtime/g1/E008-rax-roundtrip/result.json').read_text())['result']
    initial, initial_rotation = fk(report['q_before'])
    wanted = iter([7, 10, 12]); next_time = next(wanted); observed = []
    with (real/'controller_trace.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['monotonic']-start >= next_time:
                q = np.asarray(row['body_q'])[HARDWARE_TO_ISAAC]
                position, rotation = fk(q)
                observed.append(dict(elapsed_s=row['monotonic']-start,
                    elbow_return_error_rad=float(abs(q[elbow]-report['q_before'][elbow])),
                    wrist_base_change_m=(position-initial).tolist(),
                    wrist_base_displacement_m=float(np.linalg.norm(position-initial)),
                    wrist_orientation_change_rad=float((initial_rotation.inv()*rotation).magnitude())))
                try:
                    next_time = next(wanted)
                except StopIteration:
                    break
    result['real_model_fk'] = dict(scene=str(scene), samples=observed,
        limitations=['Forward kinematics from measured joints, not external position measurement.',
                     'Wrist link origin, not calibrated BrainCo grasp point.',
                     'Physical contact is now uncertain per user; no causal attribution.'])
    result['interpretation_limits'] = [
        'Native simulation uses same ONNX and reference generator, but a different grounded initial state and Dex3 model.',
        'Simulation end errors use median of 15..17 s; real snapshots report their actual times.',
        'No household obstacle or intentional arm contact was added.',
        'No claim of hardware tracking precision or grasp success.']
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, a.output/'g1_compare_sim.py')
    (a.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key != 'simulations'}, indent=2))
    for row in result['simulations']:
        print(json.dumps({key:row[key] for key in ['directory', 'elbow_return_error_rad', 'wrist_base_return_error_m',
            'native_lag_fit_s'] if key in row}))


if __name__ == '__main__':
    main()
