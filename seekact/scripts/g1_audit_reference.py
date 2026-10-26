"""Offline audit of one recorded reference job. Never opens robot/network interfaces."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.g1_sonic.reference import pulse_envelope
from services.g1_sonic.state import NAMES, HARDWARE_TO_ISAAC


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executor', required=True, type=Path)
    p.add_argument('--job-id', required=True)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    events = [json.loads(line) for line in (a.executor/'executor_events.jsonl').read_text().splitlines()]
    start = next(e for e in events if e['event'] == 'job_started' and e['id'] == a.job_id)
    finish = next(e for e in events if e['event'] == 'job_finished' and e['id'] == a.job_id)
    fault = next((e for e in events if e['event'] == 'fault_idle' and e['monotonic'] > start['monotonic']), None)
    # Do not mix fallback planner motion into the return assessment.
    end = min(start['monotonic']+17, fault['monotonic'] if fault else float('inf'))
    rows = []
    with (a.executor/'controller_trace.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if start['monotonic']-2 <= row['monotonic'] < end and row['motion'] == 'streamed':
                rows.append(row)
            if row['monotonic'] >= end:
                break
    baseline = next(r for r in reversed(rows) if r['monotonic'] < start['monotonic'])
    times = np.array([r['monotonic']-start['monotonic'] for r in rows])
    q = np.array([r['body_q'] for r in rows])[:, HARDWARE_TO_ISAAC]
    targets = np.array([r['q_target'] for r in rows])[:, HARDWARE_TO_ISAAC]
    encoder = np.array([r['encoder_obs'] for r in rows])
    refq = encoder[:, 4:294].reshape(-1, 10, 29)
    refdq = encoder[:, 294:584].reshape(-1, 10, 29)
    base_obs = np.asarray(baseline['encoder_obs'])
    base_q = np.asarray(baseline['body_q'])[HARDWARE_TO_ISAAC]
    base_target = np.asarray(baseline['q_target'])[HARDWARE_TO_ISAAC]
    i = NAMES.index(start['side']+'_elbow_joint')
    base_ref = base_obs[4:294].reshape(10, 29)
    base_dq = base_obs[294:584].reshape(10, 29)
    mask = (times >= 0) & (times <= 11)
    lags = np.arange(0, 4.001, .02)
    errors = [float(np.sqrt(np.mean((refq[mask, 0, i]-base_ref[0, i]-
               start['delta_rad']*pulse_envelope(times[mask]-lag)[0])**2))) for lag in lags]
    lag_index = int(np.argmin(errors))
    returned = ((times >= 7) & (np.max(abs(refq-base_ref), axis=(1, 2)) < 1e-4) &
                (np.max(abs(refdq-base_dq), axis=(1, 2)) < 1e-4))
    returned_indices = np.flatnonzero(returned)
    other = [j for j in range(29) if j != i]
    report = dict(job_id=a.job_id, source_executor=str(a.executor.resolve()),
        assessment='transport_verified_physical_return_failed' if finish.get('return_error_rad', 0) > .05 else 'inspect_results',
        source_result_unmodified=True, sender_finished_elapsed_s=finish['monotonic']-start['monotonic'],
        reference_delta_rad=start['delta_rad'],
        native_current_reference_lag_fit_s=float(lags[lag_index]),
        lag_fit_rmse_rad=errors[lag_index], lag_fit_is_observation_not_root_cause=True,
        native_full_q_dq_return_first_elapsed_s=float(times[returned_indices[0]]) if len(returned_indices) else None,
        max_other_joint_reference_change_rad=float(np.max(abs(refq[:, :, other]-base_ref[:, other]))),
        native_fault=fault,
        joint_changes={name: dict(
            peak_measured_change_rad=float(np.max(abs(q[times >= 0, j]-base_q[j]))),
            final_measured_change_rad=float(q[-1, j]-base_q[j]),
            final_native_motor_target_change_rad=float(targets[-1, j]-base_target[j]))
            for j, name in enumerate(NAMES) if any(part in name for part in ('elbow', 'wrist', 'shoulder'))},
        samples=[])
    for target_time in [-1, 3, 5, 7, 10, 12]:
        k = int(np.argmin(abs(times-target_time)))
        report['samples'].append(dict(elapsed_s=float(times[k]),
            native_elbow_reference_rad=float(refq[k, 0, i]), measured_elbow_rad=float(q[k, i]),
            native_elbow_motor_target_rad=float(targets[k, i]),
            full_encoder_max_change=float(np.max(abs(encoder[k]-base_obs)))))
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, a.output/'g1_audit_reference.py')
    (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (a.output/'elbow_signals.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['elapsed_s', 'native_reference_rad', 'measured_rad', 'native_motor_target_rad'])
        writer.writerows(zip(times, refq[:, 0, i], q[:, i], targets[:, i]))
    print(json.dumps({key: report[key] for key in ('assessment', 'native_current_reference_lag_fit_s',
        'lag_fit_rmse_rad', 'native_full_q_dq_return_first_elapsed_s')}, indent=2))


if __name__ == '__main__':
    main()
