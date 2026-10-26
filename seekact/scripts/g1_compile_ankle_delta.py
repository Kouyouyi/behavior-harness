"""Compile offline leg-only delta candidates using the installed CoMotion bridge."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/home/user/code/CoMotion')
EXP = PROJECT / 'experiments/shared_control'
sys.path[:0] = [str(ROOT), str(EXP)]
from services.g1_sonic.ankle_delta import amplify_windows
from services.g1_sonic.vendor import sha
from shared_control.smpl_bridge import SmplBridge
from shared_control.vla_arm_overlay import LiveTokenReference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gains', type=float, nargs='+', default=[1.5, 2.])
    a = p.parse_args()
    if any(not np.isfinite(g) or not 1 <= g <= 2 for g in a.gains):
        p.error('gains must be finite in [1,2]')
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, a.output / Path(__file__).name)
    shutil.copy2(ROOT/'services/g1_sonic/ankle_delta.py', a.output/'ankle_delta.py')
    assets = EXP/'assets/local_release_20261004'
    kin = EXP/'assets/kinematic_decoder_v01'
    bridge = SmplBridge(assets, kin,
        Path('/home/user/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'),
        PROJECT/'deployment/config/g1_reference_limits.json')
    fixture_path = EXP/'fixtures/smpl_standing_physical_v02.npz'
    with np.load(fixture_path) as f:
        reference = LiveTokenReference(bridge, {k:f[k].copy() for k in f.files})
    reports = []
    for side, episode in [('left', 13), ('right', 15)]:
        source = a.source/f'episode_{episode:06d}.npz'
        meta = json.loads(source.with_suffix('.json').read_text())
        with np.load(source, allow_pickle=False) as f:
            data = {k:f[k].copy() for k in f.files}
        tokens = data['action.motion_token'].astype(np.float32)
        poses, orientations, zeros = [], [], []
        for token in tokens:
            pose, orientation = reference.reference(token)
            zero, _ = bridge.encode(pose, orientation)
            poses.append(pose);orientations.append(orientation);zeros.append(zero)
        poses, orientations, zeros = map(np.asarray, (poses, orientations, zeros))
        for gain in a.gains:
            changed, deltas = amplify_windows(poses, side, gain)
            modified = tokens.copy()
            for i in range(len(tokens)):
                if np.any(deltas[i]):
                    encoded, _ = bridge.encode(changed[i], orientations[i])
                    modified[i] = tokens[i] + (encoded - zeros[i])
            assert np.array_equal(modified[0], tokens[0])
            if gain == 1:
                assert np.array_equal(modified, tokens)
            destination = a.output/f'{side}-gain-{gain:g}'
            destination.mkdir()
            payload = dict(data)
            payload['source.action.motion_token'] = tokens
            payload['action.motion_token'] = modified
            out = destination/source.name
            np.savez_compressed(out, **payload)
            np.savez_compressed(destination/'delta_evidence.npz', delta=deltas,
                reference_pose=poses, modified_pose=changed, orientation=orientations)
            decoded = bridge.kin(modified)['q'][:, 0]
            baseline = bridge.kin(tokens)['q'][:, 0]
            report = dict(side=side, gain=gain, source_npz_sha256=sha(source),
                generated_npz_sha256=sha(out), source_frames=len(tokens), first_token_exact=True,
                max_extra_local_angle_rad=float(np.linalg.norm(deltas, axis=-1).max()),
                max_decoded_joint_change_rad=float(np.max(abs(decoded-baseline))),
                max_token_component_change=float(np.max(abs(modified-tokens))),
                max_extra_angle_rad=.35, extra_rate_rad_s=.6,
                formula='z_new = z_original + Encoder(R_ref @ Exp(delta)) - Encoder(R_ref)',
                reference='CoMotion calibrated LiveTokenReference, per-source-token 10-frame window',
                hardware_validated=False, simulation_validated=False,
                note='Only chosen leg SMPL rotations modified; robot coupling remains. Original measured q arrays are historical, not predictions.')
            out.with_suffix('.json').write_text(json.dumps(dict(meta, candidate=report), indent=2)+'\n')
            (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            reports.append(report)
            print(json.dumps(report), flush=True)
    (a.output/'manifest.json').write_text(json.dumps(dict(candidates=reports,
        dependencies_sha256={str(p):sha(p) for p in [assets/'model_encoder.onnx', assets/'model_decoder.onnx',
        kin/'manifest.json', fixture_path, EXP/'shared_control/smpl_bridge.py', EXP/'shared_control/vla_arm_overlay.py']}), indent=2)+'\n')


if __name__ == '__main__':
    main()
