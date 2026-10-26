"""Build a copied controller with atomic streamed cursor adjustment.

Does not edit CoMotion or install/launch any controller. The input controller
is pinned; build inputs, commands and the resulting binary hash are archived.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BASE = Path('/home/user/code/CoMotion/experiments/shared_control/build/controller_v2')
LINK_CWD = Path('/home/user/code/CoMotion/.deps/GR00T-WholeBodyControl/gear_sonic_deploy/build-comotion/src/g1/g1_deploy_onnx_ref')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Native source marker changed: '+old[:80])
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--target',choices=('sim','field'),default='sim')
    a = p.parse_args()
    out = a.output.resolve()
    base=BASE if a.target=='sim' else BASE.parent/'controller_field'
    binary_name='g1_sim' if a.target=='sim' else 'g1_field'
    expected=('sim',77,'lo') if a.target=='sim' else ('field',0,'eno1')
    manifest = json.loads((base/'manifest.json').read_text())
    if (manifest['target'], manifest['dds_domain'], manifest['interface']) != expected:
        raise ValueError('Unexpected controller deployment')
    if sha(base/binary_name) != manifest['binary_sha256']:
        raise ValueError('Base binary changed')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(base/'src',out/'src')
    endpoint = out/'src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_endpoint_interface.hpp'
    old = endpoint.read_text()
    text = once(old,
        'auto result = DecodeIntoMotionSequence(current_frame, streamed_motion_, stream_window_start_, heading_state_buffer);',
        '''int playback_snapshot;
                    {
                        std::lock_guard<std::mutex> motion_lock(current_motion_mutex);
                        playback_snapshot = current_frame;
                    }
                    auto result = DecodeIntoMotionSequence(playback_snapshot, streamed_motion_, stream_window_start_, heading_state_buffer);''')
    text = once(text,
        'int adjusted_frame = current_frame - frame_offset_adjustment;',
        '''// Read and update the cursor under the SAME lock. Otherwise a
                    // control tick between the read and assignment is lost.
                    std::lock_guard<std::mutex> motion_lock(current_motion_mutex);
                    int adjusted_frame = current_frame - frame_offset_adjustment;''')
    text = once(text,
        'std::lock_guard<std::mutex> lock(current_motion_mutex);\n                    current_frame = adjusted_frame;',
        'current_frame = adjusted_frame;')
    endpoint.write_text(text)
    source = out/'src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp'
    text = source.read_text()
    text = once(text,
        '<< ",\\\"reference_revision\\\":" << harness_ref_revision_\n           << ",\\\"locomotion\\\":"',
        '<< ",\\\"reference_revision\\\":" << harness_ref_revision_\n'
        '           << ",\\\"source_start\\\":" << motion.trace_source_start\n'
        '           << ",\\\"source_step\\\":" << motion.trace_source_step\n'
        '           << ",\\\"locomotion\\\":"')
    source.write_text(text)
    commands = [[arg.replace(str(base),str(out)) for arg in cmd] for cmd in manifest['commands']]
    if a.target=='field':
        # The legacy builder links g1_sim, then renames it for field deployment.
        commands[-1]=[str(out/binary_name) if arg==str(out/'g1_sim') else arg for arg in commands[-1]]
    for i,cmd in enumerate(commands):
        with (out/f'build-{i}.log').open('w') as log:
            subprocess.run(cmd,cwd=LINK_CWD,stdout=log,stderr=subprocess.STDOUT,check=True)
    manifest.update(binary_sha256=sha(out/binary_name),original_binary_sha256=sha(base/binary_name),
        commands=commands,link_cwd=str(LINK_CWD),source_sha256=sha(source),stream_cursor_atomic=True,
        endpoint_sha256=sha(endpoint),original_endpoint_sha256=sha(base/endpoint.relative_to(out)),
        build_script_sha256=sha(Path(__file__)),hardware_validated=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps(dict(build=str(out),binary_sha256=manifest['binary_sha256'])))


if __name__ == '__main__':
    main()
