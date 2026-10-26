"""Prepare operator-owned standing startup scripts; never starts robot processes."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.g1_sonic.vendor import sha, WIRE


def prepare(run, reference_hold=False, reach_profile=False, grasp_profile=False,gait_profile=False,native_build=None,ankle_profile=False,thermal_auto_sides=(),ankle_demo_source=None,thermal_rule='strict91',thermal_repeat_seconds=0.,vla=False,locomotion=False):
    if locomotion:ankle_profile=True
    if vla:
        ankle_profile=True;thermal_auto_sides=('left','right');thermal_rule='native90';thermal_repeat_seconds=10.
    if thermal_rule not in ('strict91','native90') or (thermal_repeat_seconds!=0 and not 10<=thermal_repeat_seconds<=600):
        raise ValueError('Invalid thermal rule/repeat interval')
    if not ankle_profile and (ankle_demo_source or thermal_rule!='strict91' or thermal_repeat_seconds):
        raise ValueError('Thermal settings require ankle profile')
    thermal_auto_sides=sorted(set(thermal_auto_sides))
    if thermal_auto_sides and (not ankle_profile or not set(thermal_auto_sides)<={'left','right'}):
        raise ValueError('Automatic ankle sides require the ankle profile and valid sides')
    if ankle_profile and any((reference_hold,reach_profile,grasp_profile,gait_profile)):
        raise ValueError('Ankle demo owner is exclusive of other reference profiles')
    if reach_profile and not reference_hold:
        raise ValueError('Reach profile requires reference hold')
    if grasp_profile and not reach_profile:
        raise ValueError('Grasp profile requires reach profile')
    root = Path(__file__).resolve().parents[1]
    comotion = Path('/home/user/code/CoMotion')
    exp = comotion/'experiments/shared_control'
    build = Path(native_build).resolve() if native_build else exp/'build/controller_field'
    assets = exp/'assets/local_release_20261004'
    manifest = json.loads((build/'manifest.json').read_text())
    binary = build/'g1_field'
    if (manifest['dds_domain'], manifest['interface'], manifest['transport_lease_ms']) != (0, 'eno1', 350):
        raise ValueError('Unexpected native controller deployment')
    if sha(binary) != manifest['binary_sha256']:
        raise ValueError('Native controller binary hash mismatch')
    if gait_profile and (not reach_profile or not manifest.get('stream_cursor_atomic')):
        raise ValueError('Gait requires reach profile and validated atomic reference cursor build')
    run.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root/'services/g1_sonic', run/'runtime_source/g1_sonic',
                    ignore=shutil.ignore_patterns('__pycache__'))
    if ankle_profile:
        source=Path(ankle_demo_source).resolve() if ankle_demo_source else root/'services/g1_sonic/data/ankle_relief'
        dest=run/'ankle_demos';dest.mkdir();demo_manifest={}
        packaged=json.loads((source/'manifest.json').read_text()) if (source/'manifest.json').exists() else None
        for side,episode in [('left',13),('right',15)]:
            name=f'episode_{episode:06d}'
            if packaged is not None:
                if sha(source/(name+'.npz'))!=packaged[side]['sha256']:
                    raise ValueError('Packaged ankle demonstration changed')
                shutil.copy2(source/(name+'.npz'),dest/(name+'.npz'))
                demo_manifest[side]=packaged[side]
                continue
            metadata=json.loads((source/(name+'.json')).read_text())
            if sha(metadata['source'])!=metadata['sha256']:raise ValueError('Original ankle demonstration changed')
            for suffix in ('.npz','.json'):shutil.copy2(source/(name+suffix),dest/(name+suffix))
            demo_manifest[side]=dict(episode=episode,sha256=sha(dest/(name+'.npz')),
                source_parquet_sha256=metadata['sha256'],source=metadata['source'],candidate=metadata.get('candidate'))
        (dest/'manifest.json').write_text(json.dumps(demo_manifest,indent=2)+'\n')
    if gait_profile:
        from services.g1_sonic.gait import SOURCE
        dest=run/'runtime_source/g1_sonic/gait_data';dest.mkdir()
        for name in ('joint_pos.csv','body_quat.csv'):
            shutil.copy2(SOURCE/name,dest/name)
    shutil.copy2(Path(__file__), run/'runtime_source/g1_prepare_native.py')
    shutil.copy2(root/'scripts/g1_manage_terminals.py', run/'runtime_source/g1_manage_terminals.py')
    command = [str(binary), 'eno1', str(assets/'model_decoder.onnx'),
        str(comotion/'.deps/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example'),
        '--obs-config', str(assets/'observation_config.yaml'), '--encoder-file', str(assets/'model_encoder.onnx'),
        '--planner-file', str(assets/'planner/target_vel/V2/planner_sonic.onnx'),
        '--input-type', 'zmq_manager', '--output-type', 'zmq', '--zmq-host', '127.0.0.1',
        '--zmq-port', '17586', '--zmq-out-port', '17587', '--logs-dir', str(run/'controller_logs')]
    native = ['env', 'HARNESS_TRACE='+str(run/'controller_trace.jsonl'), *command]
    script = '\n'.join(['#!/usr/bin/env bash', 'set -euo pipefail',
        'cd '+shlex.quote(str(comotion)), 'source ./env.local.sh',
        'unset SONIC_AUTO_START',
        'exec flock -n /tmp/comotion_field_native.lock script -q -e -f -c '+shlex.quote(shlex.join(native))+' '+shlex.quote(str(run/'controller.log')), ''])
    (run/'controller.sh').write_text(script)
    python = '/home/user/GR00T-WholeBodyControl/.venv_data_collection/bin/python'
    if ankle_profile:python='/home/user/GR00T-WholeBodyControl/.venv_teleop/bin/python'
    if vla:python='/home/user/GR00T-WholeBodyControl/.venv_inference/bin/python'
    module = 'g1_sonic.ankle_executor' if ankle_profile else 'g1_sonic.executor' if reference_hold else 'g1_sonic.commission'
    owner = ['env', 'PYTHONPATH='+str(run/'runtime_source'), python, '-m', module,
             '--run', str(run), '--start-policy']
    if vla:owner.append('--vla')
    if locomotion:owner.append('--locomotion')
    if reference_hold:
        owner.append('--reference-hold')
    if reach_profile:
        owner.append('--reach-profile')
    if grasp_profile:
        owner.append('--grasp-profile')
    if gait_profile:
        owner.append('--gait-profile')
    if thermal_auto_sides:
        owner.extend(['--thermal-auto-sides',*thermal_auto_sides])
    if ankle_profile:
        owner.extend(['--lease-seconds','0'])
        owner.extend(['--thermal-rule',thermal_rule,'--thermal-repeat-seconds',str(thermal_repeat_seconds)])
    (run/'owner.sh').write_text('\n'.join(['#!/usr/bin/env bash', 'set -euo pipefail',
        'cd '+shlex.quote(str(root)),
        shlex.join(['python3', str(run/'runtime_source/g1_manage_terminals.py'),
                    '--runs-root',str(root/'runtime/g1'),'--keep-run',str(run),'--apply']),
        'exec '+shlex.join(owner), '']))
    if ankle_profile:
        watch=['env','PYTHONPATH='+str(run/'runtime_source'),'/home/user/GR00T-WholeBodyControl/.venv_teleop/bin/python','-m','g1_sonic.temperature_watch',
               '--run',str(run),'--output',str(run/'temperature_dashboard.jsonl')]
        (run/'temperature-watch.sh').write_text('\n'.join(['#!/usr/bin/env bash','set -euo pipefail',
            'cd '+shlex.quote(str(root)),'exec '+shlex.join(watch),'']))
        for name,options in [('test-right',['test','--side','right']),('test-left',['test','--side','left']),
                             ('arm-right',['arm','--sides','right']),('arm-both',['arm','--sides','left','right']),('disarm',['disarm'])]:
            command_request=['env','PYTHONPATH='+str(run/'runtime_source'),python,'-m','g1_sonic.ankle_request','--run',str(run),*options]
            (run/(name+'.sh')).write_text('\n'.join(['#!/usr/bin/env bash','set -euo pipefail',
                'cd '+shlex.quote(str(root)),'exec '+shlex.join(command_request),'']))
    thermal = ['env','PYTHONPATH='+str(run/'runtime_source'),
        '/home/user/GR00T-WholeBodyControl/.venv_teleop/bin/python','-m','g1_sonic.ankle_monitor',
        '--run',str(run/'thermal_observer'),'--seconds','600']
    (run/'thermal-observer.sh').write_text('\n'.join(['#!/usr/bin/env bash','set -euo pipefail',
        'cd '+shlex.quote(str(root)),'exec '+shlex.join(thermal),'']))
    if vla or locomotion:
        backend_options=dict(control_directory=str(run),trace_path=str(run/'controller_trace.jsonl'),
                             camera_directory=str(run/'camera'))
        (run/'backend-options.json').write_text(json.dumps(backend_options,indent=2)+'\n')
        # The UI/MCP configuration belongs to the operator. This wrapper starts
        # only task dispatch, never another frontend or hardware controller.
        agent=['/home/user/GR00T-WholeBodyControl/.venv_data_collection/bin/python',
               '-m','harness.codex_cli.ui_agent','--config',str(run/'operator-input.json')]
        (run/'ui-agent.sh').write_text('\n'.join(['#!/usr/bin/env bash','set -euo pipefail',
            'cd '+shlex.quote(str(root)), 'export PYTHONPATH='+shlex.quote(str(root)),
            'exec '+shlex.join(agent)+' --socket "${RAX_MCP_SOCKET:?Set the existing frontend socket}"','']))
    if vla:
        vla_dependencies=[Path('/home/user/code/Isaac-GR00T-rtc')/name for name in (
            'gr00t/policy/rtc_client.py','gr00t/policy/server_client.py',
            'scripts/local/rtc_robot_session.py','scripts/local/run_brainco_rtc_inference.py')]
        vla_dependencies += [Path('/home/user/Isaac-GR00T/scripts/local')/name for name in (
            'brainco_policy_bridge.py','brainco_model_adapter.py','run_brainco_inference.py')]
    else:vla_dependencies=[]
    (run/'manifest.json').write_text(json.dumps(dict(controller=command, owner=owner,
        thermal_observer=thermal,thermal_motion_enabled=bool(thermal_auto_sides),thermal_motion_available=ankle_profile,
        thermal_initial_armed_sides=thermal_auto_sides,
        thermal_trigger_rule='native alarm: enter >=90 C, clear each motor <85 C' if thermal_rule=='native90' else 'max(winding,driver) > 91 C',
        thermal_rule=thermal_rule,thermal_repeat_seconds=thermal_repeat_seconds,
        ankle_demo_source=str(source) if ankle_profile else None,
        binary_sha256=manifest['binary_sha256'], hardware_validated=False,
        owner_source_sha256={str(path.relative_to(run/'runtime_source')): sha(path)
                             for path in sorted((run/'runtime_source').rglob('*')) if path.is_file()},
        external_python_sha256={str(path): sha(path) for path in
                                (WIRE, exp/'shared_control/reference.py')},
        scope='planner_locomotion_upper_hold_v1' if locomotion else 'original_pico_ankle_demo_v1' if ankle_profile else 'grasp_step_v1' if grasp_profile else 'clear_reach_v1' if reach_profile else 'bounded elbow reference' if reference_hold else 'planner idle',
        walking=gait_profile or locomotion,native_build=str(build),stream_cursor_atomic=manifest.get('stream_cursor_atomic',False),
        session_lease_unlimited=ankle_profile,session_lease_seconds=0 if ankle_profile else None,
        locomotion_enabled=locomotion,vla_enabled=vla,vla_dependencies_sha256={str(path):sha(path) for path in vla_dependencies},
        hand_commands_from_rax=vla), indent=2)+'\n')
    print(run)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reference-hold', action='store_true')
    p.add_argument('--reach-profile', action='store_true')
    p.add_argument('--grasp-profile',action='store_true')
    p.add_argument('--gait-profile',action='store_true')
    p.add_argument('--native-build',type=Path)
    p.add_argument('--vla',action='store_true',help='Agent RTC toolbox plus default bilateral ankle relief; one body owner')
    p.add_argument('--agent-tools',action='store_true',help='Enable all current agent tools: VLA, left/right/both hand control and locomotion in one owner')
    p.add_argument('--locomotion',action='store_true',help='Enable g1_move in the same owner, with upper-body posture hold')
    p.add_argument('--ankle-profile',action='store_true',help='Exclusive original Pico token owner')
    p.add_argument('--thermal-auto-sides',choices=['left','right'],nargs='+',default=[],help='Enable corresponding-side automatic temperature actions at startup')
    p.add_argument('--ankle-demo-source',type=Path,help='Explicit original or validated delta candidate directory')
    p.add_argument('--thermal-rule',choices=['strict91','native90'],default='strict91')
    p.add_argument('--thermal-repeat-seconds',type=float,default=0.)
    args = p.parse_args()
    prepare(args.run.resolve(), args.reference_hold, args.reach_profile,args.grasp_profile,args.gait_profile,args.native_build,args.ankle_profile,args.thermal_auto_sides,args.ankle_demo_source,args.thermal_rule,args.thermal_repeat_seconds,args.vla or args.agent_tools,args.locomotion or args.agent_tools)
