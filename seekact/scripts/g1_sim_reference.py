"""Native SONIC / MuJoCo reference diagnostic. Fixed DDS 77 + lo; never hardware.

Reuses the pinned CoMotion simulator and simulation-only native controller. The
production RAX pulse generator is unchanged. This is not the real executor: its
feedback comes from MuJoCo, and all output is kept in a new experiment directory.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pty
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
COMOTION = Path('/home/user/code/CoMotion')
EXP = COMOTION/'experiments/shared_control'
SONIC = Path('/home/user/GR00T-WholeBodyControl')
sys.path[:0] = [str(ROOT), str(EXP)]
from services.g1_sonic.reference import pulse_window, hold_window
from services.g1_sonic.state import NAMES
from services.g1_sonic.vendor import wire_module, sha
from shared_control.reference import from_idle_observation
from shared_control.initialization import ReadinessWindow
from services.g1_sonic.sim_reach import Geometry, Transition, run_reach
from services.g1_sonic.walk import pulse as walk_pulse, upper_fields, validate_walk, RAMP_SECONDS
from services.g1_sonic.reach import Kinematics
from services.g1_sonic.gait import make as make_gait,fields as gait_fields
from services.g1_sonic.locomotion import LocomotionJobs, MOTIONS, submit as move_submit, result as move_result, yaw as body_yaw
from services.g1_sonic.walk import UPPER_IDS
from services.g1_sonic.state import HARDWARE_TO_ISAAC


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def tail(path):
    if not path.exists():
        return None
    with path.open('rb') as f:
        size = f.seek(0, 2)
        offset = max(0, size-262144)
        f.seek(offset)
        raw = f.read()
    if offset:
        raw = raw.partition(b'\n')[2]
    rows = raw.split(b'\n')[:-1]
    return json.loads(rows[-1]) if rows else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--case', required=True, choices=['zero', 'pulse', 'reach','walk','gait','heel','recorded_token','locomotion'])
    p.add_argument('--directions',nargs='+',choices=MOTIONS,default=list(MOTIONS))
    p.add_argument('--move-speed',type=float,default=.3)
    p.add_argument('--move-duration',type=float,default=2.)
    p.add_argument('--move-mode',choices=['slowWalk','walk'],default='walk')
    p.add_argument('--carry-pose',action='store_true',help='Simulation: raise wrists into a bent-arm carrying-like pose, no payload')
    p.add_argument('--recorded-episode',type=Path,help='Offline exported NPZ; simulation-only token replay')
    p.add_argument('--heel-side',choices=['left','right'],default='left')
    p.add_argument('--heel-angle',type=float,default=.20)
    p.add_argument('--heel-shift',type=float,default=.035)
    p.add_argument('--foot-lift-m',type=float,default=0.,help='Simulation only: whole-foot lift instead of toe pivot')
    p.add_argument('--gait-amplitude',type=float,default=.5)
    p.add_argument('--gait-mirrored-pair',action='store_true')
    p.add_argument('--sim-build', type=Path, default=EXP/'build/controller_v2')
    p.add_argument('--walk-speed',type=float,default=.1)
    p.add_argument('--walk-plateau-seconds',type=float,default=.5)
    p.add_argument('--hold-seconds', type=float, default=6.)
    p.add_argument('--disable-hand-collisions', action='store_true', help='Simulation diagnostic only; retain hand inertia')
    p.add_argument('--disable-arm-collisions', action='store_true', help='Simulation diagnostic only; retain all upper-limb inertia')
    args = p.parse_args()
    validate_walk(args.walk_speed,args.walk_plateau_seconds)
    if args.case in ('reach','walk','gait','heel','recorded_token','locomotion') and (args.disable_hand_collisions or args.disable_arm_collisions):
        p.error('Reach validation requires normal collisions')
    demo=None;demo_candidate=None
    if args.case=='recorded_token':
        if args.recorded_episode is None:p.error('recorded_token requires --recorded-episode')
        demo_candidate=json.loads(args.recorded_episode.with_suffix('.json').read_text()).get('candidate')
        with np.load(args.recorded_episode,allow_pickle=False) as f:
            demo=f['action.motion_token'].astype(np.float32)
            demo_times=f['timestamp'].reshape(-1)
            if not np.all(f['teleop.stream_mode']==1):raise ValueError('Expected SMPL demonstration')
        if demo.ndim!=2 or demo.shape[1]!=64 or len(demo)<2 or not np.isfinite(demo).all():
            raise ValueError('Invalid recorded token sequence')
        if not np.allclose(np.diff(demo_times),.02,atol=1e-5):raise ValueError('Expected 50 Hz episode')
    if not np.isfinite(args.hold_seconds) or not 6 <= args.hold_seconds <= 60:
        p.error('hold-seconds must be 6..60')
    run = args.run.resolve()
    build = args.sim_build.resolve()
    manifest = json.loads((build/'manifest.json').read_text())
    if (manifest['dds_domain'], manifest['interface'], manifest['target']) != (77, 'lo', 'sim'):
        raise ValueError('Not an isolated simulation controller')
    binary = build/'g1_sim'
    if sha(binary) != manifest['binary_sha256']:
        raise ValueError('Simulation binary hash mismatch')
    lock = open('/tmp/comotion_dds77.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run.mkdir(parents=True, exist_ok=False)
    moves=None
    if args.case=='locomotion':
        (run/'movement').mkdir()
        moves=LocomotionJobs(run/'movement','simulation')
    if demo is not None:
        shutil.copy2(args.recorded_episode,run/'recorded_episode.npz')
        shutil.copy2(args.recorded_episode.with_suffix('.json'),run/'recorded_episode.json')
    shutil.copy2(__file__, run/'g1_sim_reference.py')
    shutil.copy2(ROOT/'scripts/g1_sim_scene.py', run/'g1_sim_scene.py')
    shutil.copy2(ROOT/'scripts/g1_sim_heel_scene.py', run/'g1_sim_heel_scene.py')
    shutil.copytree(ROOT/'services/g1_sonic', run/'reference_source', ignore=shutil.ignore_patterns('__pycache__'))
    config = json.loads((EXP/'configs/pico_live_v01.json').read_text())['initialization']
    dump(run/'initialization.json', config)
    # Isolate writable TensorRT caches; ONNX weights remain read-only references.
    original_assets = EXP/'assets/local_release_20261004'
    assets = run/'assets'
    def copy_asset(source, dest):
        if str(source).endswith('.onnx'):
            Path(dest).symlink_to(Path(source).resolve())
        else:
            shutil.copy2(source, dest)
    shutil.copytree(original_assets, assets, copy_function=copy_asset)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MUJOCO_GL='egl',
        COMOTION_SONIC_ROOT=str(SONIC), HARNESS_BLEND_SECONDS='0.6',
        HARNESS_TRACE=str(run/'controller_trace.jsonl'),
        PYTHONPATH=os.pathsep.join([str(EXP), str(SONIC), str(SONIC/'external_dependencies/unitree_sdk2_python')]),
        XDG_CACHE_HOME=str(run/'cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.pop('SONIC_AUTO_START', None)
    env.pop('SONIC_INFERENCE_BACKEND', None)
    env['LD_LIBRARY_PATH'] = os.pathsep.join(map(str, [EXP/'vendor/onnxruntime-linux-x64-1.16.3/lib',
        SONIC/'TensorRT/lib', Path('/home/user/code/.toolchains/cuda-12.8/lib64'),
        Path('/home/user/code/.toolchains/cuda-12.8/lib'),
        SONIC/'gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64']))
    env['CYCLONEDDS_HOME'] = str(EXP/'vendor/cyclonedds')
    context = zmq.Context()
    pub = context.socket(zmq.PUB); pub.setsockopt(zmq.LINGER, 0)
    pub.bind('tcp://127.0.0.1:17576')
    wire = wire_module()
    children, handles = [], []
    master, slave = pty.openpty()
    events = (run/'events.jsonl').open('w')
    samples = (run/'samples.jsonl').open('w')
    state = trace = reference = None
    active = False
    frame = 0
    job_start = None
    walk_start=None; walk_upper=None; walk_records=[]; gait=None
    external_token=None; demo_phase=None; demo_index=None
    readiness = ReadinessWindow(config)
    records = []
    transition = None
    geometry = Geometry() if args.case in ('reach','walk','gait','heel','recorded_token','locomotion') else None
    reach_solver = Geometry() if args.case == 'reach' else None
    # qpos includes Dex3 joints interleaved with body joints; use actual model IDs.
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(SONIC/'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'))
    qids = [int(model.jnt_qposadr[model.joint(name).id]) for name in NAMES]
    vids = [int(model.jnt_dofadr[model.joint(name).id]) for name in NAMES]

    def event(kind, **values):
        row = dict(event=kind, monotonic=time.monotonic(), sim_time=state['sim_time'] if state else None, **values)
        events.write(json.dumps(row)+'\n'); events.flush()
        print(json.dumps(row), flush=True)

    def launch(command, name, cwd, stdin=None):
        handle = (run/(name+'.log')).open('w'); handles.append(handle)
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=stdin, stdout=handle,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        event('process_started', name=name, pid=child.pid, command=list(map(str, command)))

    def tick():
        nonlocal state, trace, frame
        tick_start = time.monotonic()
        for child in children:
            if child.poll() is not None:
                raise RuntimeError(f'Child {child.pid} exited {child.returncode}')
        path = run/'scene/latest.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state['dropped_records'] or state['io_errors'] or state['initialization']['failure']:
                raise RuntimeError('Simulator recording or initialization failure')
            readiness.add(state)
            if state['fell'] or np.linalg.norm(state['root_qpos'][:2]) > (3. if moves else .35):
                raise RuntimeError('Simulation fall or excessive root excursion')
            if active and time.monotonic()-state['monotonic'] > .3:
                raise RuntimeError('Stale simulated state')
        trace = tail(run/'controller_trace.jsonl')
        if trace and (trace['transport_fault'] or trace['log_dropped']):
            raise RuntimeError('Native simulator transport/log fault')
        if active:
            if external_token is not None:
                pub.send(wire.pack_pose_message(dict(token_state=external_token[None].astype(np.float32),
                    frame_index=np.array([frame],dtype=np.int64)),version=4));frame+=1
            elif reference is None:
                if moves and walk_upper and trace:
                    native=dict(trace,base_quat=state['root_qpos'][3:7],
                        body_q=np.asarray(state['qpos'])[qids][np.argsort(HARDWARE_TO_ISAAC)].tolist())
                    settled=max(abs(np.asarray(state['qvel'])[vids]))<.3 and np.linalg.norm(state['root_qvel'][3:6])<.25
                    command=moves.tick(time.monotonic(),native,settled,True,600)
                    pub.send(moves.packet(wire,command or moves.idle_command))
                    dump(run/'movement/ankle_status.json',dict(phase='ready',fault=None,session_id='simulation',
                        updated_monotonic=time.monotonic(),lease_remaining_seconds=600,locomotion=moves.snapshot()))
                else:
                    command=walk_pulse(time.monotonic()-walk_start,args.walk_speed,args.walk_plateau_seconds) if walk_start else walk_pulse(-1)
                    pub.send(wire.build_planner_message(**command,**(walk_upper or {})))
            else:
                elapsed = max(0., time.monotonic()-job_start) if job_start else 0.
                fields = (gait_fields(gait,time.monotonic()-walk_start,frame) if gait is not None and walk_start else
                          transition.fields(time.monotonic(), frame) if transition else
                          pulse_window(reference, 'right', .03, elapsed, frame)
                          if job_start and args.case == 'pulse' else hold_window(reference, frame))
                pub.send(wire.pack_pose_message(fields, version=1)); frame += 1
            pub.send(wire.build_command_message(start=True, stop=False, planner=reference is None and external_token is None))
        if walk_start and state and trace:
            geom=geometry.measure(state)
            row=dict(elapsed_s=time.monotonic()-walk_start,sim_time=state['sim_time'],
                     root_xyz=state['root_qpos'][:3],root_quat=state['root_qpos'][3:7],root_velocity=state['root_qvel'][:6],
                     feet=state['feet'],loads=state['foot_ground_normal_force_n'],
                     body_q=np.asarray(state['qpos'])[qids].tolist(),body_dq=np.asarray(state['qvel'])[vids].tolist(),
                     geometry=geom,locomotion=trace['locomotion'],speed=trace['speed'],native_motion=trace['motion'],
                     support_wrench=state['support_wrench'])
            if args.case in ('heel','recorded_token'): row['ankle_telemetry']=state['ankle_telemetry']
            if moves:
                target=np.asarray(moves.upper['upper_body_position'])
                row.update(move_phase=moves.phase,move_job=moves.job,
                    upper_target=target.tolist(),
                    upper_measured_error_rad=float(np.max(abs(np.asarray(row['body_q'])[UPPER_IDS]-target))),
                    upper_encoder_error_rad=float(np.max(abs(np.asarray(trace['encoder_obs'])[4:294].reshape(10,29)[:,UPPER_IDS]-target))))
            if args.case=='recorded_token':
                row.update(phase=demo_phase,source_frame=demo_index,
                    submitted_token=external_token.tolist() if external_token is not None else None,
                    native_token=trace['token'])
                if 2*np.arccos(min(1.,abs(state['root_qpos'][3])))>.5:
                    raise RuntimeError('Recorded-token simulation excessive base rotation')
            walk_records.append(row);samples.write(json.dumps(row)+'\n');samples.flush()
        if job_start and state and trace and reference is not None:
            quat = np.array(state['root_qpos'][3:7])
            base = Rotation.from_quat(quat[[1, 2, 3, 0]])
            wrist = base.inv().apply(np.array(state['wrists']['right'])-state['root_qpos'][:3])
            row = dict(elapsed_s=time.monotonic()-job_start, sim_time=state['sim_time'],
                q=np.asarray(state['qpos'])[qids].tolist(), dq=np.asarray(state['qvel'])[vids].tolist(),
                native_q=np.asarray(trace['encoder_obs'])[4:294].reshape(10, 29)[0].tolist(),
                native_future_qdq_max_error=float(np.max(abs(np.asarray(trace['encoder_obs'])[4:584]-
                    np.r_[np.tile(reference['joint_pos'][0], 10), np.tile(reference['joint_vel'][0], 10)]))),
                wrist_world=state['wrists']['right'], wrist_base=wrist.tolist(),
                root_tilt_rad=float(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2), -1, 1))),
                root_xyz=state['root_qpos'][:3], motion=trace['motion'])
            if transition:
                target = np.r_[np.tile(transition.target, 10), np.tile(reference['joint_vel'][0], 10)]
                row.update(phase=transition.phase, geometry=geometry.measure(state),
                    native_target_qdq_max_error=float(np.max(abs(np.asarray(trace['encoder_obs'])[4:584]-target))))
            records.append(row); samples.write(json.dumps(row)+'\n'); samples.flush()
        time.sleep(max(0., tick_start+.02-time.monotonic()))

    def wait(predicate, timeout, label):
        deadline = time.monotonic()+timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise RuntimeError('Timeout: '+label)
            tick()
        event(label)

    def hold(seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            tick()

    dump(run/'manifest.json', dict(case=args.case, dds_domain=77, interface='lo', hardware_commands=False,
        binary_sha256=sha(binary), source_sha256=sha(__file__),
        sim_build=str(build), stream_cursor_atomic=manifest.get('stream_cursor_atomic',False),
        encoder_sha256=sha(original_assets/'model_encoder.onnx'), decoder_sha256=sha(original_assets/'model_decoder.onnx'),
        initial_state='existing grounded simulation reset; not exact real robot state',
        scene='stock G1 43 DoF with Dex3; no household obstacles, not BrainCo inertial calibration',
        hand_collision_ablation=args.disable_hand_collisions,
        arm_collision_ablation=args.disable_arm_collisions,
        reference=('shared g1_move job state machine; native planner lower body with captured 17-joint upper-body hold' if moves else 'recorded full-body SONIC token sequence; synthesized entry/return; simulation only' if args.case=='recorded_token' else 'simulation-only toe pivot with support-side weight shift, then return' if args.case=='heel' else 'native walking clip leg/root-tilt reference, cleared upper body, then standing reference'
                   if args.case=='gait' else 'clear arms, planner upper-body hold, bounded slow-walk pulse, idle and measured stopping'
                   if args.case=='walk' else 'normal-collision clear pose then measured-relative 0.10 m forward + 0.04 m upward reach and return; four-second quintic transitions'
                   if args.case == 'reach' else 'same RAX pulse_window, 0.03 rad right elbow, 7 seconds; observe 17 seconds'),
        handoff_hold_s=args.hold_seconds, simulated_transport='native v1 ZMQ sliding windows; same source as field apart from deployment/trace',
        locomotion_settings=dict(directions=args.directions,speed=args.move_speed,duration_s=args.move_duration,
            mode=args.move_mode,carry_pose=args.carry_pose) if moves else None,
        heel_side=args.heel_side if args.case=='heel' else None,
        recorded_episode_sha256=sha(args.recorded_episode) if demo is not None else None,
        demo_candidate=demo_candidate,
        recorded_episode=str(args.recorded_episode) if demo is not None else None,
        audio_gate_required=args.case in ('heel','recorded_token'),
        simulator_source_sha256=sha(ROOT/'scripts/g1_sim_heel_scene.py' if args.case in ('heel','recorded_token') else EXP/'shared_control/sim_scene.py')))
    try:
        simulator = [sys.executable, '-m', 'shared_control.sim_scene']
        if args.case in ('heel','recorded_token'): simulator=[sys.executable,str(run/'g1_sim_heel_scene.py')]
        if args.disable_hand_collisions or args.disable_arm_collisions:
            simulator = [sys.executable, str(run/'g1_sim_scene.py'),
                         '--disable-arm-collisions' if args.disable_arm_collisions else '--disable-hand-collisions']
        launch([*simulator, '--output', str(run/'scene'), '--init-config', str(run/'initialization.json')],
               'simulator', SONIC)
        wait(lambda: (run/'scene/scene.json').exists(), 30, 'scene_ready')
        command = [str(binary), 'lo', str(assets/'model_decoder.onnx'), str(SONIC/'gear_sonic_deploy/reference/example'),
            '--obs-config', str(assets/'observation_config.yaml'), '--encoder-file', str(assets/'model_encoder.onnx'),
            '--planner-file', str(assets/'planner/target_vel/V2/planner_sonic.onnx'), '--input-type', 'zmq_manager',
            '--output-type', 'zmq', '--zmq-host', '127.0.0.1', '--zmq-port', '17576', '--zmq-out-port', '17577',
            '--disable-crc-check', '--logs-dir', str(run/'controller_logs')]
        launch(command, 'controller', SONIC/'gear_sonic_deploy', slave)
        wait(lambda: 'Init Done' in (run/'controller.log').read_text(errors='replace'), 90, 'controller_initialized')
        active = True
        wait(lambda: trace and trace['encoder_source'] and trace['motion'] == 'planner_motion', 15, 'planner_ready')
        dump(run/'scene/command.json', dict(id='start-physics', command='start_physics'))
        wait(lambda: readiness.latest.get('ready', False), 35, 'grounded_standing_ready')
        dump(run/'readiness.json', readiness.latest)
        if demo is not None:
            # Separate experiment: native version-4 tokens, no joint/SMPL refit.
            # Entry/return blends are new synthesized transitions, not recorded actions.
            neutral=np.asarray(trace['token'],dtype=np.float32).copy()
            wav=run/'demo_warning.wav'
            kind='Modified' if demo_candidate else 'Recorded'
            message=f'Simulation replay. {kind} {args.heel_side} ankle relaxation movement, then return.'
            subprocess.run(['espeak','-s','155','-w',str(wav),message],check=True)
            event('demo_audio_started',message=message,synthetic_warning=True)
            audio=subprocess.Popen(['paplay',str(wav)])
            try:
                end=time.monotonic()+20
                while audio.poll() is None:
                    if time.monotonic()>end:raise RuntimeError('Demo audio timeout')
                    tick()
                if audio.returncode:raise RuntimeError('Demo audio failed')
            finally:
                if audio.poll() is None:audio.terminate();audio.wait(timeout=3)
            event('demo_audio_completed')
            walk_start=time.monotonic()
            def blend_token(start,target,seconds,phase):
                nonlocal external_token,demo_phase
                demo_phase=phase;event('demo_phase',phase=phase)
                begin=time.monotonic()
                while time.monotonic()-begin<seconds:
                    u=np.clip((time.monotonic()-begin)/seconds,0,1);b=u**3*(10-15*u+6*u*u)
                    external_token=((1-b)*start+b*target).astype(np.float32);tick()
                external_token=target.copy()
            blend_token(neutral,demo[0],4.,'entry_blend')
            demo_phase='recorded_start_hold';event('demo_phase',phase=demo_phase);hold(2.)
            demo_phase='recorded_episode';event('demo_phase',phase=demo_phase)
            for demo_index in range(len(demo)):
                external_token=demo[demo_index];tick()
            external_token=demo[-1].copy();demo_phase='end_hold';hold(.2)
            demo_index=None
            blend_token(external_token.copy(),demo[0],3.,'synthesized_return')
            demo_phase='returned_start_hold';event('demo_phase',phase=demo_phase);hold(2.)
            blend_token(external_token.copy(),neutral,3.,'idle_blend')
            external_token=None;demo_phase='planner_idle';event('demo_phase',phase=demo_phase);hold(4.)
            def summary(rows):
                return {s:dict(load_mean_n=float(np.mean([r['loads'][s] for r in rows])),
                    heel_mean_m=float(np.mean([r['ankle_telemetry'][s]['heel_min_z_m'] for r in rows])),
                    heel_max_m=float(max(r['ankle_telemetry'][s]['heel_min_z_m'] for r in rows)),
                    toe_contact_fraction=float(np.mean([r['ankle_telemetry'][s]['toe_contact_force_n']>1 for r in rows])))
                        for s in ('left','right')}
            quat=np.asarray([r['root_quat'] for r in walk_records])
            late=[r for r in walk_records if r['elapsed_s']>walk_records[-1]['elapsed_s']-2.]
            result=dict(case='recorded_token',sequence_completed=True,hardware_validated=False,
                thermal_relief_validated=False,recorded_source_frames=len(demo),
                submitted_source_frames=sorted(set(r['source_frame'] for r in walk_records if r['phase']=='recorded_episode')),
                baseline=summary([r for r in walk_records if r['phase']=='recorded_start_hold']),
                replay=summary([r for r in walk_records if r['phase']=='recorded_episode']),
                returned=summary([r for r in walk_records if r['phase']=='returned_start_hold']),
                max_tilt_rad=float(np.max(np.arccos(np.clip(1-2*(quat[:,1]**2+quat[:,2]**2),-1,1)))),
                final_two_second_drift_m=float(np.max(np.linalg.norm(np.asarray([r['root_xyz'] for r in late])-late[0]['root_xyz'],axis=1))),
                max_support_wrench=float(np.max(np.abs([r['support_wrench'] for r in walk_records]))),
                arm_contact_samples=sum(bool(r['geometry']['arm_contacts']) for r in walk_records),
                demo_candidate=demo_candidate,
                note='Delta-modified tokens; entry/return are synthetic. Not thermal or hardware validation.' if demo_candidate else 'Original full-body tokens; entry/return are synthetic. Sequence completion is not a thermal or hardware validation.')
            dump(run/'result.json',result);event('completed',**result);return
        reference = from_idle_observation(trace['encoder_obs'], state)
        np.savez_compressed(run/'reference.npz', **reference)
        event('reference_handoff')
        hold(args.hold_seconds)
        if trace['motion'] != 'streamed':
            raise RuntimeError('Native stream not observed')
        if np.max(abs(np.array(trace['encoder_obs'])[4:294].reshape(10, 29)-reference['joint_pos'][0])) > 1e-4:
            raise RuntimeError('Held reference readback mismatch')
        dump(run/'anchor.json', dict(state=state, native=trace))
        if args.case in ('walk','gait','heel','locomotion'):
            # Clear the arms with the real bounded IK, then hand off only the
            # lower-body references to the native planner. No household objects.
            q0=reference['joint_pos'][0].copy();kin=Kinematics();poses=kin.fk(q0)
            goals={s:np.asarray(poses[s]['position_m'])+[.08,.08 if s=='left' else -.08,.06] for s in poses}
            target,details=kin.solve(q0,goals)
            transition=Transition(reference);transition.duration=6.;transition.begin(target,time.monotonic(),'clear_arms')
            hold(9.)
            if moves and args.carry_pose:
                goals={s:np.asarray(poses[s]['position_m'])+[.22,.06 if s=='left' else -.06,.20] for s in poses}
                target,details=kin.solve(target,goals)
                transition.begin(target,time.monotonic(),'carry_pose');event('carry_pose_prepared',ik=details)
                hold(9.)
            clear_geometry=geometry.measure(state)
            if clear_geometry['arm_contacts']:
                raise RuntimeError('Arm contact before simulated walk')
            reference['joint_pos'][:]=target;transition=None
            if args.case in ('walk','locomotion'):
                walk_upper=upper_fields(target,reference['joint_vel'][0]);reference=None
                if moves:
                    # Exercise a captured measured pose, as the production tool does.
                    moves.upper=upper_fields(np.asarray(state['qpos'])[qids],np.zeros(29))
                event('planner_upper_hold_handoff',ik=details,clear_geometry=clear_geometry)
                hold(4.)
                if trace['motion']!='planner_motion':raise RuntimeError('Planner handoff not observed')
            elif args.case=='heel':
                from concurrent.futures import ThreadPoolExecutor
                from services.g1_sonic.sim_heel import make as make_heel
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future=pool.submit(make_heel,reference,args.heel_side,args.heel_angle,args.heel_shift,args.foot_lift_m)
                    while not future.done():tick()
                    gait,heel_details=future.result()
                dump(run/'heel_plan.json',heel_details)
                np.savez_compressed(run/'heel_reference.npz',**gait)
            else:
                gait=make_gait(reference,args.gait_amplitude,mirrored_pair=args.gait_mirrored_pair)
                np.savez_compressed(run/'gait_reference.npz',**gait)
                event('gait_prepared',amplitude=args.gait_amplitude,source_start=gait['source_start'],source_end=gait['source_end'])
            if moves:
                walk_start=time.monotonic();event('locomotion_started')
                trials=[]
                for direction in args.directions:
                    hold(1.)
                    origin=np.asarray(state['root_qpos'][:3]);heading=body_yaw(state['root_qpos'][3:7])
                    start_index=len(walk_records)
                    command=dict(direction=direction,speed=args.move_speed,duration_s=args.move_duration,
                        mode=args.move_mode if direction in MOTIONS[:8] else 'slowWalk',
                        frame_id='a'*32,evidence='Isolated MuJoCo free floor; captured upper posture; simulated observation')
                    sent=move_submit(run/'movement','run',command,time.monotonic())
                    event('move_started',direction=direction,job_id=sent['job_id'],command=command)
                    wait(lambda:(run/'movement/move_results'/(sent['job_id']+'.json')).exists(),20,'move_result_ready')
                    result=move_result(run/'movement',sent['job_id']);hold(2.)
                    rows=walk_records[start_index:];delta=np.asarray(state['root_qpos'][:3])-origin
                    result.update(direction=direction,actual_root_displacement_m=delta.tolist(),
                        body_frame_displacement_m=[float(np.cos(heading)*delta[0]+np.sin(heading)*delta[1]),float(-np.sin(heading)*delta[0]+np.cos(heading)*delta[1])],
                        upper_measured_max_error_rad=max(r['upper_measured_error_rad'] for r in rows),
                        upper_encoder_max_error_rad=max(r['upper_encoder_error_rad'] for r in rows),
                        max_support_wrench=float(np.max(abs(np.asarray([r['support_wrench'] for r in rows])))),
                        arm_contact_samples=sum(bool(r['geometry']['arm_contacts']) for r in rows))
                    trials.append(result);dump(run/'trials.json',trials);event('move_finished',**result)
                    if not result['completed']:raise RuntimeError('Movement segment failed: '+result['reason'])
                result=dict(case='locomotion',trials=trials,hardware_validated=False,
                    sequence_completed=True,scene_has_carried_object=False,
                    carry_pose=args.carry_pose,
                    note='Native SONIC + MuJoCo, captured waist/arm references, stock Dex3. No BrainCo or payload retention claim.')
                dump(run/'result.json',result);event('completed',**result);return
            if args.case=='heel':
                # Explicit user requirement: audible explanation precedes motion.
                # Failure to synthesize/play blocks this simulation trial.
                part='foot' if args.foot_lift_m else 'heel'
                message=f'Simulation. Warning! High temperature at {args.heel_side} ankle. Lifting the {args.heel_side} {part}.'
                wav=run/'ankle_warning.wav'
                subprocess.run(['espeak','-s','155','-w',str(wav),message],check=True)
                event('heel_audio_started',side=args.heel_side,message=message,synthetic_warning=True)
                audio=subprocess.Popen(['paplay',str(wav)],start_new_session=True)
                children.append(audio)
                # tick checks children; audio is allowed to exit, unlike physics peers.
                children.remove(audio)
                audio_deadline=time.monotonic()+20
                try:
                    while audio.poll() is None:
                        if time.monotonic()>audio_deadline:raise RuntimeError('Simulation audio playback timed out')
                        tick()
                    if audio.returncode!=0:raise RuntimeError('Simulation warning audio failed; heel motion blocked')
                finally:
                    if audio.poll() is None:audio.terminate();audio.wait(timeout=3)
                event('heel_audio_completed',side=args.heel_side,exit_code=audio.returncode)
            origin=np.array(state['root_qpos'][:3]);walk_start=time.monotonic()
            if gait is not None:
                event('heel_started' if args.case=='heel' else 'gait_started',amplitude=args.gait_amplitude,mirrored_pair=args.gait_mirrored_pair,duration_s=gait['duration_s'])
            else:
                event('walk_started',speed=args.walk_speed,plateau_seconds=args.walk_plateau_seconds,
                      command_integral_m=args.walk_speed*(args.walk_plateau_seconds+RAMP_SECONDS))
            hold((gait['duration_s'] if gait is not None else 2*RAMP_SECONDS+args.walk_plateau_seconds)+8.)
            if args.case=='heel':
                from services.g1_sonic.sim_heel import audit
                result=audit(walk_records,heel_details)
                dump(run/'result.json',result);event('completed',**result);return
            positions=np.array([r['root_xyz'] for r in walk_records]);quat=np.array([r['root_quat'] for r in walk_records])
            tilts=np.arccos(np.clip(1-2*(quat[:,1]**2+quat[:,2]**2),-1,1))
            late=[r for r in walk_records if r['elapsed_s']>walk_records[-1]['elapsed_s']-2.]
            drift=float(np.max(np.linalg.norm(np.array([r['root_xyz'] for r in late])-late[0]['root_xyz'],axis=1)))
            delta=positions[-1]-origin
            arm_contacts=sum(bool(r['geometry']['arm_contacts']) for r in walk_records)
            stop_speed=max(float(np.linalg.norm(r['root_velocity'][:3])) for r in late)
            times=np.array([r['sim_time'] for r in walk_records]);dt=np.r_[np.diff(times),0]
            footsteps={}
            for side in ('left','right'):
                feet=np.array([r['feet'][side] for r in walk_records]);loads=np.array([r['loads'][side] for r in walk_records])
                lift=feet[:,2]-feet[0,2]
                footsteps[side]=dict(peak_lift_m=float(max(lift)),unloaded_lift_s=float(dt[(lift>.005)&(loads<40)].sum()),
                    displacement_m=(feet[-1]-feet[0]).tolist())
            both_stepped=all(v['unloaded_lift_s']>=.04 and v['displacement_m'][0]>.01 for v in footsteps.values())
            expected_feet=('left','right') if args.case=='walk' or args.gait_mirrored_pair else ('left',)
            expected_stepped=all(footsteps[s]['unloaded_lift_s']>=.04 and footsteps[s]['displacement_m'][0]>.01 for s in expected_feet)
            passed=bool(.03<=delta[0]<=.20 and abs(delta[1])<.05 and max(tilts)<.25 and
                        drift<.025 and stop_speed<.08 and not arm_contacts and trace['locomotion']==0 and expected_stepped)
            result=dict(completed=passed,hardware_validated=False,case=args.case,actual_root_displacement_m=delta.tolist(),
                        commanded_distance_is_not_odometry=True,max_tilt_rad=float(max(tilts)),
                        final_two_second_drift_m=drift,final_two_second_max_speed_m_s=stop_speed,
                        arm_contact_samples=arm_contacts,final_locomotion=trace['locomotion'],
                        max_body_joint_speed_rad_s=max(max(map(abs,r['body_dq'])) for r in walk_records),
                        both_feet_stepped=both_stepped,footsteps=footsteps,
                        expected_stepping_feet=list(expected_feet),expected_feet_stepped=expected_stepped,
                        final_foot_load_n=state['foot_ground_normal_force_n'],
                        max_support_wrench=float(np.max(abs(np.array([r['support_wrench'] for r in walk_records])))))
            dump(run/'result.json',result);event('completed',**result);return
        if args.case == 'reach':
            transition = Transition(reference)
        job_start = time.monotonic()
        event('job_started', case=args.case)
        if args.case == 'reach':
            result = run_reach(reference, transition, reach_solver, lambda: state, records, tick, event,
                               lambda name, value: dump(run/name, value))
            dump(run/'result.json', result)
            event('completed', **result)
            return
        hold(17)
        elbow = NAMES.index('right_elbow_joint')
        q = np.array([r['q'] for r in records]); wrists = np.array([r['wrist_base'] for r in records])
        q0 = np.array(state['qpos'])[qids] if not len(q) else q[0]
        late = np.array([r['elapsed_s'] >= 15 for r in records])
        result = dict(status='completed', case=args.case, samples=len(records),
            elbow_peak_change_rad=float(np.max(abs(q[:, elbow]-q0[elbow]))),
            elbow_same_direction_response_rad=float(np.max(q[:, elbow]-q0[elbow])),
            elbow_return_error_rad=float(abs(np.median(q[late, elbow])-q0[elbow])),
            max_body_return_error_rad=float(np.max(abs(np.median(q[late], axis=0)-q0))),
            wrist_base_peak_displacement_m=float(np.max(np.linalg.norm(wrists-wrists[0], axis=1))),
            wrist_base_return_error_m=float(np.linalg.norm(np.median(wrists[late], axis=0)-wrists[0])),
            max_root_tilt_rad=max(r['root_tilt_rad'] for r in records),
            native_reference_returned=bool(records[-1]['native_future_qdq_max_error'] < 1e-4),
            joint_changes={name:dict(peak_rad=float(np.max(abs(q[:, i]-q0[i]))),
                return_rad=float(abs(np.median(q[late, i])-q0[i]))) for i, name in enumerate(NAMES)})
        dump(run/'result.json', result)
        event('completed', **{k:v for k,v in result.items() if k != 'joint_changes'})
    except BaseException as exc:
        dump(run/'failure.json', dict(error=repr(exc), state=state, readiness=readiness.latest))
        raise
    finally:
        # Only our isolated simulation children; no field controller is involved.
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
        for handle in handles:
            handle.close()
        os.close(master); os.close(slave)
        events.close(); samples.close(); pub.close(0); context.term(); lock.close()


if __name__ == '__main__':
    main()
