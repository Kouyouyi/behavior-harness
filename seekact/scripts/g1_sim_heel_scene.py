"""Simulation-only instrumented copy of CoMotion shared_control/sim_scene.py.
Adds measured sole contacts and actuator torque; physics/control unchanged.

Headless SONIC MuJoCo scene with explicit support release and bounded logging.

Instrumentation follows Harness mujoco_scene.py. Physics and the Unitree bridge
remain the original SONIC implementation. This process only joins DDS domain 77.
"""
from pathlib import Path
import argparse
import json
import queue
import signal
import threading
import time

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-height", type=float, default=0.85)
    parser.add_argument("--release-ramp", type=float, default=1.0)
    parser.add_argument("--friction-scale", type=float, default=1.0)
    parser.add_argument("--init-config", type=Path)
    parser.add_argument('--viewer', action='store_true')
    args = parser.parse_args()
    if not .8 <= args.support_height <= 1 or not 0 < args.release_ramp <= 2 or not .25 <= args.friction_scale <= 2:
        raise ValueError("Invalid simulation initialization")
    args.output.mkdir(parents=True, exist_ok=False)
    init_config = json.loads(args.init_config.read_text()) if args.init_config else None
    if init_config:
        from shared_control.initialization import ContactInitializer, GroundInitializer
        (args.output / "initialization_config.json").write_text(json.dumps(init_config,indent=2))
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory
    config = SimLoopConfig(interface="lo", enable_onscreen=args.viewer, enable_offscreen=False,
                           enable_image_publish=False)
    wbc = config.load_wbc_yaml()
    wbc.update(DOMAIN_ID=77, INTERFACE="lo", PRINT_SCENE_INFORMATION=False)
    (args.output / "effective_config.json").write_text(json.dumps(wbc, indent=2))
    records = queue.Queue(maxsize=2048)
    errors = []
    pending_command = [None]
    stopping = threading.Event()
    def io_worker():
        last_command_id = None
        with (args.output / "trajectory.jsonl").open("w") as file:
            while not stopping.is_set() or not records.empty():
                try:
                    record = records.get(timeout=.02)
                except queue.Empty:
                    record = None
                if record is not None:
                    file.write(json.dumps(record, separators=(",", ":")) + "\n")
                    file.flush()
                    temporary = args.output / "latest.tmp"
                    temporary.write_text(json.dumps(record))
                    temporary.replace(args.output / "latest.json")
                command_file = args.output / "command.json"
                if command_file.exists():
                    command = json.loads(command_file.read_text())
                    if command["id"] != last_command_id:
                        pending_command[0] = command
                        last_command_id = command["id"]
    def guarded_io():
        try:
            io_worker()
        except Exception as error:
            errors.append(repr(error))
    writer = threading.Thread(target=guarded_io, daemon=True)
    writer.start()
    old_init = DefaultEnv.init_scene
    old_step = DefaultEnv.sim_step
    old_body_torques = DefaultEnv.compute_body_torques
    last_sample, release_start, band_gains = 0.0, None, None
    last_command, fell, dropped = None, False, 0
    initializer = None
    ground_ids = set()
    foot_geoms = {}
    def init_scene(self):
        nonlocal initializer, ground_ids, foot_geoms
        old_init(self)
        ground = [i for i in range(self.mj_model.ngeom)
                  if self.mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
                  and self.mj_model.geom_bodyid[i] == 0]
        self.mj_model.geom_friction[ground, 0] *= args.friction_scale
        self.elastic_band.point = self.elastic_band.point.astype(float)
        self.elastic_band.point[2] = args.support_height
        ground_ids = set(ground)
        if init_config:
            weight = float(self.mj_model.body_mass.sum()*abs(self.mj_model.opt.gravity[2]))
            if init_config.get('mode') == 'ground_start':
                q = np.asarray(init_config['initial_qpos'], dtype=float)
                v = np.asarray(init_config['initial_qvel'], dtype=float)
                if q.shape != self.mj_data.qpos.shape or v.shape != self.mj_data.qvel.shape or not np.isfinite(q).all() or not np.isfinite(v).all():
                    raise ValueError('Invalid ground reset state')
                self.mj_data.qpos[:] = q
                self.mj_data.qvel[:] = v
                self.mj_data.qacc_warmstart[:] = 0
                self.mj_data.xfrc_applied[:] = 0
                self.mj_data.qfrc_applied[:] = 0
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.elastic_band.point[:] = q[:3]
                initializer = GroundInitializer(self.elastic_band,init_config,weight)
                (args.output/'initial_state.json').write_text(json.dumps(dict(qpos=q.tolist(),qvel=v.tolist(),
                    physics_paused=True,external_wrench_zero=True),indent=2))
            else:
                initializer = ContactInitializer(self.elastic_band,init_config,weight)
            if init_config.get("vertical_support_lift_only",False):
                original_advance = self.elastic_band.Advance
                def lift_only_advance(pose):
                    if initializer.handoff_wrench is not None:
                        return initializer.handoff_wrench*initializer.support_scale
                    wrench = original_advance(pose)
                    wrench[:2] *= initializer.nonvertical_scale
                    wrench[3:] *= initializer.nonvertical_scale
                    # This temporary initialization support may lift, but cannot
                    # pull the standing robot downward toward a low anchor.
                    wrench[2] = max(0.0,wrench[2])
                    return wrench
                self.elastic_band.Advance = lift_only_advance
            for side in ("left","right"):
                body = self.mj_model.body(side+"_ankle_roll_link").id
                foot_geoms[side] = [i for i in range(self.mj_model.ngeom)
                    if self.mj_model.geom_bodyid[i] == body
                    and self.mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
                    and (self.mj_model.geom_contype[i] or self.mj_model.geom_conaffinity[i])]
                if not foot_geoms[side]:
                    raise ValueError("Expected box foot contact geometry")
            if init_config.get('mode') == 'ground_start':
                for indices in foot_geoms.values():
                    sole=min(float(self.mj_data.geom_xpos[i,2]-np.abs(self.mj_data.geom_xmat[i].reshape(3,3)[2])@self.mj_model.geom_size[i]) for i in indices)
                    if abs(sole-init_config['expected_sole_min_z_m'])>1e-6:
                        raise ValueError('Ground start feet do not match frozen placement')
        metadata = dict(dds_domain=77, interface="lo", support_height=args.support_height,
                        release_ramp_s=init_config["unload_sim_s"] if init_config else args.release_ramp, friction_scale=args.friction_scale,
                        scene=wbc["ROBOT_SCENE"], sim_dt=wbc["SIMULATE_DT"],
                        body_names=[self.mj_model.body(i).name for i in range(self.mj_model.nbody)])
        if initializer:
            metadata.update(initialization=init_config,robot_weight_n=initializer.weight_n)
        (args.output / "scene.json").write_text(json.dumps(metadata, indent=2))
    def sim_step(self):
        if init_config and init_config.get('mode') == 'ground_start':
            if initializer.phase == 'ground_wait':
                # Controller receives a fresh message, but physics time stays at
                # zero. This is reset preparation, never physical stability data.
                self.obs = self.prepare_obs()
                self.unitree_bridge.PublishLowState(self.obs)
                self.check_fall()
                return
            initializer.physics_steps += 1
        old_step(self)
    def compute_body_torques(self):
        if not init_config or not init_config.get('joint_handoff'):
            return old_body_torques(self)
        bridge=self.unitree_bridge
        if bridge is None or not bridge.low_cmd:
            raise RuntimeError('Ground joint handoff requires received motor commands')
        commands=bridge.low_cmd.motor_cmd[:bridge.num_body_motor]
        q=np.asarray([c.q for c in commands]);dq=np.asarray([c.dq for c in commands])
        mixed_q,mixed_dq=initializer.mix_joint_targets(float(self.mj_data.time),q,dq)
        if bridge.use_sensor:
            measured_q=self.mj_data.sensordata[:bridge.num_body_motor]
            measured_dq=self.mj_data.sensordata[bridge.num_body_motor:2*bridge.num_body_motor]
        else:
            measured_q=self.mj_data.qpos[self.body_joint_index+self.qpos_offset-1]
            measured_dq=self.mj_data.qvel[self.body_joint_index+self.qvel_offset-1]
        return (np.asarray([c.tau for c in commands])+np.asarray([c.kp for c in commands])*(mixed_q-measured_q)
                +np.asarray([c.kd for c in commands])*(mixed_dq-measured_dq))
    def check_fall(self):
        nonlocal last_sample, release_start, band_gains, last_command, fell, dropped
        now = time.monotonic()
        command = pending_command[0]
        if command is not None and command["id"] != last_command:
            if initializer:
                initializer.command(command["command"],float(self.mj_data.time),
                                    self.mj_data.xfrc_applied[self.band_attached_link])
            elif command["command"] != "release":
                raise ValueError("Only support release is accepted; reset restarts this process")
            else:
                release_start = now
                band_gains = {key: getattr(self.elastic_band, key) for key in ("kp_pos", "kd_pos", "kp_ang", "kd_ang")}
            last_command = command["id"]
        if release_start is not None:
            phase = min(1.0, (now - release_start) / args.release_ramp)
            factor = 1 - phase * phase * (3 - 2 * phase)
            for key, value in band_gains.items():
                setattr(self.elastic_band, key, value * factor)
            if phase == 1:
                self.elastic_band.enable = False
                release_start = None
        fell = fell or float(self.mj_data.qpos[2]) < .4
        self.fall = fell
        if now - last_sample < .02 and not initializer:
            return
        forces = dict(left=0.0, right=0.0)
        ground_forces = dict(left=0.0,right=0.0)
        feet_ids = {side: self.mj_model.body(side + "_ankle_roll_link").id for side in forces}
        for index, contact in enumerate(self.mj_data.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, index, force)
            bodies = [int(self.mj_model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
            for side, body in feet_ids.items():
                if body in bodies:
                    forces[side] += abs(float(force[0]))
                    if any(int(g) in ground_ids for g in (contact.geom1,contact.geom2)):
                        ground_forces[side] += abs(float(force[0]))
        if initializer:
            initializer.update(float(self.mj_data.time),ground_forces,float(self.mj_data.qvel[2]),
                               float(self.mj_data.xfrc_applied[self.band_attached_link,2]))
        if now - last_sample < .02:
            return
        record = dict(monotonic=now, sim_time=float(self.mj_data.time), qpos=self.mj_data.qpos.tolist(),
                      qvel=self.mj_data.qvel.tolist(), root_qpos=self.mj_data.qpos[:7].tolist(),
                      root_qvel=self.mj_data.qvel[:6].tolist(), support=bool(self.elastic_band.enable),
                      release_in_progress=release_start is not None, fell=bool(fell),
                      feet={side: self.mj_data.body(side + "_ankle_roll_link").xpos.tolist() for side in forces},
                      wrists={side: self.mj_data.body(side + "_wrist_yaw_link").xpos.tolist() for side in forces},
                      foot_normal_force_n=forces, dropped_records=dropped, io_errors=list(errors))
        ankle_telemetry = {}
        for side, body_id in feet_ids.items():
            body = self.mj_data.body(body_id)
            rot = body.xmat.reshape(3, 3)
            toe = [body.xpos + rot @ np.array([.12, y, -.035]) for y in (-.03, .03)]
            heel = [body.xpos + rot @ np.array([-.05, y, -.035]) for y in (-.03, .03)]
            toe_force = heel_force = 0.
            for index, contact in enumerate(self.mj_data.contact):
                bodies = [int(self.mj_model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
                if body_id not in bodies or not any(int(g) in ground_ids for g in (contact.geom1, contact.geom2)):
                    continue
                force = np.zeros(6); mujoco.mj_contactForce(self.mj_model, self.mj_data, index, force)
                local = rot.T @ (contact.pos - body.xpos)
                if local[0] > .08: toe_force += abs(float(force[0]))
                if local[0] < 0.: heel_force += abs(float(force[0]))
            ankle_telemetry[side] = dict(toe_min_z_m=min(float(p[2]) for p in toe),
                heel_min_z_m=min(float(p[2]) for p in heel), toe_contact_force_n=toe_force,
                heel_contact_force_n=heel_force,
                pitch_torque_nm=float(self.mj_data.qfrc_actuator[self.mj_model.joint(side+'_ankle_pitch_joint').dofadr[0]]),
                roll_torque_nm=float(self.mj_data.qfrc_actuator[self.mj_model.joint(side+'_ankle_roll_joint').dofadr[0]]))
        record['ankle_telemetry'] = ankle_telemetry
        if initializer:
            record.update(initialization=initializer.status(),foot_ground_normal_force_n=ground_forces,
                          support_wrench=self.mj_data.xfrc_applied[self.band_attached_link].tolist(),
                          release_in_progress=initializer.phase in ("lowering","load_for_release","unloading"),
                          foot_sole_min_z_m={side:min(float(self.mj_data.geom_xpos[i,2]
                              -np.abs(self.mj_data.geom_xmat[i].reshape(3,3)[2]) @ self.mj_model.geom_size[i])
                              for i in indices) for side,indices in foot_geoms.items()})
            if init_config.get('mode') == 'ground_start':
                record['physics_paused'] = initializer.phase == 'ground_wait'
                record['physics_steps'] = initializer.physics_steps
                record['joint_handoff_in_progress'] = initializer.joint_handoff_alpha < 1.
                record['applied_joint_target'] = initializer.applied_joint_target
        try:
            records.put_nowait(record)
        except queue.Full:
            dropped += 1
        last_sample = now
    viewers = []
    if args.viewer:
        from gear_sonic.utils.mujoco_sim import base_sim
        from shared_control.viewer_lifecycle import PassiveViewer
        original_viewer = base_sim.mujoco.viewer.launch_passive
        def protocol_viewer(*pos, **kwargs):
            kwargs.pop('key_callback', None)
            viewer = PassiveViewer(*pos, launcher=original_viewer, **kwargs)
            viewers.append(viewer)
            return viewer.handle
        base_sim.mujoco.viewer.launch_passive = protocol_viewer
    DefaultEnv.init_scene = init_scene
    DefaultEnv.sim_step = sim_step
    DefaultEnv.compute_body_torques = compute_body_torques
    DefaultEnv.check_fall = check_fall
    # BaseSimulator initializes its own channel exactly once.
    simulator = SimulatorFactory.create_simulator(config=wbc, env_name="default", onscreen=args.viewer,
                                                 offscreen=False, enable_image_publish=False)
    signal.signal(signal.SIGTERM, lambda *_: setattr(simulator, "_running", False))
    try:
        SimulatorFactory.start_simulator(simulator, as_thread=False, enable_image_publish=False)
    finally:
        stopping.set()
        writer.join(timeout=2.)
        (args.output / "recorder_final.json").write_text(json.dumps(dict(
            dropped_records=dropped, io_errors=errors, pending=records.qsize(), writer_alive=writer.is_alive())))
        for viewer in viewers:
            viewer.close()


if __name__ == "__main__":
    main()
