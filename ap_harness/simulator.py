"""Reuse the installed official R1Pro runtime, never its old OpenPI policy."""
from pathlib import Path
import time
import cv2
import numpy as np
from .paths import HANDOFF
from .control import save_json
from .fixtures import ScenarioBackend

class Simulator:
    def __init__(self, output, *, max_steps=800, ood='none', seed=0, container_name=None,
                 task_name='turning_on_radio', instance_id=301, target_key='radio_receiver.n.01_1',
                 initial_visibility_audit=False, initial_yaw_offset_deg=0.0, initial_offset_x=0.0, initial_offset_y=0.0, scenario_spec=None, seed_override=None):
        self.initial_visibility_audit = initial_visibility_audit
        self.backend = ScenarioBackend({
            'output_dir': str(output), 'task_name': task_name, 'instance_id': instance_id,
            'target_key': target_key, 'scenario_spec':scenario_spec,
            'mode': 'public_test', 'run_mode': 'assisted', 'head_mode': 'physical_trunk',
            'ood': ood, 'container_name':container_name, 'initial_yaw_offset_deg':initial_yaw_offset_deg, 'initial_offset_x':initial_offset_x, 'initial_offset_y':initial_offset_y, 'seed': seed if seed_override is None else seed_override, 'max_steps': max_steps, 'headless': True,
            'write_video': True, 'policy_kind': 'local',
            'robot_config': str(HANDOFF / 'r1pro.yaml')})

    def start(self):
        self.backend.start()
        if self.backend.config.get('scenario_spec'):
            from .ood99 import face_away
            face_away(self)
        # Evaluator finalization reloads the robot pose; apply initial heading
        # only AFTER that reset, before any task action or observation is sent.
        offset = float(self.backend.config.get('initial_yaw_offset_deg', 0))
        if offset or self.backend.config.get('initial_offset_x') or self.backend.config.get('initial_offset_y'):
            self.backend._setup_initial_heading(self.backend.state)
            hold = self.backend._make_hold_action(self.backend.robot)
            for _ in range(30):
                self.backend.robot.apply_action(hold)
                self.backend.og.sim.step()
            self.refresh()
            import json
            setup = json.loads((self.backend.output_dir/'initial_heading_setup.json').read_text())
            actual = self.backend._array(self.backend.robot.get_position_orientation()[1])
            error = (self.backend.Rotation.from_quat(actual) * self.backend.Rotation.from_quat(setup['requested_orientation']).inv()).magnitude()
            save_json(self.backend.output_dir/'initial_heading_validation.json', {
                'actual_orientation':actual.tolist(), 'orientation_error_rad':float(error),
                'setup_only_physics_steps':30, 'task_steps':self.steps})
            if error > 0.05:
                raise RuntimeError('Initial search heading failed to persist through settling')
        actual = {k: v.tolist() for k, v in self.backend._action_idx.items() if len(v)}
        expected = {'base': list(range(3)), 'trunk': list(range(3,7)),
                    'arm_left': list(range(7,14)), 'gripper_left': [14],
                    'arm_right': list(range(15,22)), 'gripper_right': [22]}
        if actual != expected:
            raise ValueError(f'Action ordering mismatch: {actual}')
        dt = float(self.backend.og.sim.get_sim_step_dt())
        if not np.isclose(dt, 1/30):
            raise ValueError(f'Expected 30Hz simulation, got dt={dt}')
        save_json(self.backend.output_dir / 'interface_validation.json', {
            'action_indices': actual, 'sim_step_dt': dt,
            'robot_config': str(HANDOFF / 'r1pro.yaml'),
            'observation_source': 'official robot_r1::proprio and native robot sensors',
            'joint_axis_physical_probe_passed': False})
        from .scoring import RenderedTargetScorer
        self.target = self.backend.state.env_accessor.object_scope[self.backend.config['target_key']]
        self.scorer = None
        # Isaac 5.1's synthetic-data graph crashes during physics stepping here.
        # Isolate optional visibility auditing in initialize-only processes.
        if self.initial_visibility_audit:
            self.scorer = self.make_scorer()
            self.scorer.enable()
            self.refresh()
            evidence = self.scorer()
            save_json(self.backend.output_dir/'initial_occlusion_private.json', evidence)
            if hasattr(self.scorer,'save_raw'):self.scorer.save_raw(self.backend.output_dir/'initial_masks_private')
            from .ood99 import fingerprint
            save_json(self.backend.output_dir/'initial_fingerprint_private.json', fingerprint(self))
            if self.backend.config.get('scenario_spec'):
                save_json(self.backend.output_dir/'initial_fingerprint_private.json', fingerprint(self))
                if any(r['target_visible'] for r in evidence['equivalent_target_evidence']):
                    raise RuntimeError('OOD rejected: an equivalent target is initially visible')
            if self.backend.config['ood'] == 'radio_inside_closed_container' and any(
                    x['pixel_count'] for x in evidence['visibility_evidence']['cameras'].values()):
                raise RuntimeError('Target is still rendered; occlusion fixture rejected')

    @property
    def steps(self): return self.backend.steps

    def make_scorer(self):
        from .scoring import RenderedTargetScorer
        if self.backend.config.get('scenario_spec'):
            from .ood99 import RawMeshTargetSetScorer
            return RawMeshTargetSetScorer(self.backend.robot,self.backend.suite_targets,self.backend.evaluator.robot_camera_names)
        from .ood99 import RawMeshTargetSetScorer
        return RawMeshTargetSetScorer(self.backend.robot,[self.target],self.backend.evaluator.robot_camera_names)
    @property
    def done(self): return self.backend.done or self.backend._stop_event.is_set()
    @property
    def success(self): return self.backend.success
    @property
    def termination_reason(self):
        return 'stop_requested' if self.backend._stop_event.is_set() else self.backend.termination_reason

    def refresh(self): self.backend._refresh_sensor_observations()
    def step(self, action):
        if getattr(self, '_terminal_audit_started', False):
            raise RuntimeError('Physics cannot resume after terminal visibility audit')
        return self.backend._step(action, 'vla')

    def capture(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = self.backend._array(self.backend.state.obs['robot_r1::proprio'])
        if state.shape != (61,) or not np.isfinite(state).all():
            raise ValueError('Invalid official raw proprio state')
        frames = self.backend._frames()
        paths = []
        for role in ('head', 'left_wrist', 'right_wrist'):
            rgb = frames[role]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f'Invalid current RGB: {role}')
            path = directory / f'{role}.png'
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise IOError(path)
            paths.append(str(path.resolve()))
        result = {'state': state.astype(np.float32).tolist(), 'image_paths': paths,
                  'step': self.steps, 'captured_ns': time.time_ns()}
        save_json(directory / 'observation.json', result)
        if hasattr(self, 'target'):
            private = self.scorer() if self.scorer else {
                'target_visible': None, 'visibility_measurement':'not instrumented during physics rollout',
                'target_position':self.backend._array(self.target.get_position_orientation()[0]).tolist(),
                'base_position':self.backend._array(self.backend.robot.get_position_orientation()[0]).tolist()}
            private.update(step=self.steps, official_task_success=self.success)
            if self.backend.config.get('scenario_spec'):
                from omnigibson.object_states import Open
                private['container_open_states']={name:bool(obj.states[Open].get_value()) for name,obj in self.backend.suite_closed.items()}
                private['all_target_positions']={obj.name:self.backend._array(obj.get_position_orientation()[0]).tolist() for obj in self.backend.suite_targets}
            if self.backend.config['ood'] == 'radio_inside_closed_container':
                target, container = self.backend._fixture
                private['container_open'] = bool(container.states[self.backend.Open].get_value())
                private['container_joint_positions'] = self.backend._array(container.get_joint_positions()).tolist()
            save_json(directory/'scorer_private.json', private)
        return result

    def verify_approach(self, directory):
        from .scoring import RenderedTargetScorer, evaluate_approach
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        save_json(directory/'started.json', {'step': self.steps, 'physics_must_not_resume': True})
        self._terminal_audit_started = True
        # Flush videos before touching the native synthetic-data graph.
        # A native audit crash must not destroy completed physical evidence.
        for writer in self.backend._writers.values():
            writer.close()
        save_json(directory/'videos_finalized.json', {k:w.metadata() for k,w in self.backend._writers.items()})
        self.scorer = self.make_scorer()
        self.scorer.enable()
        self.refresh()
        evidence = evaluate_approach(self.scorer())
        if hasattr(self.scorer,'save_raw'):self.scorer.save_raw(directory/'masks_private')
        evidence['step'] = self.steps
        save_json(directory/'result.json', evidence)
        return evidence

    def save_handoff(self, directory):
        import hashlib
        from .ood99 import fingerprint
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        state=self.backend._array(self.backend.og.sim.dump_state(serialized=True))
        np.save(directory/'world_state.npy',state)
        save_json(directory/'metadata.json',{'task_name':self.backend.task_name,
            'instance_id':self.backend.instance_id,'source_steps':self.steps,
            'objects':[o.name for o in self.backend.state.env_accessor.scene.objects],
            'fingerprint':fingerprint(self),'state_sha256':hashlib.sha256((directory/'world_state.npy').read_bytes()).hexdigest(),
            'saved_before_terminal_segmentation':True})

    def restore_handoff(self, directory):
        import hashlib,json
        from .ood99 import fingerprint,match_fingerprint
        directory=Path(directory);m=json.loads((directory/'metadata.json').read_text())
        if m['task_name']!=self.backend.task_name or m['instance_id']!=self.backend.instance_id:
            raise ValueError('Handoff task identity mismatch')
        if m['objects']!=[o.name for o in self.backend.state.env_accessor.scene.objects]:
            raise ValueError('Handoff scene object order differs')
        if hashlib.sha256((directory/'world_state.npy').read_bytes()).hexdigest()!=m['state_sha256']:
            raise ValueError('Handoff state checksum mismatch')
        state=np.load(directory/'world_state.npy',allow_pickle=False)
        self.backend.og.sim.load_state(self.backend.th.as_tensor(state,device=self.backend.og.sim.device),serialized=True)
        self.refresh()
        # OmniGibson intentionally skips AABB loading and wakes moved objects.
        # Restore sleep flags only after all poses are loaded, and recompute
        # the derived AABB cache from restored link poses without a physics step.
        expected_tree,_=self.backend.og.sim.deserialize(self.backend.th.as_tensor(state,device=self.backend.og.sim.device))
        for scene_idx,scene in enumerate(self.backend.og.sim.scenes):
            saved_objects=expected_tree[scene_idx]['registry']['object_registry']
            for obj in scene.objects:
                saved=saved_objects.get(obj.name,{})
                if saved.get('is_asleep',False):obj.sleep()
        from omnigibson.utils.usd_utils import RigidBodyViewAPI
        from omnigibson.object_states import AABB
        import warp as wp
        RigidBodyViewAPI.invalidate_kinematic([link for scene in self.backend.og.sim.scenes for obj in scene.objects for link in obj.links.values()])
        wp.copy(RigidBodyViewAPI.POSES_GPU,RigidBodyViewAPI.POSES)
        RigidBodyViewAPI.update()
        AABB._update_values(AABB.VALUES)
        wp.synchronize()
        AABB.VALUES_CPU.copy_(AABB.VALUES)
        restored=self.backend._array(self.backend.og.sim.dump_state(serialized=True))
        if restored.shape!=state.shape or not np.allclose(restored,state,atol=1e-4,rtol=1e-5):
            np.save(self.backend.output_dir/'handoff_restored_actual.npy',restored)
            diffs=[]
            expected_tree,_=self.backend.og.sim.deserialize(self.backend.th.as_tensor(state,device=self.backend.og.sim.device))
            actual_tree=self.backend.og.sim.dump_state(serialized=False)
            def compare(a,b,path):
                if isinstance(a,dict) and isinstance(b,dict):
                    for k in a:
                        if k in b:compare(a[k],b[k],path+'/'+str(k))
                    return
                try:
                    aa=self.backend._array(a);bb=self.backend._array(b)
                    if aa.shape!=bb.shape or not np.allclose(aa,bb,atol=1e-4,rtol=1e-5):
                        diffs.append({'path':path,'expected':aa.tolist(),'actual':bb.tolist()})
                except (TypeError,ValueError,AttributeError):
                    if str(a)!=str(b):diffs.append({'path':path,'expected':str(a),'actual':str(b)})
            compare(expected_tree,actual_tree,'')
            save_json(self.backend.output_dir/'handoff_mismatch.json',{'differences':diffs,'shape_expected':list(state.shape),'shape_actual':list(restored.shape)})
            raise ValueError('Restored full simulator state differs from saved handoff')
        match_fingerprint(fingerprint(self),m['fingerprint'])
        save_json(self.backend.output_dir/'handoff_restore.json',{'verified':True,'source':str(directory.resolve()),'source_steps':m['source_steps'],'full_state_checked':True,'physics_steps_added':0})

    def close(self, reason): return self.backend.close(reason)
    def shutdown(self): self.backend.shutdown_runtime()
