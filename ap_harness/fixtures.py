"""Privileged initial scene setup, never a policy observation or runtime action."""
import copy
import numpy as np
from .paths import ROOT
from .control import save_json
from behavior_harness.backend import BehaviorBackend

class ScenarioBackend(BehaviorBackend):
    def _setup_fixture(self, state):
        if self.config.get('scenario_spec'):
            from .ood99 import setup
            return setup(self, state)
        from omnigibson.object_states import Inside, Open
        save_json(self.output_dir/'container_inventory_private.json', [
            {'name':obj.name, 'category':obj.category}
            for obj in state.env_accessor.scene.objects if Open in getattr(obj, 'states', {})])
        if self.config.get('ood') != 'radio_inside_closed_container':
            super()._setup_fixture(state)
            return
        self.Inside, self.Open = Inside, Open
        target = state.env_accessor.object_scope['radio_receiver.n.01_1']
        container = state.env_accessor.scene.object_registry('name', self.config['container_name'])
        if container is None or Open not in container.states:
            raise ValueError('Named container is missing or cannot open')
        original = copy.deepcopy(self.og.sim.dump_state(serialized=False))
        try:
            if not container.states[Open].set_value(True, fully=True):
                raise RuntimeError('Failed to open setup container')
            if not target.states[Inside].set_value(container, True):
                raise RuntimeError('Failed to sample an inside placement')
            sampled_position, sampled_orientation = [v.clone() for v in target.get_position_orientation()]
            point = self._array(sampled_position)
            enclosing_links = []
            for name, link in container.links.items():
                lower, upper = [self._array(v) for v in link.aabb]
                if np.all(point >= lower-0.005) and np.all(point <= upper+0.005):
                    pos, quat = link.get_position_orientation()
                    enclosing_links.append((float(np.prod(upper-lower)), name, link,
                                            self._array(pos).copy(), self._array(quat).copy()))
            if not container.states[Open].set_value(False, fully=True):
                raise RuntimeError('Failed to close setup container')
            carried_by = None
            for _, name, link, open_pos, open_quat in sorted(enclosing_links, key=lambda row:row[0]):
                closed_pos, closed_quat = [self._array(v) for v in link.get_position_orientation()]
                open_rotation = self.Rotation.from_quat(open_quat)
                closed_rotation = self.Rotation.from_quat(closed_quat)
                delta = closed_rotation * open_rotation.inv()
                if np.linalg.norm(closed_pos-open_pos) < 1e-4 and delta.magnitude() < 1e-4:
                    continue
                # Open.set_value teleports an articulated drawer. Preserve the
                # sampled object's drawer-local pose during INITIAL setup only.
                sampled_position = self.th.as_tensor(closed_pos + delta.apply(point-open_pos), dtype=self.th.float32)
                sampled_orientation = self.th.as_tensor((delta*self.Rotation.from_quat(
                    self._array(sampled_orientation))).as_quat(), dtype=self.th.float32)
                carried_by = name
                break
            container_state = copy.deepcopy(container.dump_state(serialized=False))
        finally:
            # Samplers can advance the world. Restore all other objects and robot.
            self.og.sim.load_state(original, serialized=False)
        container.load_state(container_state, serialized=False)
        # Apply the sampled pose explicitly after restoring the rest of the world.
        # Do not carry the sampler's temporary object velocities into the rollout.
        target.set_position_orientation(position=sampled_position, orientation=sampled_orientation)
        target.keep_still()
        self._fixture = (target, container)
        save_json(self.output_dir/'setup.json', {'ood':'radio_inside_closed_container',
                  'target':target.name,'container':container.name,'privileged_setup_only':True,
                  'placement_sampler':'official Inside.set_value',
                  'sampled_position':self._array(sampled_position).tolist(),
                  'sampled_orientation':self._array(sampled_orientation).tolist(),
                  'initial_setup_drawer_frame_transform':carried_by,
                  'task_goal_modified':False, 'independent_penetration_audit':False})

    def _setup_initial_heading(self, state):
        offset = float(self.config.get('initial_yaw_offset_deg', 0))
        robot = state.env_accessor.robot
        position, quaternion = robot.get_position_orientation()
        rotated = (self.Rotation.from_euler('z', offset, degrees=True) *
                   self.Rotation.from_quat(self._array(quaternion))).as_quat()
        position = position.clone()
        position[0] += float(self.config.get('initial_offset_x',0))
        position[1] += float(self.config.get('initial_offset_y',0))
        if offset or self.config.get('initial_offset_x') or self.config.get('initial_offset_y'):
            robot.set_position_orientation(position=position, orientation=self.th.as_tensor(rotated,dtype=self.th.float32))
            robot.keep_still()
        save_json(self.output_dir/'initial_heading_setup.json', {
            'initial_setup_only':True, 'yaw_offset_deg':offset,
            'position':self._array(position).tolist(),'original_orientation':self._array(quaternion).tolist(),
            'requested_orientation':rotated.tolist()})

    def _validate_fixture(self, state):
        if self.config.get('scenario_spec'):
            from .ood99 import validate
            return validate(self, state)
        if self.config.get('ood') != 'radio_inside_closed_container':
            return super()._validate_fixture(state)
        target, container = self._fixture
        inside = bool(target.states[self.Inside].get_value(container))
        opened = bool(container.states[self.Open].get_value())
        lower, upper = [self._array(v) for v in container.aabb]
        t_lower, t_upper = [self._array(v) for v in target.aabb]
        bbox_contained = bool(np.all(t_lower >= lower-0.005) and np.all(t_upper <= upper+0.005))
        toggled = bool(target.states[self.ToggledOn].get_value())
        report = {'inside':inside, 'open':opened,'full_aabb_containment':bbox_contained,
                  'target_toggled_on':toggled,'independent_penetration_audit':False,
                  'target_position':self._array(target.get_position_orientation()[0]).tolist(),
                  'target_aabb':[t_lower.tolist(),t_upper.tolist()],
                  'container_aabb':[lower.tolist(),upper.tolist()]}
        save_json(self.output_dir/'container_validation.json',report)
        if not inside or opened or not bbox_contained or toggled:
            raise RuntimeError('Closed-container fixture did not survive physics settling')
