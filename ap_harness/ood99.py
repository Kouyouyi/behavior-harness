"""Private, setup-only per-task perturbations and category-equivalent scoring."""
import copy
import json
from pathlib import Path
import numpy as np
from .control import save_json
from .scoring import RenderedTargetScorer, evaluate_approach

def setup(backend, state):
    from omnigibson.object_states import OnTop, Inside, Open
    spec = json.loads(Path(backend.config['scenario_spec']).read_text())
    if spec['task_name'] != backend.task_name:
        raise ValueError('Scenario task mismatch')
    backend.suite_spec = spec
    scope = state.env_accessor.object_scope
    world = copy.deepcopy(backend.og.sim.dump_state(serialized=False))
    targets = [scope[p['key']] for p in spec['placements']]
    original = {t.name: backend._array(t.get_position_orientation()[0]).copy() for t in targets}
    robot_origin=backend._array(state.env_accessor.robot.get_position_orientation()[0]).copy()
    cluster_bearing=None
    closed = {}
    records = []
    try:
        # Open enclosing articulated objects before sampling original relations.
        for placement in spec['placements']:
            for ancestor in placement['ancestors']:
                obj = scope[ancestor['object_key']]
                if ancestor['relation'] == 'inside' and Open in obj.states:
                    if obj.name not in closed:
                        closed[obj.name] = obj
                        if not obj.states[Open].set_value(True, fully=True):
                            raise RuntimeError('Cannot open setup ancestor: ' + obj.name)
        for placement, target in zip(spec['placements'], targets):
            support = scope[placement['support_key']]
            relation = {'ontop': OnTop, 'inside': Inside}[placement['relation']]
            if relation not in target.states:
                raise ValueError('Target does not implement original relation')
            found = False
            for attempt in range(spec['placement_attempts']):
                if target.states[relation].set_value(support, True):
                    pos = backend._array(target.get_position_orientation()[0])
                    delta=pos-robot_origin
                    bearing=float(np.arctan2(delta[1],delta[0]))
                    clustered=True
                    if spec.get('floor_target_cluster_half_angle_deg') is not None and placement['support_key'].startswith('floor.'):
                        if np.linalg.norm(delta[:2])<spec.get('minimum_initial_distance_m',1.25):clustered=False
                        if cluster_bearing is not None:
                            error=abs(np.arctan2(np.sin(bearing-cluster_bearing),np.cos(bearing-cluster_bearing)))
                            clustered=clustered and error<=np.deg2rad(spec['floor_target_cluster_half_angle_deg'])
                    if clustered and np.linalg.norm((pos-original[target.name])[:2]) >= spec['minimum_relocation_m']:
                        if cluster_bearing is None:cluster_bearing=bearing
                        found = True
                        break
            if not found:
                raise RuntimeError('No novel collision-checked placement on original support: ' + target.name)
            records.append({'key':placement['key'], 'object_name':target.name,
                            'support_name':support.name, 'relation':placement['relation'],
                            'original_position':original[target.name].tolist(), 'sampler_attempts':attempt+1})
        # Closing articulated drawers teleports links. Carry all sampled targets
        # and their supports in the same enclosing moving-link frame at setup only.
        for container in closed.values():
            moving = []
            for obj in state.env_accessor.scene.objects:
                if obj is container or obj is state.env_accessor.robot: continue
                if Inside not in getattr(obj, 'states', {}): continue
                if not obj.states[Inside].get_value(container): continue
                pos, quat = [backend._array(x).copy() for x in obj.get_position_orientation()]
                enclosing = []
                for name, link in container.links.items():
                    low, high = [backend._array(v) for v in link.aabb]
                    if np.all(pos >= low-.005) and np.all(pos <= high+.005):
                        lp, lq = [backend._array(x).copy() for x in link.get_position_orientation()]
                        enclosing.append((float(np.prod(high-low)), name, link, lp, lq))
                moving.append((obj,pos,quat,sorted(enclosing,key=lambda x:x[0])))
            if not container.states[Open].set_value(False, fully=True):
                raise RuntimeError('Cannot close setup ancestor: '+container.name)
            for obj,pos,quat,enclosing in moving:
                for _,name,link,lp,lq in enclosing:
                    cp,cq = [backend._array(x) for x in link.get_position_orientation()]
                    delta=backend.Rotation.from_quat(cq)*backend.Rotation.from_quat(lq).inv()
                    if np.linalg.norm(cp-lp)<1e-4 and delta.magnitude()<1e-4: continue
                    obj.set_position_orientation(position=backend.th.tensor(cp+delta.apply(pos-lp),dtype=backend.th.float32),
                         orientation=backend.th.tensor((delta*backend.Rotation.from_quat(quat)).as_quat(),dtype=backend.th.float32))
                    obj.keep_still()
                    break
        # Save only intended entities. Restore unrelated scene and robot state.
        changed = {t.name:t for t in targets}
        changed.update(closed)
        for placement in spec['placements']:
            for ancestor in placement['ancestors']:
                obj=scope[ancestor['object_key']]
                if any(Inside in obj.states and obj.states[Inside].get_value(c) for c in closed.values() if c is not obj):
                    changed[obj.name]=obj
        # Carry only kinematics back, not temperature/cooked/dirty/etc. states
        # that may have evolved while the official sampler advanced physics.
        snapshots={name:{'pose':[v.clone() for v in obj.get_position_orientation()],
                         'joints':obj.get_joint_positions().clone() if obj.n_joints else []} for name,obj in changed.items()}
    finally:
        backend.og.sim.load_state(world, serialized=False)
    for name,obj in changed.items():
        obj.set_position_orientation(position=snapshots[name]['pose'][0],orientation=snapshots[name]['pose'][1])
        if len(snapshots[name]['joints']):obj.set_joint_positions(snapshots[name]['joints'])
        obj.keep_still()
    backend.suite_targets=targets
    backend.suite_closed=closed
    backend.suite_records=records
    save_json(backend.output_dir/'ood_setup_private.json', {'task':backend.task_name,'targets':records,
              'closed_containers':list(closed),'mutation_phase':'initialization_only',
              'original_relations_preserved':True,'sampler':'official object state sampler',
              'minimum_relocation_m':spec['minimum_relocation_m'],
              'independent_mesh_penetration_depth_verified':False})

def validate(backend, state):
    from omnigibson.object_states import OnTop, Inside, Open
    scope=state.env_accessor.object_scope
    checks=[]
    for p,t,r in zip(backend.suite_spec['placements'],backend.suite_targets,backend.suite_records):
        relation={'ontop':OnTop,'inside':Inside}[p['relation']]
        support=scope[p['support_key']]
        position=backend._array(t.get_position_orientation()[0])
        displacement=float(np.linalg.norm((position-np.asarray(r['original_position']))[:2]))
        valid=bool(t.states[relation].get_value(support)) and np.isfinite(position).all() and displacement>=backend.suite_spec['minimum_relocation_m']
        chain_checks=[]
        child=t
        for ancestor in p['ancestors']:
            parent=scope[ancestor['object_key']]
            predicate={'ontop':OnTop,'inside':Inside}[ancestor['relation']]
            holds=bool(child.states[predicate].get_value(parent))
            chain_checks.append({'child':child.name,'parent':parent.name,'relation':ancestor['relation'],'holds':holds})
            valid=valid and holds
            child=parent
        checks.append({'key':p['key'],'relation_holds':bool(t.states[relation].get_value(support)),
                       'position':position.tolist(),'displacement_m':displacement,'valid':bool(valid),'ancestor_relations':chain_checks})
    closed={name:not bool(obj.states[Open].get_value()) for name,obj in backend.suite_closed.items()}
    result={'targets':checks,'containers_closed':closed,'valid':all(c['valid'] for c in checks) and all(closed.values())}
    save_json(backend.output_dir/'ood_geometry_private.json',result)
    if not result['valid']: raise RuntimeError('OOD placement failed post-settling validation')

def face_away(sim):
    b=sim.backend
    pos,quat=b.robot.get_position_orientation()
    deltas=[b._array(t.get_position_orientation()[0])-b._array(pos) for t in b.suite_targets]
    bearings=np.asarray([np.arctan2(d[1],d[0]) for d in deltas])
    candidates=np.linspace(-np.pi,np.pi,360,endpoint=False)
    separations=np.abs(np.arctan2(np.sin(candidates[:,None]-bearings),np.cos(candidates[:,None]-bearings)))
    yaw=float(candidates[np.argmax(separations.min(axis=1))])
    current=b.Rotation.from_quat(b._array(quat)).as_euler('xyz')[2]
    b.config['initial_yaw_offset_deg']=float(np.rad2deg(yaw-current))

class TargetSetScorer:
    def __init__(self,robot,targets,camera_names):
        self.scorers=[RenderedTargetScorer(robot,t,camera_names) for t in targets]
    def enable(self): self.scorers[0].enable()
    def __call__(self):
        rows=[{'object_name':s.target.name,**evaluate_approach(s())} for s in self.scorers]
        # Visibility and distance must belong to the SAME physical object.
        best=min(rows,key=lambda r:(not r['find_and_approach_success'],not r['target_visible'],r['horizontal_distance_m']))
        return {**best,'equivalent_target_evidence':rows,'accepts_any_same_class_target':True}

def mesh_pixels(mask, labels, prim_path):
    mask=np.asarray(mask)
    if mask.ndim!=2 or mask.dtype.kind not in 'iu': raise ValueError('Invalid rendered mesh instance mask')
    if not labels or not all(isinstance(p,str) for p in labels.values()):raise ValueError('Missing renderer prim-path mapping')
    mapping={int(k):v for k,v in labels.items()}
    unknown=set(map(int,np.unique(mask)))-set(mapping)-{0,1}
    if unknown:raise ValueError('Unmapped renderer instance IDs')
    if not any(p.startswith('/World/') for p in mapping.values()):raise ValueError('Renderer mapping has no world prim paths')
    ids=[i for i,p in mapping.items() if p==prim_path or p.startswith(prim_path.rstrip('/')+'/')]
    if not ids: raise ValueError('Target mesh absent from renderer mapping: '+prim_path)
    n=int(np.isin(mask,ids).sum())
    return {'pixel_count':n,'visible':n>=16,'image_pixels':int(mask.size),'pixel_fraction':n/mask.size,'threshold_pixels':16,'matched_renderer_ids':ids,'target_prim_path':prim_path}

class RawMeshTargetSetScorer:
    """Exact rendered object pixels without Isaac's crashing semantic host map.

    The installed Replicator exposes renderer mesh IDs and USD prim-path tokens
    directly. Object identity is matched by full USD subtree, never a substring.
    """
    def __init__(self,robot,targets,camera_names):
        self.robot,self.targets=robot,targets
        self.sensors={role:robot.sensors[name.split('::',1)[1]] for role,name in camera_names.items()}
        self.annotators={};self.raw={}
    def enable(self):
        import omnigibson as og
        import omni.replicator.core as rep
        for role,sensor in self.sensors.items():
            with og.sim.editing_usd():
                annotator=rep.AnnotatorRegistry.get_annotator('instance_id_segmentation_fast',device='cpu')
                annotator.attach([sensor._render_product])
            self.annotators[role]=annotator
            for _ in range(4):og.sim.render()
    def __call__(self):
        from .scoring import array
        self.raw={role:a.get_data() for role,a in self.annotators.items()}
        rows=[]
        for target in self.targets:
            cameras={role:mesh_pixels(v['data'],v['info']['idToLabels'],target.prim_path) for role,v in self.raw.items()}
            row={'object_name':target.name,'base_position':array(self.robot.get_position_orientation()[0]).tolist(),
                 'target_position':array(target.get_position_orientation()[0]).tolist(),
                 'target_visible':any(c['visible'] for c in cameras.values()),
                 'visibility_evidence':{'method':'rendered_mesh_instance_ids_exact_USD_object_subtree','annotator':'instance_id_segmentation_fast','cameras':cameras}}
            rows.append(evaluate_approach(row))
        best=min(rows,key=lambda r:(not r['find_and_approach_success'],not r['target_visible'],r['horizontal_distance_m']))
        return {**best,'equivalent_target_evidence':rows,'accepts_any_same_class_target':True}
    def save_raw(self,directory):
        directory=Path(directory);directory.mkdir(exist_ok=True)
        for role,v in self.raw.items():
            np.save(directory/(role+'_mesh_ids.npy'),v['data'])
            save_json(directory/(role+'_id_to_prim.json'),v['info']['idToLabels'])

def fingerprint(sim):
    b=sim.backend
    objects={t.name:[b._array(v).tolist() for v in t.get_position_orientation()] for t in getattr(b,'suite_targets',[sim.target])}
    return {'task_name':b.task_name,'instance_id':b.instance_id,'objects':objects,
            'robot_pose':[b._array(v).tolist() for v in b.robot.get_position_orientation()]}

def match_fingerprint(current, expected, tolerance=.02):
    if current['task_name']!=expected['task_name'] or current['instance_id']!=expected['instance_id'] or set(current['objects'])!=set(expected['objects']):
        raise ValueError('Preflight scene identity mismatch')
    for name,pose in {**current['objects'],'robot':current['robot_pose']}.items():
        old=expected['robot_pose'] if name=='robot' else expected['objects'][name]
        if np.linalg.norm(np.asarray(pose[0])-old[0])>tolerance:
            raise ValueError('Preflight pose mismatch: '+name)
        if min(np.linalg.norm(np.asarray(pose[1])-old[1]),np.linalg.norm(np.asarray(pose[1])+old[1]))>.02:
            raise ValueError('Preflight orientation mismatch: '+name)
