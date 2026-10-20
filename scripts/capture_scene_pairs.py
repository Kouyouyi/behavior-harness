import sys,json,copy
from pathlib import Path
import numpy as np
import cv2
from ap_harness.simulator import Simulator
from ap_harness import ood99
from ap_harness.control import save_json
root=Path('/media/cody/软件/behavior_harness'); task=sys.argv[1]
out=root/'outputs/scene_pairs_20261020_v2'/task
spec=root/'configs/ood99'/f'{task}.json';cfg=json.loads(spec.read_text())
sim=Simulator(out/'sim',task_name=task,target_key=cfg['target_key'],seed=cfg['seed'])
sim.start();b=sim.backend
from omnigibson.object_states import Open
scope=b.state.env_accessor.object_scope
container=scope[cfg['placements'][0]['support_key']]
targets=[scope[p['key']] for p in cfg['placements']]
ref=np.mean([b._array(t.get_position_orientation()[0]) for t in targets],axis=0)
robot=b._array(b.robot.get_position_orientation()[0]);direction=robot-ref;direction[2]=0;direction/=np.linalg.norm(direction)
eye=ref+direction*(1.8 if task=='can_meat' else 3.8);eye[2]=2.1 if task=='can_meat' else 2.7
if task=='carrying_in_groceries': eye=np.array([-6.389077,8.185542,2.15])
forward=ref-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
quat=b.Rotation.from_matrix(np.column_stack([right,up,-forward])).as_quat()
def shot(name):
 cam=b.og.sim.viewer_camera;cam.focal_length=16.;cam.set_position_orientation(position=eye.tolist(),orientation=quat.tolist())
 for _ in range(8):b.og.sim.render()
 obs,_=cam.get_obs();rgb=b._array(obs['rgb'])[...,:3].astype(np.uint8)
 cv2.imwrite(str(out/(name+'.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
def record():return {'container_open':bool(container.states[Open].get_value()),'targets':{t.name:b._array(t.get_position_orientation()[0]).tolist() for t in targets}}
original=record();shot('original')
b.config['scenario_spec']=str(spec);ood99.setup(b,b.state)
shot('ood');changed=record()
save_json(out/'comparison.json',{'original':original,'ood':changed,'camera_position':eye.tolist(),'camera_orientation':quat.tolist(),'note':'Fresh visualization reconstruction; no policy actions. Same fixed observer camera, before and after OOD setup.','task_steps':sim.steps})
b.close('visualization_only');print('PAIR_COMPLETE',flush=True)
b.og.shutdown()
