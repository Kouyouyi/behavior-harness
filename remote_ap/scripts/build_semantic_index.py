import os
"""Build goal-conditioned hindsight VLA examples, with explicit provenance and episode splits."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,json,hashlib,time
import numpy as np,pandas as pd,pyarrow.parquet as pq
# labels.py is shipped alongside this script.
from labels import clean_object
ROOT=Path(os.environ['AP_RAW_DATA_ROOT']);OUT=Path(os.environ.get('AP_PROJECT_ROOT',str(Path.cwd()/'ap_work')));META=None
WORDS={'turn_left':'向左转','turn_right':'向右转','forward':'向前移动','backward':'向后移动','strafe_left':'向左平移','strafe_right':'向右平移','torso_left':'向左转腰','torso_right':'向右转腰','raise_view':'抬高观察位置','lower_view':'降低观察位置','open_door':'打开门或柜门','open_drawer':'拉开抽屉','open_lid':'揭开盖子'}
def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
def metadata():
 return {int(r['episode_index']):r for p in (ROOT/'meta/episodes').glob('*/*.parquet') for r in pd.read_parquet(p).to_dict('records')}
def task_split(meta):
 # All radio-task demos held out, to test cross-task transfer; other splits by whole episode.
 if any('radio' in t for t in meta['tasks']):return 'radio_task_holdout'
 value=int(hashlib.sha256(('behavior-ap-v1:'+str(meta['episode_index'])).encode()).hexdigest()[:8],16)%100
 return 'validation' if value<10 else 'test' if value<20 else 'train'
def build(pathstr):
 global META
 if META is None:META=metadata()
 p=Path(pathstr);rel=p.relative_to(ROOT/'data');dest=OUT/'index'/rel;report=OUT/'reports/index_shards'/rel.with_suffix('.json')
 if report.exists() and dest.exists():return json.loads(report.read_text())
 d=pq.read_table(p,columns=['episode_index','frame_index','observation.state','action','observation.robot2cam_pose.zed_link_camera_0','annotation.valid']).to_pandas()
 states=np.stack(d['observation.state']);actions=np.stack(d.action);camera=np.stack(d['observation.robot2cam_pose.zed_link_camera_0']);valid=d['annotation.valid'].to_numpy();eps=d.episode_index.to_numpy()
 offsets={int(eps[i]):int(i) for i in np.r_[0,np.flatnonzero(np.diff(eps))+1]};records=[];skips=Counter();classes=Counter();splits=Counter();anns={}
 segpath=ROOT/'_control/segments/data'/rel.with_suffix('.jsonl')
 for line in segpath.open():
  seg=json.loads(line)
  if not seg['active_perception_candidate']:continue
  ep=seg['episode_index'];lo,hi=seg['frame_range'];a=offsets[ep]+lo;b=offsets[ep]+hi
  if hi-lo<12 or b>=len(d) or eps[b]!=ep or not valid[a:b].all():skips['short_or_invalid']+=1;continue
  m=META[ep]
  if ep not in anns:anns[ep]={int(x['skill_idx']):x for x in json.loads((ROOT/'_source'/m['annotation_path']).read_text())['skill_annotation']}
  skill=anns[ep][seg['skill_index']];objects=[]
  def flat(x):
   if isinstance(x,str):return [x]
   if isinstance(x,list):return [s for y in x for s in flat(y)]
   return []
  objects=list(dict.fromkeys(x for x in flat(skill.get('object_id',[])) if x not in ['left','right']))
  if not objects:skips['missing_target']+=1;continue
  target='、'.join(clean_object(x) for x in objects)
  v=states[a:b,:3].astype(float);dt=1/30;local_dist=v[:,:2].sum(0)*dt;yaw=float(v[:,2].sum()*dt);path=float(np.linalg.norm(v[:,:2],axis=1).sum()*dt);yaw_abs=float(np.abs(v[:,2]).sum()*dt)
  tags=[];verbs=skill.get('skill_description',[]);opening=[{'open door':'open_door','open drawer':'open_drawer','open lid':'open_lid'}[x] for x in verbs if x in ['open door','open drawer','open lid']]
  if abs(yaw)>=np.deg2rad(5):
   if abs(yaw)/max(yaw_abs,1e-9)<.8:skips['reversing_yaw']+=1;continue
   tags.append('turn_left' if yaw>0 else 'turn_right')
  for axis,pos,neg in [(0,'forward','backward'),(1,'strafe_left','strafe_right')]:
   value=local_dist[axis]
   if abs(value)>=.05:
    if abs(value)/max(float(np.abs(v[:,axis]).sum()*dt),1e-9)<.8:skips['reversing_translation']+=1;tags=None;break
    tags.append(pos if value>0 else neg)
  if tags is None:continue
  torso=float(states[b,56]-states[a,56]);height=float(camera[b,2]-camera[a,2]);stationary=path<.02 and yaw_abs<np.deg2rad(2)
  if abs(torso)>np.deg2rad(5) and abs(float(actions[b,6]-actions[a,6]))>np.deg2rad(5):tags.append('torso_left' if torso>0 else 'torso_right')
  if stationary and abs(height)>.05:tags.append('raise_view' if height>0 else 'lower_view')
  if opening:
   # Official opening skill establishes the command; camera/base movement is diagnostic, not a separate mandatory command.
   tags=list(dict.fromkeys(opening));goal_kind='inspect_container_interior';goal='观察'+target+'内部';movement='并'.join({'open_door':'打开'+target,'open_drawer':'拉开'+target,'open_lid':'揭开'+target}[x] for x in tags)
  else:
   goal_kind='inspect_target';goal='观察并接近'+target;movement='并'.join(WORDS[x] for x in tags)
  if not tags:skips['no_supported_motion']+=1;continue
  instruction=movement+'，'+goal+'。'
  split=task_split(m);classes['+'.join(tags)]+=1;splits[split]+=1
  video={}
  for key in ['observation.rgb.zed_link_camera_0','observation.rgb.left_realsense_link_camera_0','observation.rgb.right_realsense_link_camera_0']:
   prefix='videos/'+key;video[key]={'path':f'videos/{key}/chunk-{int(m[prefix+"/chunk_index"]):03d}/file-{int(m[prefix+"/file_index"]):03d}.mp4','start_seconds':float(m[prefix+'/from_timestamp'])+lo/30,'end_seconds':float(m[prefix+'/from_timestamp'])+hi/30}
  records.append({'sample_id':f'{ep}:{lo}:{hi}','episode_index':ep,'task_index':seg['task_index'],'task_name':str(m['tasks'][0]),'frame_start':lo,'frame_end_exclusive':hi,'data_path':str(Path('data')/rel),'row_start':a,'split':split,'instruction':instruction,'motion_tags':tags,'target_object_ids':objects,'target_name':target,'goal_kind':goal_kind,'base_constraint':'stationary' if stationary else 'unconstrained','video_segments_json':json.dumps(video),'motion_evidence_json':json.dumps({'base_local_integral_xy_m':local_dist.tolist(),'base_yaw_rad':yaw,'base_path_m':path,'base_abs_yaw_rad':yaw_abs,'torso4_delta_rad':torso,'camera_height_delta_m':height}),'label_source':'official_skill_target_plus_measured_motion_hindsight','visual_goal_verified':False,'original_skill_index':seg['skill_index']})
 dest.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(records).to_parquet(dest.with_suffix('.part'),index=False);dest.with_suffix('.part').replace(dest)
 reportvalue={'file':str(rel),'samples':len(records),'classes':dict(classes),'splits':dict(splits),'skips':dict(skips)};atomic(report,reportvalue);return reportvalue
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
 META=metadata();files=[ROOT/'data/chunk-017/file-000.parquet'] if args.pilot else sorted((ROOT/'data').glob('*/*.parquet'));counts=Counter();splits=Counter();skips=Counter();errors=[];done=0;started=time.time()
 with ProcessPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(build,str(p)):p for p in files}
  for f in as_completed(futures):
   try:r=f.result();counts.update(r['classes']);splits.update(r['splits']);skips.update(r['skips']);done+=1
   except Exception as e:errors.append({'file':str(futures[f]),'error':repr(e)})
   status={'status':'running','files':done,'total_files':len(files),'samples':sum(counts.values()),'classes':dict(counts),'splits':dict(splits),'skips':dict(skips),'errors':errors,'elapsed_s':time.time()-started}
   atomic(OUT/'reports/semantic_index_status.json',status);print(done,len(files),sum(counts.values()),flush=True)
 status['status']='pilot_complete' if args.pilot and not errors else 'complete' if not errors else 'failed';atomic(OUT/'reports/semantic_index_status.json',status)
