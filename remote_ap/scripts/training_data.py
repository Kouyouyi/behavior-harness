from pathlib import Path
from collections import OrderedDict
import json,os
os.environ.setdefault("OPENCV_FFMPEG_THREADS","1")
import cv2,numpy as np,pandas as pd,torch
import pyarrow.parquet as pq
cv2.setNumThreads(1)
from torch.utils.data import Dataset
ROOT=Path(os.environ.get('AP_RAW_DATA_ROOT','/cs/datasets/behavior2026_dual_language'))
STATE_IDX=[0,1,2,53,54,55,56,3,4,5,6,7,8,9,24,25,28,29,30,31,32,33,34,49,50]
IMAGE_KEYS=['observation.images.base_0_rgb','observation.images.left_wrist_0_rgb','observation.images.right_wrist_0_rgb']
VIDEO_KEYS=['observation.rgb.zed_link_camera_0','observation.rgb.left_realsense_link_camera_0','observation.rgb.right_realsense_link_camera_0']
def load_index(root,split):
 parts=[]
 for p in sorted(Path(root).glob('*/*.parquet')):
  d=pd.read_parquet(p);d=d[d.split==split]
  if len(d):parts.append(d)
 if not parts:raise ValueError('No index rows for '+split)
 return pd.concat(parts,ignore_index=True)
class APDataset(Dataset):
 def __init__(self,index,stats,tokenizer,horizon=32,language='goal'):
  self.index=index.reset_index(drop=True);self.stats=stats;self.tokenizer=tokenizer;self.horizon=horizon;self.language=language;self.tables=OrderedDict();self.caps=OrderedDict()
 def __len__(self):return len(self.index)
 def table(self,path,start,horizon):
  first=start//10000;last=(start+horizon-1)//10000;key=(path,first,last)
  if key not in self.tables:
   pf=pq.ParquetFile(ROOT/path)
   # Converted v3 files use exactly 10,000 rows per group except the last group.
   assert all(pf.metadata.row_group(i).num_rows==10000 for i in range(first))
   d=pf.read_row_groups(list(range(first,min(last+1,pf.num_row_groups))),columns=['episode_index','frame_index','action','observation.state','annotation.valid']).to_pandas()
   self.tables[key]=(np.stack(d.action),np.stack(d['observation.state']),d.episode_index.to_numpy(),d.frame_index.to_numpy(),d['annotation.valid'].to_numpy(),first*10000)
   while len(self.tables)>4:self.tables.popitem(last=False)
  self.tables.move_to_end(key);return self.tables[key]
 def image(self,path,seconds):
  if path not in self.caps:
   self.caps[path]=cv2.VideoCapture(str(ROOT/path))
   while len(self.caps)>6:self.caps.popitem(last=False)[1].release()
  self.caps.move_to_end(path);cap=self.caps[path];cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000);ok,im=cap.read()
  if not ok:raise RuntimeError(f'Failed video frame {path} {seconds}')
  im=cv2.cvtColor(cv2.resize(im,(224,224)),cv2.COLOR_BGR2RGB)
  return torch.from_numpy(im.copy()).permute(2,0,1).float()/255
 def normalize(self,x,key):
  s=self.stats[key];lo=np.asarray(s['q01'],np.float32);hi=np.asarray(s['q99'],np.float32)
  return 2*(x-lo)/np.maximum(hi-lo,1e-6)-1
 def __getitem__(self,i):
  r=self.index.iloc[i];n=min(self.horizon,int(r.frame_end_exclusive-r.frame_start));actions,states,episodes,frames,valid,offset=self.table(r.data_path,int(r.row_start),n);start=int(r.row_start)-offset
  assert int(episodes[start])==int(r.episode_index) and int(frames[start])==int(r.frame_start)
  assert np.all(episodes[start:start+n]==r.episode_index) and valid[start:start+n].all()
  state=self.normalize(states[start,STATE_IDX],'state');a=actions[start:start+n];a=np.concatenate([a,np.repeat(a[-1:],self.horizon-n,axis=0)])
  a=self.normalize(a,'action');mask=np.arange(self.horizon)<n
  prompt=r.instruction if self.language=='goal' else r.instruction.split('，')[0]+'。'
  bins=np.digitize(np.clip(state,-1,1),np.linspace(-1,1,257)[:-1])-1
  text='Task: '+prompt+', State: '+' '.join(map(str,bins))+';\nAction: '
  tokens=self.tokenizer(text,add_special_tokens=True,max_length=200,padding='max_length',truncation=True,return_tensors='pt')
  sample={'observation.state':torch.from_numpy(state.astype(np.float32)),'action':torch.from_numpy(a.astype(np.float32)),'action_loss_mask':torch.from_numpy(mask),'observation.language.tokens':tokens['input_ids'][0],'observation.language.attention_mask':tokens['attention_mask'][0].bool()}
  videos=json.loads(r.video_segments_json)
  for key,vkey in zip(IMAGE_KEYS,VIDEO_KEYS):
   v=videos[vkey];sample[key]=self.image(v['path'],v['start_seconds'])
  return sample
