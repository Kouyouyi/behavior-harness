from pathlib import Path
from collections import Counter
import json,numpy as np,pandas as pd
from training_data import ROOT,STATE_IDX
OUT=Path('/cs/kyy/behavior_ap_experiment');rng=np.random.default_rng(20261016)
status=json.loads((OUT/'reports/semantic_index_status.json').read_text());assert status['status']=='complete' and status['files']==955 and not status['errors'],status
counts=Counter();splits={};aa=[];ss=[];preview=[]
for p in sorted((OUT/'index').glob('*/*.parquet')):
 d=pd.read_parquet(p)
 if not len(d):continue
 d['motion_class']=d.motion_tags.apply(lambda x:'+'.join(x))
 for name,g in d.groupby('split'):splits.setdefault(name,[]).append(g)
 train=d[d.split=='train'];counts.update(train.motion_class)
 if not len(train):continue
 q=train.sample(min(32,len(train)),random_state=20261016)
 source=pd.read_parquet(ROOT/q.data_path.iloc[0],columns=['action','observation.state'])
 for r in q.itertuples():
  n=min(32,r.frame_end_exclusive-r.frame_start);aa.extend(source.action.iloc[r.row_start:r.row_start+n].tolist());ss.append(np.asarray(source['observation.state'].iloc[r.row_start])[STATE_IDX])
 if len(preview)<30:preview.extend(q[['sample_id','task_name','instruction','motion_class','split']].head(2).to_dict('records'))
(OUT/'training').mkdir(exist_ok=True);summary={}
for name,parts in splits.items():
 d=pd.concat(parts,ignore_index=True);d.to_parquet(OUT/f'training/{name}.parquet',index=False);summary[name]={'samples':len(d),'episodes':int(d.episode_index.nunique()),'tasks':int(d.task_index.nunique()),'classes':d.motion_class.value_counts().to_dict()}
train=pd.read_parquet(OUT/'training/train.parquet',columns=['episode_index','task_name']);train_eps=set(train.episode_index);assert not train.task_name.str.contains('radio').any()
for name in ['validation','test','radio_task_holdout']:
 other=pd.read_parquet(OUT/f'training/{name}.parquet',columns=['episode_index']);assert not(train_eps&set(other.episode_index))
stats={}
for name,values in [('state',ss),('action',aa)]:
 x=np.asarray(values,dtype=np.float32);assert np.isfinite(x).all();stats[name]={'q01':np.quantile(x,.01,axis=0).tolist(),'q99':np.quantile(x,.99,axis=0).tolist(),'mean':x.mean(0).tolist(),'std':x.std(0).tolist(),'count':len(x)}
(OUT/'training/norm_stats.json').write_text(json.dumps(stats,indent=2));(OUT/'reports/training_split_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(OUT/'reports/instruction_examples.json').write_text(json.dumps(preview,ensure_ascii=False,indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a!='classes'} for k,v in summary.items()},ensure_ascii=False),flush=True)
