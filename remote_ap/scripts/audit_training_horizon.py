"""Audit all indexed supervision horizons without changing the live training dataset."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd,pyarrow.parquet as pq
from direction_metrics import DIRECTIONS
ROOT=Path("/cs/kyy/behavior_ap_experiment");DATA=Path("/cs/datasets/behavior2026_dual_language");OUT=ROOT/"evaluation/horizon_audit";OUT.mkdir(exist_ok=True)
cols=["sample_id","data_path","row_start","frame_start","frame_end_exclusive","motion_tags"]
frames=[]
for split in ("train","validation","test","radio_task_holdout"):
 d=pd.read_parquet(ROOT/f"training/{split}.parquet",columns=cols);d["split"]=split;frames.append(d)
df=pd.concat(frames,ignore_index=True);groups=list(df.groupby("data_path"));started=time.time()
for file_i,(path,group) in enumerate(groups):
 dest=OUT/(path.replace("/","_")+".parquet")
 if dest.exists():continue
 pf=pq.ParquetFile(DATA/path);records=[]
 # Row groups are decoded once per source file. Cache only base command axes.
 cache={}
 for r in group.itertuples():
  n=min(32,int(r.frame_end_exclusive-r.frame_start));start=int(r.row_start);parts=[]
  for g in range(start//10000,(start+n-1)//10000+1):
   if g not in cache:cache[g]=np.stack(pf.read_row_group(g,columns=["action"]).column(0).to_pylist())[:,:3]
   parts.append(cache[g][max(0,start-g*10000):min(len(cache[g]),start+n-g*10000)])
  a=np.concatenate(parts);tags=[t for t in r.motion_tags if t in DIRECTIONS];record={"sample_id":r.sample_id,"split":r.split,"motion_class":"+".join(r.motion_tags),"base_direction_applicable":bool(tags)}
  for h in (8,32):
   mean=a[:h].mean(0);signed=[float(mean[DIRECTIONS[t][0]]*DIRECTIONS[t][1]) for t in tags]
   record[f"h{h}_all_aligned"]=all(v>.02 for v in signed) if tags else None
   record[f"h{h}_any_opposed"]=any(v<-.02 for v in signed) if tags else None
   record[f"h{h}_min_signed_command"]=min(signed) if tags else None
  records.append(record)
 pd.DataFrame(records).to_parquet(dest,index=False)
 status={"status":"running","files_done":file_i+1,"files_total":len(groups),"elapsed_s":time.time()-started}
 (OUT/"status.json").write_text(json.dumps(status));print(json.dumps(status),flush=True)
result=pd.concat([pd.read_parquet(p) for p in OUT.glob("*.parquet")],ignore_index=True)
summary={}
for (split,cls),group in result.groupby(["split","motion_class"]):
 eligible=group[group.base_direction_applicable]
 summary[split+"/"+cls]={"samples":len(group),"applicable":len(eligible),**{key:float(eligible[key].astype(float).mean()) if len(eligible) else None for key in ("h8_all_aligned","h32_all_aligned","h8_any_opposed","h32_any_opposed")}}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));(OUT/"status.json").write_text(json.dumps({"status":"complete","samples":len(result),"files":len(groups)}))
