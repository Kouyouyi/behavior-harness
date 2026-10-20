"""Preserve segment evidence and deliver verified full-process videos for both conditions."""
import json,time,subprocess,fcntl,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/champion_start_comparison_20261020_v3'
lock=(OUT/'finalizer.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (OUT/'audit_repair_complete.json').exists():time.sleep(15)
data=json.loads((OUT/'summary.json').read_text())
assert data['active']=='complete'
def probe(p):
 r=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_streams','-of','json',str(p)],capture_output=True,text=True,check=True)
 s=json.loads(r.stdout)['streams'][0]
 assert s['codec_name']=='h264' and s['pix_fmt']=='yuv420p'
 subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],capture_output=True,check=True)
 return {'frames':int(s['nb_read_frames']),'width':s['width'],'height':s['height'],'fps':s['avg_frame_rate'],'duration':s.get('duration'),'fully_decoded':True}
report={'scene':'radio_on_tv_cabinet_facing_room','seed':20261020,'yaw':180,'budget_each':10000,'scope':'one unfamiliar-position case; fresh initial scene for both; not a population success rate','groups':[]}
video_dir=OUT/'full_process_videos';video_dir.mkdir(exist_ok=True)
for condition in ['harness','direct']:
 candidates=[r for r in data['conditions'] if r['condition']==condition]
 valid=[r for r in candidates if r.get('valid_evaluation')]
 row=(valid[0] if valid else candidates[-1]).copy()
 row['videos']={};segments=[]
 for key in ['search_run','completion_run']:
  if row.get(key):segments.append(OUT/row[key])
 row['vla_calls']={};row['planner_calls']=0
 for seg in segments:
  rounds=list((seg/'control').glob('round_*'))
  row['vla_calls'][seg.name]=sum((r/'predicted_actions.npy').exists() for r in rounds)
  s=json.loads((seg/'control/summary.json').read_text()) if (seg/'control/summary.json').exists() else {}
  row['planner_calls']+=s.get('planner_calls',0)
 for camera in ['third_person','robot_cameras']:
  sources=[seg/'sim/videos'/f'{camera}.mp4' for seg in segments]
  if not all(p.exists() for p in sources):
   row['videos'][camera]={'available':False};continue
  proofs=[probe(p) for p in sources]
  assert len({(p['width'],p['height'],p['fps']) for p in proofs})==1
  destination=video_dir/f'{condition}_{camera}.mp4'
  if destination.exists():raise RuntimeError('Refuse overwriting deliverable '+str(destination))
  if len(sources)==1:destination.symlink_to(sources[0])
  else:
   manifest=video_dir/f'{condition}_{camera}.concat.txt'
   manifest.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n" for p in sources))
   subprocess.run(['ffmpeg','-v','error','-f','concat','-safe','0','-i',str(manifest),'-c','copy','-movflags','+faststart',str(destination)],check=True)
  proof=probe(destination)
  assert proof['frames']==sum(p['frames'] for p in proofs),'Concatenated video lost frames'
  row['videos'][camera]={'path':str(destination),'segments':[str(p) for p in sources],'segment_validation':proofs,'full_validation':proof,'note':'A: search then champion, with clean-state process handoff between segments. Simulation steps exclude setup frames.'}
 report['groups'].append(row)
(OUT/'final_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
rows=[]
for r in report['groups']:
 label='Harness＋冠军' if r['condition']=='harness' else '冠军从起点直接控制'
 status='成功开机' if r.get('original_task_success') else ('预算内未开机' if r.get('valid_evaluation') else '未能有效评估')
 links=' · '.join(f'<a href="{Path(v["path"]).relative_to(OUT)}">{camera}</a>' for camera,v in r['videos'].items() if 'path' in v)
 rows.append(f'<tr><td>{label}</td><td>{status}</td><td>{r.get("search_steps",0)}</td><td>{r.get("completion_steps",0)}</td><td>{r.get("total_steps",r.get("search_steps",0))}</td><td>{links}</td></tr>')
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>从起点两组对照</title><h1>从起点：Harness＋冠军 vs 冠军直接控制</h1><p>同场景、同种子、初始目标不可见，各10000步总预算；只代表这一个场景。</p><table border="1" cellpadding="8"><tr><th>组别</th><th>官方结果</th><th>搜索步数</th><th>冠军步数</th><th>总步数</th><th>全过程视频</th></tr>'+''.join(rows)+'</table><p>Harness视频按搜索→冠军执行顺序连接，完整解码与总帧数已验证。误判导致的冗余搜索及初始化失败保留，不算能力失败。</p><a href="final_comparison.json">完整验收与调用统计</a>')
print(json.dumps(report,ensure_ascii=False),flush=True)
