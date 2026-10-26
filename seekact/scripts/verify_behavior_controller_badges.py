#!/usr/bin/env python3
"""Decode every badge frame and compare its color class to acknowledged ownership."""
import argparse,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT)]
from annotate_behavior_controller_videos import read,action_owners,save,sha

def main():
 import av
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output.resolve()
 receipts=sorted(root.glob('*/zero_shot_*/video_annotation.json'));results=[]
 if not receipts:raise ValueError('No verified annotation receipts')
 for receipt in receipts:
  meta=read(receipt);directory=receipt.parent;owners=action_owners(directory/'execution.jsonl',meta['steps'])
  for role,item in meta['videos'].items():
   path=Path(item['annotated_path']);counts=Counter();count=0
   if sha(path)!=item['annotated_sha256']:raise ValueError('Annotated bytes changed')
   with av.open(str(path))as container:
    for frame in container.decode(video=0):
     color=frame.to_ndarray(format='rgb24')[8:44,0:4].mean(axis=(0,1))
     expected=owners[count]['controller']
     actual='ap_vla'if color[2]-color[0]>80 else 'champion_vla'if color[0]-color[2]>80 else 'native_tools'if color[0]-color[1]>15 and color[2]-color[1]>30 else 'unattributed'
     if actual!=expected:raise ValueError(f'Wrong controller badge at{path} frame{count}: {actual} != {expected}')
     counts[actual]+=1;count+=1
   if count!=len(owners):raise ValueError('Decoded badge count mismatch')
   results.append({'task':directory.parent.name,'role':role,'frames':count,'decoded_badge_counts':dict(counts),'all_frames_match_acknowledged_controller':True,'sha256':item['annotated_sha256']})
   print(json.dumps(results[-1]),flush=True)
 save(root/'controller_badge_frame_verification.json',{'status':'verified','videos':len(results),'every_frame_checked':True,'results':results})
if __name__=='__main__':main()
