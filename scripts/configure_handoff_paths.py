"""Explicit path relocation for this handoff copy. Dry run by default."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--python',required=True,help='Local simulator Python executable');p.add_argument('--assets',required=True,help='Licensed BEHAVIOR asset root');p.add_argument('--remote-root',required=True);p.add_argument('--local-user',required=True);p.add_argument('--ssh-host',default='huoshan');p.add_argument('--write',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parents[1]
replacements={'/media/cody/软件/behavior_harness':str(r),'/home/cody/miniconda3/envs/behavior-2026/bin/python':a.python,'/root/download/datasets/BEHAVIOR-1K':a.assets,'/cs/kyy/behavior_ap_experiment':a.remote_root,"'cody'":repr(a.local_user),'"cody"':'"'+a.local_user+'"',"'huoshan'":repr(a.ssh_host),'ssh huoshan':'ssh '+a.ssh_host,'ssh -t huoshan':'ssh -t '+a.ssh_host}
for folder in ['ap_harness','runtime','scripts','configs','remote']:
 for f in (r/folder).rglob('*'):
  if not f.is_file() or f.suffix not in ['.py','.sh','.json'] or f==Path(__file__).resolve():continue
  s=f.read_text();t=s
  for old,new in replacements.items():t=t.replace(old,new)
  if s!=t:
   print(f.relative_to(r))
   if a.write:f.write_text(t)
print('Review git diff after --write. Other machine-specific paths require manual review (planner CLI, remote models/envs, DISPLAY/XAUTHORITY).')
