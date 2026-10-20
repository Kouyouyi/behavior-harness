"""Wait for external experiments and five minutes of sustained GPU availability."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/champion_start_comparison_20261020_v3';OUT.mkdir(exist_ok=True)
lock=(OUT/'waiter.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
clear_since=None
while True:
 result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
 busy=[int(s.strip()) for s in result.stdout.splitlines() if s.strip().isdigit()]
 memory=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
 free=int(memory.stdout.strip().splitlines()[0]) if memory.returncode==0 else 0
 external=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  except OSError:continue
  if args and 'python' in Path(args[0]).name and any(any(k in a for k in ['/home/cody/rax-livingroom-native/','/home/cody/repo/HumanoidHarness','/home/cody/repo/HumanoidArena/','/home/cody/repo/gwp05','services.livingroom.server','compare-rax-harness']) for a in args[1:]):external.append(int(p.name))
 available=result.returncode==0 and not busy and not external and free>=20000
 if not available:clear_since=None
 elif clear_since is None:clear_since=time.monotonic()
 stable=0 if clear_since is None else time.monotonic()-clear_since
 (OUT/'waiting_status.json').write_text(json.dumps({'status':'waiting_for_external_experiments','gpu_pids':busy,'external_experiment_pids':external,'gpu_free_mib':free,'idle_seconds':round(stable),'required_idle_seconds':300,'does_not_stop_other_tasks':True}))
 if stable>=300:break
 time.sleep(15)
env=os.environ.copy();env['CHAMPION_COMPARISON_OUTPUT']='outputs/champion_start_comparison_20261020_v3'
(OUT/'waiting_status.json').write_text(json.dumps({'status':'gpu_released_starting_comparison','idle_seconds':300}))
with (OUT/'runner.log').open('w') as log:
 result=subprocess.run(['/home/cody/miniconda3/envs/behavior-2026/bin/python','-u','scripts/run_champion_start_comparison.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
(OUT/'supervisor_exit.json').write_text(json.dumps({'returncode':result.returncode}))
