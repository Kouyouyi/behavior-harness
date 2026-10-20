import os,signal,time,subprocess,sys
from pathlib import Path
script="/cs/kyy/behavior_ap_experiment/scripts/champion_suite_server.py"
for p in Path("/proc").iterdir():
 if not p.name.isdigit():continue
 try:args=(p/"cmdline").read_bytes().split(b"\0")
 except OSError:continue
 if script.encode() in args:
  os.kill(int(p.name),signal.SIGTERM)
  for _ in range(100):
   if not (p/"cmdline").exists() or not (p/"cmdline").read_bytes():break
   time.sleep(.1)
  else:raise RuntimeError("Previous champion server did not exit")
log=open("/cs/kyy/behavior_ap_experiment/reports/champion_suite_service.log","ab",buffering=0)
p=subprocess.Popen(["bash","/cs/kyy/behavior_ap_experiment/scripts/start_champion_suite.sh"],env=dict(os.environ,CHAMPION_TASK_ID=sys.argv[1]),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
