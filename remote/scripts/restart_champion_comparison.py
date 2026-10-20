import os,signal,time,subprocess
from pathlib import Path
script="/cs/kyy/behavior_ap_experiment/scripts/champion_server.py"
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
log=open("/cs/kyy/behavior_ap_experiment/reports/champion_comparison_service.log","ab",buffering=0)
p=subprocess.Popen(["bash","/cs/kyy/behavior_ap_experiment/scripts/start_champion.sh"],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
