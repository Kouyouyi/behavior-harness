"""Local one-shot continuation after remote real inference verification."""
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
while True:
    try:
        check=subprocess.run(['ssh','-o','ConnectTimeout=10','huoshan','cat /cs/kyy/behavior_ap_experiment/reports/task_vla_inference_verified.json'],capture_output=True,text=True,timeout=30)
    except subprocess.TimeoutExpired:
        time.sleep(30);continue
    if check.returncode==0:
        proof=json.loads(check.stdout)
        if proof.get('remote_inference_verified'):break
    time.sleep(30)
p=ROOT/'configs/task_executor_readiness.json';ready=json.loads(p.read_text())
if not ready.get('cross_process_state_restore_verified'):raise RuntimeError('Restore verification missing')
ready['remote_inference_verified']=True;ready['remote_evidence']=proof;p.write_text(json.dumps(ready,indent=2))
(ROOT/'outputs/ood99_two_stage_20261019/remote_inference_verified.json').write_text(json.dumps(proof,indent=2))
subprocess.run(['ssh','-fN','-o','ControlMaster=no','-o','ControlPath=none','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-L','127.0.0.1:18771:127.0.0.1:18771','huoshan'],check=True)
subprocess.run(['/home/cody/.codex/skills/environment/scripts/sudo_run.sh','/home/cody/miniconda3/envs/behavior-2026/bin/python','scripts/run_two_stage.py','--output','outputs/ood99_two_stage_20261019','--attempt','pilot_v1','--tasks','bringing_paper_to_recycling'],cwd=ROOT,check=True)
