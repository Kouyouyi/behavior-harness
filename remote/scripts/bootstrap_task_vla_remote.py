"""Wait for downloads/install, launch inference-only server, then run real smoke."""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error
root=Path('/cs/kyy/behavior_ap_experiment');vendor=root/'vendor/task_openpi'
models=Path('/cs/kyy/models/b1k-task-execution-pi05')
checkpoint=models/'multi-task-all-tasks-bs512/resume/checkpoint-5000'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',XLA_PYTHON_CLIENT_MEM_FRACTION='0.35',OMP_NUM_THREADS='4',HF_HOME='/cs/kyy/cache/task_vla_hf',OPENPI_DATA_HOME='/cs/kyy/cache/task_openpi',XDG_CACHE_HOME='/cs/kyy/cache/task_vla',HF_ENDPOINT='https://hf-mirror.com')
python=str(vendor/'.venv/bin/python')
while True:
    if (models/'DOWNLOAD_COMPLETE.json').exists():
        with (root/'reports/task_vla_import_probe.log').open('w') as log:
            probe=subprocess.run([python,'-c','import openpi, jax, flax, orbax.checkpoint, cv2'],cwd=vendor,env=env,stdout=log,stderr=log)
        if probe.returncode==0:break
    time.sleep(30)
with (root/'reports/task_vla_service.log').open('a') as log:
    proc=subprocess.Popen([python,'-u',str(root/'harness_ood99_20261019/task_vla_http_server.py'),'--checkpoint',str(checkpoint),'--manifest',str(root/'harness_ood99_20261019/ood99_manifest.json')],cwd=vendor,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(root/'reports/task_vla_service_pid.json').write_text(json.dumps({'pid':proc.pid}))
while True:
    if proc.poll() is not None:raise RuntimeError('Task VLA service exited; inspect task_vla_service.log')
    try:
        with urllib.request.urlopen('http://127.0.0.1:18771/health',timeout=5) as response:health=json.load(response)
        break
    except (OSError,TimeoutError):time.sleep(10)
request=urllib.request.Request('http://127.0.0.1:18771/predict',data=(root/'reports/task_vla_smoke_request.json').read_bytes(),headers={'Content-Type':'application/json'})
started=time.monotonic()
while True:
    try:
        with urllib.request.urlopen(request,timeout=600) as response:answer=json.load(response)
        break
    except (urllib.error.URLError,TimeoutError) as exc:
        if isinstance(exc,urllib.error.HTTPError) and exc.code not in (408,429,500,502,503,504):raise
        if proc.poll() is not None:raise RuntimeError('Inference server exited while waiting')
        time.sleep(15)
import numpy as np
actions=np.asarray(answer['actions'],dtype=np.float32)
assert actions.shape==(32,23) and np.isfinite(actions).all()
np.save(root/'reports/task_vla_smoke_actions.npy',actions)
(root/'reports/task_vla_inference_verified.json').write_text(json.dumps({'remote_inference_verified':True,'seconds':time.monotonic()-started,'health':health,'shape':list(actions.shape),'physical_task_success':False},indent=2))
