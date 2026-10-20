"""Verify the transferred official checkpoint before serving it on huoshan."""
import json,os,subprocess,time,zlib,urllib.request,urllib.error
from pathlib import Path
root=Path('/cs/kyy/behavior_ap_experiment');models=Path('/cs/kyy/models/pi05-BEHAVIOR-turning-on-radio')
checkpoint=models/'pi05_turn_on_the_radio';vendor=root/'vendor/task_openpi';python=str(vendor/'.venv/bin/python')
while True:
    try:
        manifest=json.loads((models/'download_manifest.json').read_text())
        files=[f for f in manifest['files'] if not f['path'].endswith('/')]
        if all((models/f['path']).stat().st_size==f['size'] for f in files):break
    except (OSError,ValueError):pass
    time.sleep(30)
for f in files:
    crc=0
    with (models/f['path']).open('rb') as stream:
        while b:=stream.read(8*1024*1024):crc=zlib.crc32(b,crc)
    if crc!=f['crc32']:raise ValueError('Checkpoint CRC mismatch: '+f['path'])
(models/'TRANSFER_VERIFIED.json').write_text(json.dumps({'files':len(files),'crc_verified':True,'bytes':sum(f['size'] for f in files)}))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',XLA_PYTHON_CLIENT_MEM_FRACTION='0.30',OMP_NUM_THREADS='4',HF_HOME='/cs/kyy/cache/task_vla_hf',OPENPI_DATA_HOME='/cs/kyy/cache/task_openpi',XDG_CACHE_HOME='/cs/kyy/cache/task_vla',HF_ENDPOINT='https://hf-mirror.com')
while True:
    with (root/'reports/radio_vla_import_probe.log').open('w') as log:
        r=subprocess.run([python,'-c','import openpi,jax,flax,orbax.checkpoint,cv2'],cwd=vendor,env=env,stdout=log,stderr=log)
    if r.returncode==0:break
    time.sleep(30)
manifest_path=root/'configs/radio_task_manifest.json';manifest_path.write_text(json.dumps([{'task_name':'turning_on_radio'}]))
with (root/'reports/radio_vla_service.log').open('a') as log:
    proc=subprocess.Popen([python,'-u',str(root/'harness_ood99_20261019/task_vla_http_server.py'),'--checkpoint',str(checkpoint),'--manifest',str(manifest_path),'--model-profile','radio','--port','18772'],cwd=vendor,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(root/'reports/radio_vla_service_pid.json').write_text(json.dumps({'pid':proc.pid}))
while True:
    if proc.poll() is not None:raise RuntimeError('Radio service exited; inspect log')
    try:
        with urllib.request.urlopen('http://127.0.0.1:18772/health',timeout=5) as r:health=json.load(r)
        break
    except OSError:time.sleep(10)
body=json.loads((root/'reports/task_vla_smoke_request.json').read_text());body['instruction']='turning_on_radio'
request=urllib.request.Request('http://127.0.0.1:18772/predict',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
while True:
    try:
        with urllib.request.urlopen(request,timeout=600) as r:answer=json.load(r)
        break
    except (OSError,TimeoutError) as exc:
        if isinstance(exc,urllib.error.HTTPError) and exc.code not in (408,429,500,502,503,504):raise
        if proc.poll() is not None:raise RuntimeError('Radio service exited during smoke')
        time.sleep(15)
import numpy as np
a=np.asarray(answer['actions']);assert a.shape==(32,23) and np.isfinite(a).all()
np.save(root/'reports/radio_vla_smoke_actions.npy',a)
(root/'reports/radio_vla_inference_verified.json').write_text(json.dumps({'remote_inference_verified':True,'health':health,'shape':list(a.shape),'physical_task_success':False},indent=2))
