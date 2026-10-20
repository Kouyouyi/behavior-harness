"""Run on huoshan only; download pinned inference files, never optimizer state."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import time
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT','120')
from huggingface_hub import hf_hub_download

m=json.loads(Path('/cs/kyy/behavior_ap_experiment/configs/task_vla_download.json').read_text())
root=Path(m['remote_root']);root.mkdir(parents=True,exist_ok=True)
if shutil.disk_usage(root).free<30*1024**3:raise RuntimeError('Need at least 30 GiB free on model volume')
(root/'download_manifest.json').write_text(json.dumps(m,indent=2))
def fetch(name):
    attempt=0
    while True:
        try:
            p=hf_hub_download(m['repo'],name,revision=m['revision'],local_dir=root)
            print(json.dumps({'file':name,'bytes':Path(p).stat().st_size,'status':'downloaded'}),flush=True)
            return p
        except Exception as exc:
            attempt+=1
            print(json.dumps({'file':name,'status':'waiting_retry','attempt':attempt,'error_type':type(exc).__name__}),flush=True)
            time.sleep(min(60,attempt*5))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    paths=list(pool.map(fetch,m['files']))
(root/'DOWNLOAD_COMPLETE.json').write_text(json.dumps({'revision':m['revision'],'files':len(paths),'bytes':sum(Path(p).stat().st_size for p in paths),'inference_verified':False},indent=2))
