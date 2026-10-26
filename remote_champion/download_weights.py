import os,json,time,hashlib
os.environ.setdefault('HF_ENDPOINT','https://huggingface.co')
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]="120"
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import HfApi,snapshot_download
from pathlib import Path
root=Path(os.environ.get('CHAMPION_WEIGHTS','models/behavior-2025-champion'))
revision="89545bc1b7aa7f2e687bc0032d091f132d715d4e"
while True:
 try:
  info=HfApi().model_info("IliaLarchenko/behavior_submission",revision=revision,files_metadata=True)
  files=[s for s in info.siblings if s.rfilename.startswith("checkpoint_2/")]
  snapshot_download("IliaLarchenko/behavior_submission",revision=revision,allow_patterns=["checkpoint_2/*"],local_dir=root,max_workers=4)
  verified=[]
  for f in files:
   p=root/f.rfilename
   assert p.stat().st_size==f.size,(p,p.stat().st_size,f.size)
   h=hashlib.sha256()
   with p.open("rb") as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
   digest=h.hexdigest()
   expected=getattr(f.lfs,"sha256",None) if f.lfs else None
   if expected:assert digest==expected,(p,digest,expected)
   verified.append(dict(path=f.rfilename,size=f.size,sha256=digest,lfs_sha256=expected))
  (root/"DOWNLOAD_VERIFIED.json").write_text(json.dumps(dict(repo="IliaLarchenko/behavior_submission",revision=revision,scope="checkpoint_2 shared tasks; consult task_checkpoint_mapping.json",checkpoint="checkpoint_2",files=verified),indent=2))
  print("DOWNLOAD_VERIFIED",flush=True);break
 except Exception as e:
  print(type(e).__name__,str(e),flush=True);time.sleep(30)
