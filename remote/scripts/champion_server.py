"""2025 winner wrapper; serialized, idempotent 20-action requests."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION","0.45")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import base64,io,json,hashlib,uuid,dataclasses
from http.server import HTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
from PIL import Image
from b1k.training import config
from b1k.policies.policy_config import create_trained_policy
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper,B1KWrapperConfig
root=Path("/cs/kyy/models/behavior-2025-champion")
manifest=json.loads((root/"DOWNLOAD_VERIFIED.json").read_text())
assert manifest["revision"]=="89545bc1b7aa7f2e687bc0032d091f132d715d4e" and manifest["task_id"]==0
cfg=config.get_config("pi_behavior_b1k_fast")
cfg=dataclasses.replace(cfg,assets_base_dir=str(root/"checkpoint_2/assets"))
p=create_trained_policy(cfg,root/"checkpoint_2",sample_kwargs={"num_steps":20})
def encode_scores(value):
 if isinstance(value,list):return [encode_scores(v) for v in value]
 if isinstance(value,float) and not np.isfinite(value):return str(value)
 return value
class Traced:
 def reset(self):p.reset()
 def infer(self,*a,**kw):
  out=p.infer(*a,**kw)
  self.raw={k:encode_scores(np.asarray(v).tolist()) for k,v in out.items() if k in ("actions","subtask_logits","predicted_stage")}
  return out
trace=Traced();wrapper=B1KPolicyWrapper(trace,task_id=0,config=B1KWrapperConfig());wrapper.reset()
health=dict(role="champion_radio_execution",revision=manifest["revision"],checkpoint="checkpoint_2",task_id=0,boot_id=str(uuid.uuid4()),wrapper=dataclasses.asdict(wrapper.config),state_adapter="61_named_fields_to_2025_256_used_fields",ready=True)
cache={};episode=None;next_seq=0
# Only six named fields are read by published inference; no privileged pose supplied.
def observations(body):
 s=np.asarray(body["state"],dtype=np.float32)
 assert s.shape==(61,) and np.isfinite(s).all()
 old=np.zeros(256,dtype=np.float32)
 for dst,src in [(slice(253,256),slice(0,3)),(slice(236,240),slice(53,57)),(slice(158,165),slice(3,10)),(slice(197,204),slice(28,35)),(slice(193,195),slice(24,26)),(slice(232,234),slice(49,51))]:old[dst]=s[src]
 obs={"robot_r1::proprio":old,"task_id":np.array([0])}
 keys=["zed_link","left_realsense_link","right_realsense_link"]
 for key,encoded in zip(keys,body["images_base64"],strict=True):obs[f"robot_r1::robot_r1:{key}:Camera:0::rgb"]=np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))
 return obs
smoke=observations(json.loads(Path("/cs/kyy/behavior_ap_experiment/reports/champion_smoke_input.json").read_text()))
smoke_actions=np.stack([wrapper.act(smoke).numpy() for _ in range(20)])
assert smoke_actions.shape==(20,23) and np.isfinite(smoke_actions).all()
Path("/cs/kyy/behavior_ap_experiment/reports/champion_inference_verified.json").write_text(json.dumps(dict(health=health,shape=list(smoke_actions.shape),raw_shape=list(np.asarray(trace.raw["actions"]).shape),finite=True)))
wrapper.reset()
class Handler(BaseHTTPRequestHandler):
 def reply(self,value,status=200):
  b=json.dumps(value,allow_nan=False).encode();self.send_response(status);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(b)));self.end_headers()
  try:self.wfile.write(b)
  except (BrokenPipeError,ConnectionResetError):pass
 def do_GET(self):self.reply(health)
 def do_POST(self):
  global episode,next_seq
  if not health["ready"]:return self.reply({"error":"model state invalid; requires restart"},503)
  raw=self.rfile.read(int(self.headers["Content-Length"]));body=json.loads(raw);key=(body["episode"],int(body["seq"]));digest=hashlib.sha256(raw).hexdigest()
  if key in cache:
   if cache[key][0]!=digest:return self.reply({"error":"request payload changed"},409)
   return self.reply(cache[key][1])
  if episode is None:
   if key[1]!=0:return self.reply({"error":"new episode must start seq0"},409)
   episode=key[0]
  if key!=(episode,next_seq):return self.reply({"error":"unexpected episode or sequence; restart service for new episode"},409)
  try:
   obs=observations(body);stage=wrapper.current_stage
   actions=np.stack([wrapper.act(obs).numpy() for _ in range(20)])
   assert actions.shape==(20,23) and np.isfinite(actions).all()
   result=dict(actions=actions.tolist(),raw_model_output=trace.raw,stage_before=stage,stage_after=wrapper.current_stage,seq=key[1],boot_id=health["boot_id"],representation="published_champion_wrapper_R1Pro_controller_inputs")
   cache[key]=(digest,result);next_seq+=1
   self.reply(result)
  except Exception as e:
   # Fail closed after any potentially stateful inference exception.
   health["ready"]=False;self.reply({"error":repr(e),"requires_restart":True},500);raise
print(json.dumps(health),flush=True)
HTTPServer(("127.0.0.1",18773),Handler).serve_forever()
