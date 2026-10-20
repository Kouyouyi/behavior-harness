task_id=26
import ast,base64,io,json,hashlib,threading,urllib.request,urllib.error
from http.server import HTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
from PIL import Image
source=Path("/cs/kyy/behavior_ap_experiment/scripts/champion_suite_server.py").read_text()
nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ("observations","Handler","encode_scores")]
class Tensor:
 def numpy(self):return np.zeros(23,dtype=np.float32)
class Wrapper:
 current_stage=0
 calls=0
 def act(self,obs):self.calls+=1;return Tensor()
class Trace:raw={"actions":np.zeros((30,23)).tolist()}
wrapper=Wrapper();trace=Trace();health={"ready":True,"boot_id":"test"};cache={};episode=None;next_seq=0
exec(compile(ast.Module(body=nodes,type_ignores=[]),"isolated_protocol","exec"),globals())
trace.raw["subtask_logits"]=encode_scores([0.,float("-inf")])
server=HTTPServer(("127.0.0.1",0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
b=io.BytesIO();Image.fromarray(np.zeros((4,4,3),dtype=np.uint8)).save(b,format="PNG")
payload=dict(episode="test",seq=0,state=np.arange(61).tolist(),images_base64=[base64.b64encode(b.getvalue()).decode()]*3)
def post(x):
 req=urllib.request.Request(f"http://127.0.0.1:{server.server_port}/predict",data=json.dumps(x).encode())
 with urllib.request.urlopen(req) as r:return json.load(r)
a=post(payload);assert wrapper.calls==20
assert a["raw_model_output"]["subtask_logits"]==[0.,"-inf"]
assert post(payload)==a and wrapper.calls==20
try:post(dict(payload,state=[0]*61));raise AssertionError("accepted changed duplicate")
except urllib.error.HTTPError as e:assert e.code==409
try:post(dict(payload,seq=2));raise AssertionError("accepted skipped seq")
except urllib.error.HTTPError as e:assert e.code==409
post(dict(payload,seq=1));assert wrapper.calls==40
server.shutdown();print("PASS: identical retry does not advance wrapper; changed retry and skipped sequence refused; next sequence advances once")
