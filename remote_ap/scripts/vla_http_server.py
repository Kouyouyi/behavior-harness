"""Loopback VLA service; use SSH port forwarding from the simulator host."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
import argparse,base64,json,tempfile
import numpy as np
MAX_REQUEST_BYTES=16*1024*1024

def infer_payload(policy,payload):
 if not isinstance(payload,dict) or set(payload)!={"state","images_base64","instruction"}:raise ValueError("Expected only state, images_base64, instruction")
 state=np.asarray(payload["state"],dtype=np.float32)
 if state.shape!=(61,) or not np.isfinite(state).all():raise ValueError("Finite 61D state required")
 images=payload["images_base64"]
 if not isinstance(images,list) or len(images)!=3:raise ValueError("Three images required in head/left/right order")
 instruction=payload["instruction"]
 if not isinstance(instruction,str) or not instruction.strip():raise ValueError("Nonempty instruction required")
 with tempfile.TemporaryDirectory(prefix="vla-request-") as directory:
  paths=[]
  for i,encoded in enumerate(images):
   if not isinstance(encoded,str):raise ValueError("Encoded images must be strings")
   image=base64.b64decode(encoded,validate=True);path=Path(directory)/f"camera_{i}.png";path.write_bytes(image);paths.append(str(path))
  actions=np.asarray(policy.predict({"state":state,"image_paths":paths},instruction),dtype=np.float32)
 if actions.shape!=(32,23) or not np.isfinite(actions).all():raise ValueError("Invalid model output")
 return {"actions":actions.tolist(),"representation":"original_R1Pro_controller_inputs","prediction_steps":32,"action_dim":23}

def handler_for(policy,checkpoint):
 class Handler(BaseHTTPRequestHandler):
  def send_json(self,code,body):
   data=json.dumps(body,allow_nan=False).encode();self.send_response(code);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
  def do_GET(self):
   if self.path!="/health":self.send_json(404,{"error":"not_found"});return
   self.send_json(200,{"ready":True,"checkpoint":str(checkpoint),"state_dim":61,"action_dim":23,"prediction_steps":32})
  def do_POST(self):
   if self.path!="/predict":self.send_json(404,{"error":"not_found"});return
   try:
    length=int(self.headers.get("Content-Length","0"))
    if not 0<length<=MAX_REQUEST_BYTES:raise ValueError("Invalid request length")
    payload=json.loads(self.rfile.read(length));output=infer_payload(policy,payload)
   except (ValueError,TypeError,KeyError,AssertionError) as error:self.send_json(400,{"error":str(error)});return
   except Exception as error:self.log_error("Inference failed: %s",type(error).__name__);self.send_json(500,{"error":"inference_failed"});return
   self.send_json(200,output)
 return Handler

def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--port",type=int,default=18769);args=p.parse_args()
 assert json.loads((args.checkpoint/"READY.json").read_text())["complete"],"Incomplete checkpoint"
 from pi05_executor import Pi05Executor
 policy=Pi05Executor(args.checkpoint)
 server=HTTPServer(("127.0.0.1",args.port),handler_for(policy,args.checkpoint))
 print(json.dumps({"status":"ready","address":server.server_address,"checkpoint":str(args.checkpoint)}),flush=True)
 server.serve_forever()
if __name__=="__main__":main()
