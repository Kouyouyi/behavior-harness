"""Huoshan-only OpenPI task execution server. No simulator dependencies."""
import argparse
import base64
import dataclasses
import json
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import cv2
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--port',type=int,default=18771);p.add_argument('--model-profile',choices=['multitask','radio'],default='multitask');args=p.parse_args()
manifest=json.loads(Path(args.manifest).read_text());tasks={s['task_name'] for s in manifest}
if args.model_profile=='radio':
    if tasks != {'turning_on_radio'}:raise ValueError('Radio checkpoint only supports radio')
from openpi.training import config
from openpi.policies.policy_config import create_trained_policy
from openpi_client.image_tools import resize_with_pad
c=config.get_config('pi05_b1k')
c=dataclasses.replace(c,data=dataclasses.replace(c.data,repo_id='turning_on_radio' if args.model_profile=='radio' else 'behavior-1k/2026-challenge-demos',robot_config_name='b1k/R1Pro'))
policy=create_trained_policy(c,args.checkpoint)
health={'ready':True,'checkpoint':args.checkpoint,'role':'original_task_execution',
        'repository':'kmy17518/b1k-challenge-2026-pi','revision':'b036d3c31e541efecb266c3e76f97513f99aa803',
        'state_dim':61,'action_dim':23,'prediction_steps':32,'tasks':sorted(tasks),'prompt_source':'task_name'}
if args.model_profile=='radio':
    import hashlib
    from openpi.configs.tasks import TASK_REGISTRY
    radio_prompt=TASK_REGISTRY['b1k']['turning_on_radio']
    health.update(repository='BEHAVIOR-official-radio',revision=hashlib.sha256((Path(args.checkpoint).parent/'download_manifest.json').read_bytes()).hexdigest(),prompt_source='official_task_registry',prompt=radio_prompt)
class Handler(BaseHTTPRequestHandler):
    def respond(self,data,status=200):
        data=json.dumps(data,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        self.respond(health if self.path=='/health' else {'error':'not found'},200 if self.path=='/health' else 404)
    def do_POST(self):
        if self.path!='/predict':self.respond({'error':'not found'},404);return
        try:
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            task=body['instruction'];state=np.asarray(body['state'],dtype=np.float32)
            if task not in tasks or state.shape!=(61,) or not np.isfinite(state).all():raise ValueError('Invalid task or proprio')
            if len(body['images_base64'])!=3:raise ValueError('Three cameras required')
            obs={'observation/state':state,'prompt':radio_prompt if args.model_profile=='radio' else task}
            for i,encoded in enumerate(body['images_base64']):
                bgr=cv2.imdecode(np.frombuffer(base64.b64decode(encoded),dtype=np.uint8),cv2.IMREAD_COLOR)
                if bgr is None:raise ValueError('Invalid camera PNG')
                obs[f'observation/image_{i}']=resize_with_pad(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),224,224)
            actions=np.asarray(policy.infer(obs)['actions'],dtype=np.float32)
            if actions.shape!=(32,23) or not np.isfinite(actions).all():raise ValueError('Invalid action chunk')
            self.respond({'actions':actions.tolist(),'representation':'original_R1Pro_controller_inputs','task_name':task})
        except (ValueError,KeyError) as exc:self.respond({'error':str(exc)},400)
HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
