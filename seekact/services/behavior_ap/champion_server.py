"""Dedicated remote champion tool service; no Isaac installation/import.

Uses the original public checkpoint_2 and published B1K wrapper unchanged.
Task switches are explicit identities, never interpreted language prompts.
"""
import argparse
import base64
import dataclasses
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import uuid

SKILLS = {'turning_on_radio': 0, 'putting_shoes_on_rack': 22,
          'picking_up_toys': 7, 'collecting_childrens_toys': 21}
REVISION = '89545bc1b7aa7f2e687bc0032d091f132d715d4e'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18778)
    parser.add_argument('--weights', type=Path, default=Path(os.environ.get('CHAMPION_WEIGHTS','models/behavior-2025-champion')))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-input',type=Path,help='Optional real RGB/proprio request; default synthetic interface smoke')
    args = parser.parse_args()
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.45')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import numpy as np
    from PIL import Image
    from b1k.training import config
    from b1k.policies.policy_config import create_trained_policy
    from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
    from b1k.shared.eval_utils_2025 import TASK_NAMES_TO_INDICES
    assert all(TASK_NAMES_TO_INDICES[k] == v for k, v in SKILLS.items())
    manifest = json.loads((args.weights / 'DOWNLOAD_VERIFIED.json').read_text())
    assert manifest['revision'] == REVISION and manifest['checkpoint'] == 'checkpoint_2'
    cfg = config.get_config('pi_behavior_b1k_fast')
    cfg = dataclasses.replace(cfg, assets_base_dir=str(args.weights / 'checkpoint_2/assets'))
    policy = create_trained_policy(cfg, args.weights / 'checkpoint_2', sample_kwargs={'num_steps': 20})

    def scores(value):
        if isinstance(value, list):
            return [scores(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value

    class Trace:
        def reset(self):
            self.raw = None
            policy.reset()
        def infer(self, *a, **kw):
            result = policy.infer(*a, **kw)
            self.raw = {k: scores(np.asarray(v).tolist()) for k, v in result.items()
                        if k in ('actions', 'subtask_logits', 'predicted_stage')}
            return result
    trace = Trace()
    wrapper = None
    episode, next_seq, current_skill, epoch = None, 0, None, None
    cache = {}
    health = {'ready': True, 'role': 'seekact_champion_tools', 'revision': REVISION,
        'checkpoint': 'checkpoint_2', 'supported_skills': SKILLS, 'boot_id': uuid.uuid4().hex,
        'language_input': False, 'state_adapter': '61_named_fields_to_2025_256_used_fields',
        'wrapper': dataclasses.asdict(B1KWrapperConfig()),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def observation(body):
        s = np.asarray(body['state'], dtype=np.float32)
        if s.shape != (61,) or not np.isfinite(s).all():
            raise ValueError('Finite61D state required')
        old = np.zeros(256, dtype=np.float32)
        for dst, src in [(slice(253,256),slice(0,3)), (slice(236,240),slice(53,57)),
                         (slice(158,165),slice(3,10)), (slice(197,204),slice(28,35)),
                         (slice(193,195),slice(24,26)), (slice(232,234),slice(49,51))]:
            old[dst] = s[src]
        obs = {'robot_r1::proprio': old, 'task_id': np.array([SKILLS[body['skill']]])}
        if len(body['images_base64']) != 3:
            raise ValueError('Three ordered RGB images required')
        for camera, encoded in zip(('zed_link','left_realsense_link','right_realsense_link'),body['images_base64'],strict=True):
            obs[f'robot_r1::robot_r1:{camera}:Camera:0::rgb'] = np.asarray(
                Image.open(io.BytesIO(base64.b64decode(encoded,validate=True))).convert('RGB'))
        return obs

    # Real inference smoke, with RGB and proprioception; no simulation actions.
    if args.smoke_input:
        fixture=json.loads(args.smoke_input.read_text())
    else:
        buffer=io.BytesIO();Image.new('RGB',(224,224),(128,128,128)).save(buffer,format='PNG')
        fixture={'state':[0.0]*61,'images_base64':[base64.b64encode(buffer.getvalue()).decode()]*3}
    smoke = []
    for skill in ('turning_on_radio','putting_shoes_on_rack'):
        body = dict(fixture, skill=skill)
        w = B1KPolicyWrapper(trace, task_id=SKILLS[skill], config=B1KWrapperConfig())
        w.reset()
        actions = np.stack([w.act(observation(body)).numpy() for _ in range(20)])
        assert actions.shape == (20,23) and np.isfinite(actions).all()
        smoke.append({'skill':skill,'shape':[20,23],'finite':True})
    policy.reset()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'inference_verified.json').write_text(json.dumps({'health':health,'smoke':smoke,'physical_steps':0},indent=2))

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200):
            raw = json.dumps(value,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)));self.end_headers()
            try:self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_GET(self):
            self.reply(health if self.path == '/health' else {'error':'not_found'},200 if self.path=='/health' else 404)

        def do_POST(self):
            nonlocal wrapper,episode,next_seq,current_skill,epoch
            if self.path != '/predict':return self.reply({'error':'not_found'},404)
            if not health['ready']:return self.reply({'error':'invalid model state','requires_restart':True},503)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=16*1024*1024:raise ValueError('Request size')
                raw=self.rfile.read(size);body=json.loads(raw)
                if set(body)!={'episode','seq','context_epoch','skill','state','images_base64'}:raise ValueError('Unexpected fields')
                if body['skill'] not in SKILLS:raise ValueError('Unsupported fixed skill')
                if type(body['seq'])is not int or body['seq']<0 or type(body['context_epoch'])is not int:raise ValueError('Invalid sequence/context')
                if not isinstance(body['episode'],str) or not 1<=len(body['episode'])<=100:raise ValueError('Episode ID')
                key=(body['episode'],body['seq']);digest=hashlib.sha256(raw).hexdigest()
                if key in cache:
                    if cache[key][0]!=digest:return self.reply({'error':'request payload changed'},409)
                    return self.reply(cache[key][1])
                if body['episode']!=episode:
                    if body['seq']!=0:return self.reply({'error':'new episode requiresseq0'},409)
                elif body['seq']!=next_seq:return self.reply({'error':'unexpected sequence'},409)
                obs=observation(body)  # Validate before mutating temporal policy state.
            except Exception as exc:return self.reply({'error':str(exc)},400)
            try:
                reset=(body['episode']!=episode or body['skill']!=current_skill or body['context_epoch']!=epoch)
                if reset:
                    wrapper=B1KPolicyWrapper(trace,task_id=SKILLS[body['skill']],config=B1KWrapperConfig())
                    wrapper.reset()
                stage=wrapper.current_stage
                actions=np.stack([wrapper.act(obs).numpy() for _ in range(20)])
                assert actions.shape==(20,23) and np.isfinite(actions).all()
                assert trace.raw is not None and 'actions' in trace.raw
                result={'actions':actions.tolist(),'raw_model_output':trace.raw,'stage_before':stage,
                    'stage_after':wrapper.current_stage,'wrapper_reset':reset,'skill':body['skill'],
                    'task_id':SKILLS[body['skill']],'episode':body['episode'],'seq':body['seq'],
                    'context_epoch':body['context_epoch'],'boot_id':health['boot_id'],
                    'representation':'published_champion_wrapper_R1Pro_controller_inputs'}
                cache[key]=(digest,result)
                episode,next_seq,current_skill,epoch=body['episode'],body['seq']+1,body['skill'],body['context_epoch']
                self.reply(result)
            except Exception as exc:
                health['ready']=False
                self.reply({'error':repr(exc),'requires_restart':True},500)
    print(json.dumps(health),flush=True)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__ == '__main__':
    main()
