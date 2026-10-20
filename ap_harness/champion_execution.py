"""Matched-state final-stage evaluation of published 2025 champion wrapper."""
import base64,json,time,uuid,os
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler
from urllib.error import HTTPError
import numpy as np
from .control import save_json
from .policy import retry_model_call

def run_task(sim,task,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    opener=build_opener(ProxyHandler({}));url=os.environ.get('CHAMPION_URL','http://127.0.0.1:18773')
    waitlog=output/'model_wait.jsonl'
    def fetch(path,payload=None):
        req=Request(url+path,data=payload,headers={'Content-Type':'application/json'})
        try:
            with opener.open(req,timeout=120) as response:return json.load(response)
        except HTTPError as e:raise ValueError(f'Champion protocol failure {e.code}: {e.read().decode()}') from e
    health=retry_model_call(lambda:fetch('/health'),waitlog)
    assert health['ready'] and health['revision']=='89545bc1b7aa7f2e687bc0032d091f132d715d4e'
    if health['role']=='champion_radio_execution':
        assert task=='turning_on_radio' and health['checkpoint']=='checkpoint_2'
    else:
        assert health['role']=='champion_task_execution' and health['task_name']==task
        manifest=json.loads((Path(__file__).resolve().parents[1]/'configs/champion_search_suite/manifest.json').read_text())
        entry=next(r for r in manifest['tasks'] if r['task']==task and r['status']=='eligible')
        assert health['task_id']==entry['task_id'] and health['checkpoint']==entry['checkpoint']
    save_json(output/'remote_health.json',health)
    episode=str(uuid.uuid4());rounds=0
    try:
        while not sim.done and not sim.success:
            d=output/f'round_{rounds:05d}';d.mkdir();obs=sim.capture(d);step=sim.steps
            current=retry_model_call(lambda:fetch('/health'),waitlog)
            if current!=health:raise ValueError('Champion service identity/state changed')
            body={'episode':episode,'seq':rounds,'state':obs['state'],'images_base64':[base64.b64encode(Path(p).read_bytes()).decode() for p in obs['image_paths']]}
            payload=json.dumps(body,allow_nan=False).encode()
            result=retry_model_call(lambda:fetch('/predict',payload),waitlog)
            assert result['boot_id']==health['boot_id'] and result['seq']==rounds
            assert sim.steps==step,'Physics advanced during inference'
            actions=np.asarray(result['actions'],dtype=np.float32)
            assert actions.shape==(20,23) and np.isfinite(actions).all()
            save_json(d/'champion_response.json',result)
            np.save(d/'raw_model_actions.npy',np.asarray(result['raw_model_output']['actions']))
            np.save(d/'predicted_actions.npy',actions)
            save_json(d/'prediction.json',{'task_name':task,'execute_prefix':20,'discarded_suffix':0,'observation_step':step,'representation':'published_champion_wrapper_output_including_corrections_and_resampling','raw_model_file':'raw_model_actions.npy'})
            with (d/'execution.jsonl').open('w',buffering=1) as stream:
                for i,action in enumerate(actions):
                    if sim.done or sim.success:break
                    stream.write(json.dumps({'event':'submitted','offset':i,'step':sim.steps,'action':action.tolist(),'time_ns':time.time_ns()})+'\n')
                    if not sim.step(action):break
                    stream.write(json.dumps({'event':'ack','offset':i,'step':sim.steps,'time_ns':time.time_ns()})+'\n')
            rounds+=1
    finally:
        save_json(output/'summary.json',{'objective':'champion_original_task_completion','official_task_success':bool(sim.success),'steps':sim.steps,'rounds':rounds,'reason':'official_task_success' if sim.success else sim.termination_reason})
    return 'official_task_success' if sim.success else sim.termination_reason
