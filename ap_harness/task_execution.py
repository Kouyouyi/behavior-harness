"""Continue the original BDDL task from a verified handoff state."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .policy import retry_model_call
from .control import save_json
from vla_http_client import RemotePi05Executor

class TaskPolicy(RemotePi05Executor):
    def __init__(self,output,url='http://127.0.0.1:18771'):
        super().__init__(url=os.environ.get('TASK_VLA_URL',url),timeout=120);self.log=Path(output)/'model_wait.jsonl'
    def health(self):return retry_model_call(super().health,self.log)
    def predict(self,observation,instruction):
        call=super().predict
        return retry_model_call(lambda:call(observation,instruction),self.log)

def run_task(sim,task,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    policy=TaskPolicy(output);health=policy.health()
    if health.get('role')!='original_task_execution' or health.get('revision') not in ('b036d3c31e541efecb266c3e76f97513f99aa803', hashlib.sha256((Path(__file__).resolve().parents[1]/'configs/radio_download_manifest.json').read_bytes()).hexdigest()) or task not in health.get('tasks',[]):
        raise ValueError('Task executor identity/coverage mismatch')
    save_json(output/'remote_health.json',health)
    rounds=0
    try:
        while not sim.done and not sim.success:
            directory=output/f'round_{rounds:05d}';directory.mkdir();obs=sim.capture(directory);step=sim.steps
            if policy.health()!=health:raise ValueError('Task executor changed during episode')
            actions=np.asarray(policy.predict(obs,task),dtype=np.float32)
            if actions.shape!=(32,23) or not np.isfinite(actions).all():raise ValueError('Invalid task actions')
            if sim.steps!=step:raise ValueError('Physics advanced while waiting for task model')
            np.save(directory/'predicted_actions.npy',actions)
            save_json(directory/'prediction.json',{'task_name':task,'execute_prefix':16,'discarded_suffix':16,'observation_step':step})
            with (directory/'execution.jsonl').open('w',buffering=1) as stream:
                for i,action in enumerate(actions[:16]):
                    if sim.done or sim.success:break
                    stream.write(json.dumps({'event':'submitted','offset':i,'step':sim.steps,'action':action.tolist(),'time_ns':time.time_ns()})+'\n')
                    if not sim.step(action):break
                    stream.write(json.dumps({'event':'ack','offset':i,'step':sim.steps,'time_ns':time.time_ns()})+'\n')
            rounds+=1
    finally:
        save_json(output/'summary.json',{'objective':'original_task_completion','official_task_success':bool(sim.success),'steps':sim.steps,'rounds':rounds,'reason':'official_task_success' if sim.success else sim.termination_reason})
    return 'official_task_success' if sim.success else sim.termination_reason
