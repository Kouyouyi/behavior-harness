from .paths import HANDOFF
from vla_http_client import RemotePi05Executor
import http.client
import json
import time
from pathlib import Path
from urllib.error import URLError, HTTPError

def retry_model_call(call, log_path=None, sleep=time.sleep):
    attempt=0
    while True:
        try:return call()
        except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
            if isinstance(exc,HTTPError) and exc.code not in (408,429,500,502,503,504):raise
            attempt+=1
            if log_path:
                with Path(log_path).open('a') as stream:
                    stream.write(json.dumps({'status':'waiting_for_model','attempt':attempt,'error':type(exc).__name__,'time':time.time(),'actions_submitted':0})+'\n')
            sleep(min(30,2*attempt))

CHECKPOINT = '/cs/kyy/behavior_ap_experiment/runs/pi05_goal_horizon_v2_b8/checkpoint_010000'

class PinnedRemotePolicy(RemotePi05Executor):
    def __init__(self, url='http://127.0.0.1:18769', timeout=60, checkpoint=CHECKPOINT, retry_log=None):
        accepted = [CHECKPOINT, CHECKPOINT.replace('010000','024000')]
        if checkpoint not in accepted:
            raise ValueError('Checkpoint metadata compatibility has not been verified')
        self.expected_checkpoint = checkpoint
        self.retry_log = retry_log
        super().__init__(url=url,timeout=timeout)

    def health(self):
        health = retry_model_call(super().health,self.retry_log)
        if health.get('checkpoint') != self.expected_checkpoint:
            raise ValueError('Remote checkpoint differs from the packaged statistics/tokenizer acceptance target')
        return health

    def predict(self, observation, instruction):
        predict=super().predict
        return retry_model_call(lambda:predict(observation,instruction),self.retry_log)
