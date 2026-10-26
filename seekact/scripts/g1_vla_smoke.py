"""Exercise the real RAX VLA tool and GPU RTC with synthetic sensors, no hardware I/O."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from services.g1_sonic.vla import VLAJobs,DEFAULT_MODEL,atomic,TASKS
from services.g1_sonic.vla_runtime import Client,dependencies,identity,request_interval
from services.g1_sonic.backend import G1SonicBackend
from services.controller.config import Configuration
from services.controller.frontend import Frontend
from services.controller.session import Session


class SyntheticRuntime:
    def __init__(self,model,port):
        self.model_path=Path(model);rtc=dependencies();self.rtc=rtc
        self.adapter=rtc.legacy.BraincoModelAdapter(model,None)
        self.factory=lambda:Client(rtc,self.adapter,'127.0.0.1',port,identity(model))
        client=self.factory()
        try:self.rtc_config=client.rtc_config;self.interval=request_interval(self.rtc_config)
        finally:client.close()
        self.steps=0;self.target=0;self.pending=None;self.outputs=[];self.prompt=TASKS['mickey']
        self.session=rtc.RobotRTCSession(lambda:rtc.AsyncRTCController(self.factory,request_interval=self.interval),
            self.observation,self.stage,50)
        # Deterministic sensor fixture; no camera, state or BrainCo subscriber.
        rng=np.random.default_rng(7)
        self.rgb=rng.integers(0,256,(1,1,480,640,3),dtype=np.uint8)

    def observation(self):
        cfg=self.adapter.modalities
        return dict(video={k:np.repeat(self.rgb,len(cfg['video']['delta_indices']),axis=1) for k in cfg['video']['modality_keys']},
            state={k:np.tile(np.asarray(self.adapter.stats['state'][k]['mean'],np.float32),
                (1,len(cfg['state']['delta_indices']),1)) for k in cfg['state']['modality_keys']},
            language={k:[[self.prompt]] for k in cfg['language']['modality_keys']})

    def stage(self,action):self.pending=action;self.steps+=1
    def prepare(self,anchor):raise RuntimeError('Synthetic fixture does not move initial pose')
    def start(self,prompt,steps):
        self.prompt=prompt;self.target=steps;self.steps=0;self.session.resume()
    def poll(self):
        self.pending=None;self.session.poll()
        return self.pending,self.steps>=self.target
    def apply_hands(self,action):
        self.outputs.append({k:np.asarray(v).tolist() for k,v in action.items()})
    def pause(self):self.session.pause()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-path',default=DEFAULT_MODEL)
    p.add_argument('--port',type=int,default=5550)
    p.add_argument('--steps',type=int,default=250)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    runtime=SyntheticRuntime(a.model_path,a.port)
    jobs=VLAJobs(a.output,'synthetic-session',runtime);jobs.initialized=True
    config=Configuration(root=a.output/'session',task='vla_gpu_synthetic',robot='g1_sonic',
        backend_options=dict(control_directory=str(a.output)))
    frontend=Frontend(Session(config),G1SonicBackend(config))
    calls=[]
    def refresh():
        atomic(a.output/'ankle_status.json',dict(updated_monotonic=time.monotonic(),phase='ready',fault=None,
            session_id='synthetic-session',lease_remaining_seconds=600,vla=jobs.snapshot()))
    def call(name,arguments):
        response=frontend.call(name,arguments);calls.append(dict(tool=name,arguments=arguments,response=response))
        if response.get('isError'):raise RuntimeError(response)
        return json.loads(response['content'][0]['text'])
    try:
        refresh();call('g1_vla_status',{})
        submitted=call('g1_vla',dict(action='run',task='mickey',steps=a.steps,
            evidence='SYNTHETIC interface fixture; no physical object or robot connected'))
        at=time.monotonic()
        while True:
            jobs.tick(time.monotonic(),True,True,[0]*64,600);refresh()
            if not jobs.moving:break
            if time.monotonic()-at>30:raise RuntimeError('Smoke deadline')
            time.sleep(runtime.session.sleep_seconds())
        outcome=call('g1_vla_result',dict(job_id=submitted['job_id']))
        report=dict(model=identity(a.model_path),observation_source='synthetic_rgb_and_statistical_mean_state',
            robot_commands_sent=0,body_publishers=0,hand_publishers=0,steps=len(runtime.outputs),
            output_shapes={k:list(np.asarray(v).shape) for k,v in runtime.outputs[0].items()} if runtime.outputs else {},
            result=outcome,rtc_delay=runtime.rtc_config['rtc_delay_steps'],rtc_interval=runtime.interval,rate_hz=50)
        atomic(a.output/'report.json',report);print(json.dumps(report,indent=2))
        if not outcome.get('completed') or len(runtime.outputs)!=a.steps:raise RuntimeError('RTC smoke failed')
    finally:
        atomic(a.output/'tool_calls.json',calls)
        atomic(a.output/'actions.json',runtime.outputs)
        runtime.session.close();frontend.close()


if __name__=='__main__':main()
