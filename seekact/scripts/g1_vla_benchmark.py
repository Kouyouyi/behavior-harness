"""Measure frozen household GR00T locally and over loopback; no hardware commands.

Uses a saved real ego frame and synthetic mean state, batch=1. CUDA events
measure stream elapsed time (including launch gaps), not summed kernel time.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def summary(rows):
    import numpy as np
    keys={k for row in rows for k,v in row.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
    return {k:dict(zip(('p50','p95','max'),np.percentile([r[k] for r in rows if k in r],[50,95,100]).tolist())) for k in sorted(keys)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=40)
    p.add_argument('--rtc-steps',type=int,default=1000)
    p.add_argument('--port',type=int,default=5552)
    p.add_argument('--thread-sweep',type=int,nargs='*',default=[])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
    from services.g1_sonic.vla import DEFAULT_MODEL,DEFAULT_RTC,TASKS
    from services.g1_sonic.vla_runtime import identity,request_interval
    sys.path[:0]=[DEFAULT_RTC,str(Path(DEFAULT_RTC)/'scripts/local'),'/home/user/Isaac-GR00T/scripts/local']
    import numpy as np
    import torch
    from PIL import Image
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer,PolicyClient,MsgSerializer
    from brainco_model_adapter import BraincoModelAdapter
    from gr00t.policy.rtc_client import AsyncRTCController
    from rtc_robot_session import RobotRTCSession
    adapter=BraincoModelAdapter(DEFAULT_MODEL,None)
    rgb=np.asarray(Image.open(a.image).convert('RGB'))[None,None]
    cfg=adapter.modalities
    obs=dict(video={k:np.repeat(rgb,len(cfg['video']['delta_indices']),axis=1) for k in cfg['video']['modality_keys']},
             state={k:np.tile(np.asarray(adapter.stats['state'][k]['mean'],np.float32),(1,len(cfg['state']['delta_indices']),1)) for k in cfg['state']['modality_keys']},
             language={k:[[TASKS['mickey']]] for k in cfg['language']['modality_keys']})
    staged=ROOT/'runtime/g1/vla-model/models'/Path(DEFAULT_MODEL).name
    policy=Gr00tPolicy('UNITREE_G1_SONIC',str(staged),device='cuda',sonic_token_mode='continuous',rtc_delay_steps=8)
    report=dict(robot_commands_sent=0,body_publishers=0,hand_publishers=0,
                observation='saved_real_ego_image_and_synthetic_mean_state',image_shape=list(rgb.shape),
                gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
                torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
                parameters_by_dtype=dict(Counter({str(dtype):sum(p.numel() for p in policy.model.parameters() if p.dtype==dtype) for dtype in {p.dtype for p in policy.model.parameters()}})),
                denoising_steps=policy.model.action_head.num_inference_timesteps,
                model_identity=identity(DEFAULT_MODEL),cases={})
    def save():
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def measure(name,call,delay,count=None):
        previous=None;rows=[]
        for i in range((count or a.samples)+8):
            opts=None if previous is None else dict(rtc_delay_steps=delay,rtc_prefix={k:v[:,:delay].copy() for k,v in previous.items()})
            start=time.perf_counter();previous,info=call(obs,opts);wall=(time.perf_counter()-start)*1000
            if i>=8:rows.append(dict(wall_ms=wall,**info.get('timing',{})))
        report['cases'][name]=dict(warmup=8,samples=rows,summary=summary(rows))
        save();print(name,json.dumps(report['cases'][name]['summary']),flush=True)
    for delay in (8,10):
        policy.rtc_delay_steps=delay
        measure('direct_delay'+str(delay),policy.get_action,delay)

    # Synchronized phase timing without editing the external GR00T checkout.
    timings={};events=[]
    def cpu(name,fn):
        def wrapped(*args,**kwargs):
            at=time.perf_counter();value=fn(*args,**kwargs)
            timings[name]=timings.get(name,0)+(time.perf_counter()-at)*1000
            return value
        return wrapped
    def gpu(name,fn):
        def wrapped(*args,**kwargs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();value=fn(*args,**kwargs);end.record();events.append((name,start,end));return value
        return wrapped
    original_processor=policy.processor
    class Processor:
        def __getattr__(self,name):return getattr(original_processor,name)
        def __call__(self,*args,**kwargs):return cpu('processor_ms',original_processor)(*args,**kwargs)
        def decode_action(self,*args,**kwargs):return cpu('decode_ms',original_processor.decode_action)(*args,**kwargs)
    policy.processor=Processor()
    policy.collate_fn=cpu('collate_ms',policy.collate_fn)
    policy.model.backbone.forward=gpu('backbone_cuda_ms',policy.model.backbone.forward)
    policy.model.action_head.get_action=gpu('action_head_cuda_ms',policy.model.action_head.get_action)
    original_model=policy.model.get_action
    def model(*args,**kwargs):
        torch.cuda.synchronize();at=time.perf_counter()
        value=gpu('model_cuda_ms',original_model)(*args,**kwargs)
        torch.cuda.synchronize();timings['model_wall_ms']=(time.perf_counter()-at)*1000
        return value
    policy.model.get_action=model
    def profiled(observation,options=None):
        timings.clear();events.clear();at=time.perf_counter()
        actions,info=policy.get_action(observation,options)
        torch.cuda.synchronize()
        for key,start,end in events:timings[key]=start.elapsed_time(end)
        timings['policy_wall_ms']=(time.perf_counter()-at)*1000
        timings['outside_model_ms']=timings['policy_wall_ms']-timings['model_wall_ms']
        return actions,dict(info,timing=dict(timings))
    measure('direct_profiled_delay10',profiled,10)
    for threads in a.thread_sweep:
        if threads < 1:raise ValueError('CPU threads must be positive')
        torch.set_num_threads(threads)
        measure('direct_profiled_threads'+str(threads),profiled,10)
    if a.thread_sweep:torch.set_num_threads(min(a.thread_sweep))
    report['final_torch_threads']=torch.get_num_threads()
    server=PolicyServer(policy,host='127.0.0.1',port=a.port)
    server.register_endpoint('get_rax_identity',lambda:identity(DEFAULT_MODEL),requires_input=False)
    server.register_endpoint('get_action',profiled)
    server.register_endpoint('benchmark_runtime',lambda:dict(torch_threads=torch.get_num_threads(),thread=threading.current_thread().name),requires_input=False)
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    try:
        with PolicyClient('127.0.0.1',a.port,timeout_ms=5000) as raw:
            report['rpc_runtime']=raw.call_endpoint('benchmark_runtime',requires_input=False)
            measure('loopback_profiled_delay10',raw.get_action,10)
            calls=[]
            class TimedClient(PolicyClient):
                def get_action(self,observation,options=None):
                    at=time.perf_counter();actions,info=super().get_action(observation,options)
                    actions.pop('task_progress',None);actions.pop('action.task_progress',None)
                    result=adapter.action(None,actions),info
                    calls.append(dict(wall_ms=(time.perf_counter()-at)*1000,bootstrap=options is None))
                    return result
            factory=lambda:TimedClient('127.0.0.1',a.port,timeout_ms=2000)
            steps=[0]
            def publish(action):steps[0]+=1
            interval=request_interval(policy.get_rtc_config())
            session=RobotRTCSession(lambda:AsyncRTCController(factory,request_interval=interval),lambda:obs,publish,50)
            started=time.monotonic();error=None
            try:
                session.resume()
                while steps[0]<a.rtc_steps:
                    session.poll()
                    if time.monotonic()-started>a.rtc_steps/50+15:raise RuntimeError('benchmark deadline')
                    time.sleep(session.sleep_seconds())
            except Exception as exc:error=str(exc)
            finally:session.close()
            report['rtc_clock']=dict(steps=steps[0],requested_steps=a.rtc_steps,error=error,delay=10,interval=interval,
                                     calls=calls,summary=summary([c for c in calls if not c['bootstrap']]),
                                     server_phase_instrumentation_enabled=True)
            save();print('rtc_clock',steps[0],error,report['rtc_clock']['summary'],flush=True)
            raw.call_endpoint('kill',requires_input=False)
        thread.join(timeout=5)
    finally:server.close()
    save()


if __name__=='__main__':main()
