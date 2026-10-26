"""Read-only G1 LowState motor health snapshot. Creates no DDS publisher."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.g1_sonic.state import NAMES,HARDWARE_TO_ISAAC


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    names={hardware:name for name,hardware in zip(NAMES,HARDWARE_TO_ISAAC)}
    latest={};lock=threading.Lock()
    def receive(msg):
        row=dict(monotonic=time.monotonic(),motors=[dict(hardware_index=i,name=names[i],
            temperature_winding_driver_c=list(m.temperature),motorstate=int(m.motorstate),
            mode=int(m.mode),q_rad=float(m.q),dq_rad_s=float(m.dq),tau_est=float(m.tau_est))
            for i,m in enumerate(msg.motor_state[:29])])
        with lock:latest.update(row)
    ChannelFactoryInitialize(0,'eno1')
    subscriber=ChannelSubscriber('rt/lowstate',LowState_);subscriber.Init(receive,1)
    try:
        start=time.monotonic()
        while time.monotonic()-start<3:time.sleep(.05)
        with lock:row=dict(latest)
        if not row or time.monotonic()-row['monotonic']>.2:
            raise RuntimeError('No fresh LowState motor health')
        row.update(source='live_lowstate',commands_sent=0)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(row,f,indent=2);f.write('\n')
        print(json.dumps(dict(commands_sent=0,hottest=sorted(row['motors'],
            key=lambda m:max(m['temperature_winding_driver_c']),reverse=True)[:6],
            nonzero_motorstate=[m for m in row['motors'] if m['motorstate']]),indent=2))
    finally:subscriber.Close()


if __name__=='__main__':main()
