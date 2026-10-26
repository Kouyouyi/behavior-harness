"""Restricted stdio proxy to the EXISTING RAX frontend; no robot publisher."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

from auto_research_agent.api.runtime import Context
from services.g1_sonic.messages import Inbox
from services.workspace import ROBOT_FACTS
from .task_steps import NAME as TASK_STEP, SCHEMA as TASK_SCHEMA, TaskSteps
from .replay_route import NAME as REPLAY_ROUTE, SCHEMA as REPLAY_SCHEMA, ReplayRoute

ALLOWED = frozenset(('status', TASK_STEP, REPLAY_ROUTE, *ROBOT_FACTS['g1_sonic']['tools']))


def emit(run, task, kind, **fields):
    row=dict(id=uuid.uuid4().hex,time=time.time(),task_id=task,kind=kind,**fields)
    data=(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n').encode()
    fd=os.open(Path(run)/'agent-events.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    try:os.write(fd,data)
    finally:os.close(fd)


def check(run, session, batch, name, args):
    if name not in ALLOWED:raise ValueError('Tool not enabled for UI agent')
    box=Inbox(run)
    if box.session()!=session:raise ValueError('Owner session changed; submit a new UI instruction')
    state=box.dispatcher()
    if not state['online'] or state.get('batch')!=batch or state['phase']!='running':
        raise ValueError('UI agent dispatcher is no longer active')
    if name=='g1_vla' and args.get('action')=='run':
        if type(args.get('steps')) is not int or not 1<=args['steps']<=750:
            raise ValueError('UI VLA segments must be at most 750 steps / 15 seconds')


def serve(connection, run, session, batch, source, sink):
    compiled = None
    for line in source:
        request=json.loads(line)
        if 'id' not in request:continue
        ident=request['id'];method=request.get('method');params=request.get('params') or {}
        try:
            if method=='tools/list':
                value=connection.request(method,params)
                value['tools']=[t for t in value['tools'] if t['name'] in ALLOWED]
                value['tools'].append(TASK_SCHEMA)
                value['tools'].append(REPLAY_SCHEMA)
            elif method=='tools/call':
                name=params['name'];args=params.get('arguments') or {}
                check(run,session,batch,name,args)
                emit(run,batch,'tool',tool=name,status='running',details=dict(arguments=args))
                if name in (TASK_STEP, REPLAY_ROUTE):
                    if compiled is None:
                        compiled = TaskSteps(connection,run,session,batch,
                                             lambda: check(run,session,batch,TASK_STEP,{}))
                    value = (compiled.call(args) if name == TASK_STEP
                             else ReplayRoute(compiled,run).call(args))
                else:
                    value=connection.request(method,params)
                emit(run,batch,'tool',tool=name,status='failed' if value.get('isError') else 'completed',
                     details=dict(arguments=args),result=dict(content=[dict(type='text',text=b.get('text','')[:4000]) for b in value.get('content',[]) if b.get('type')=='text']))
            elif method in ('initialize','ping'):
                value=connection.request(method,params)
            else:raise ValueError('Unsupported MCP method')
            response=dict(jsonrpc='2.0',id=ident,result=value)
        except Exception as exc:
            if method == "tools/call":
                emit(run,batch,'tool',tool=params.get('name'),status='failed',
                     details=dict(arguments=params.get('arguments') or {}),
                     result=dict(error=str(exc)))
            value=dict(isError=True,content=[dict(type='text',text=json.dumps(
                dict(error=type(exc).__name__,reason=str(exc)),ensure_ascii=False))])
            response=dict(jsonrpc='2.0',id=ident,result=value) if method=='tools/call' else dict(jsonrpc='2.0',id=ident,error=dict(code=-32603,message=str(exc)))
        sink.write(json.dumps(response)+'\n');sink.flush()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket',required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--session',required=True);p.add_argument('--batch',required=True)
    a=p.parse_args()
    with Context(a.socket) as connection:
        serve(connection,a.run,json.loads(a.session),a.batch,sys.stdin,sys.stdout)


if __name__=='__main__':main()
