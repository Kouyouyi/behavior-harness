"""Unprivileged SeekAct backend client for a sole main-thread Isaac worker."""
import json
import socket
import uuid
from .schema import tools, validate, MUTATING, NATIVE_MOTION_TOOLS


def request(socket_path, name, arguments=None, *, request_id=None, timeout=300):
    ident=request_id or uuid.uuid4().hex
    payload={'id':ident,'name':name,'arguments':arguments or {}}
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(socket_path))
        sock.sendall((json.dumps(payload,allow_nan=False)+'\n').encode())
        with sock.makefile('rb') as stream:
            raw=stream.readline(16*1024*1024)
    if not raw: raise RuntimeError(f'Worker response absent for {ident}; do not replay physical actions')
    reply=json.loads(raw)
    if reply.get('id')!=ident: raise RuntimeError('Worker response identity mismatch')
    if 'error' in reply:
        raise RuntimeError(reply['error'])
    return reply['value'],reply.get('images',[])


class BehaviorBackend:
    mutating=MUTATING

    def __init__(self,config):
        self.config=config
        self.socket_path=config.backend_options['socket_path']
        self.status()  # Read-only handshake: never starts or resets a simulator.

    def tools(self):
        options=self.config.backend_options.get('vla_tools',{})
        return tools(include_vla=bool(options.get('enabled')), vla_only=bool(options.get('vla_only')))
    def validate(self,name,arguments):
        if self.config.backend_options.get('vla_tools',{}).get('vla_only') and name in NATIVE_MOTION_TOOLS:
            raise ValueError('Direct numeric motion is disabled in VLA-only scheduling')
        return validate(name,arguments)
    def call(self,name,arguments):
        self.validate(name,arguments)
        return request(self.socket_path,name,arguments)
    def status(self): return request(self.socket_path,'behavior_state')[0]
    def close(self):
        # A frontend reconnect must not discard or restart the physical episode.
        # Only the trial supervisor can use the private worker shutdown request.
        pass
