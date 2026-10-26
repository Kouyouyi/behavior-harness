"""Local G1 console HTTP API. Uses the existing operator authentication boundary."""
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
import hmac
import json
from pathlib import Path
import secrets

from services.dashboard.server import COOKIE, Handler as OperatorHandler
from .runtime import Console

STATIC = Path(__file__).parent / 'static'
ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/assets/console.js': ('console.js', 'application/javascript; charset=utf-8'),
          '/assets/composer.js': ('composer.js', 'application/javascript; charset=utf-8'),
          '/assets/console.css': ('console.css', 'text/css; charset=utf-8'),
          '/assets/observation.png': ('observation.png', 'image/png')}


class Handler(OperatorHandler):
    server_version = 'G1Console/1.0'

    def log_message(self, fmt, *args):
        # Do not log the bootstrap token or query strings.
        if self.command == 'POST':
            print(f'[g1-console] POST {self._route()[0]}', flush=True)

    def _send(self, status, data, content_type, extra=()):
        self.send_response(status)
        for name, value in [('Content-Type', content_type), ('Content-Length', str(len(data))),
                            ('Cache-Control', 'no-store'), ('X-Content-Type-Options', 'nosniff'),
                            ('X-Frame-Options', 'DENY'), ('Referrer-Policy', 'no-referrer'),
                            ('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; "
                             "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
                             "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"), *extra]:
            self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path, query = self._route()
        if path == '/':
            token = (query.get('token') or [''])[0]
            if token and hmac.compare_digest(token, self.token):
                self._send(303, b'', 'text/plain', [('Location', '/'),
                           ('Set-Cookie', f'{COOKIE}={self.token}; HttpOnly; SameSite=Strict; Path=/')])
                return
        if not self._authorized():
            if path == '/':
                self._send(401, '请使用启动脚本输出的完整地址打开控制台。'.encode(), 'text/plain; charset=utf-8')
            else:
                self._error(401, 'operator token required')
            return

        def handle():
            if path in ASSETS:
                filename, content_type = ASSETS[path]
                self._send(200, (STATIC / filename).read_bytes(), content_type)
            elif path == '/favicon.ico':
                self._send(204, b'', 'image/x-icon')
            elif path == '/api/v1/state':
                self._json(200, self.console.snapshot())
            elif path == '/api/v1/schema':
                self._send(200, (STATIC.parent / 'openapi.json').read_bytes(), 'application/json; charset=utf-8')
            elif path == '/api/v1/events':
                after = int((query.get('after') or ['0'])[0])
                if after < 0:
                    raise ValueError('after must be nonnegative')
                self._json(200, self.console.event_page(after))
            elif path == '/api/v1/history':
                self._json(200, {'tasks': self.console.snapshot()['history']})
            elif path == '/api/v1/export':
                self._send(200, json.dumps(self.console.export(), ensure_ascii=False, indent=2).encode(),
                           'application/json; charset=utf-8',
                           [('Content-Disposition', 'attachment; filename="g1-session.json"')])
            elif path == '/api/v1/camera/frame':
                frame_id = (query.get('frame_id') or [''])[0]
                self._send(200, self.console.camera.frame(frame_id), 'image/png')
            else:
                self._error(404, 'not found')
        self._guarded(handle)

    def do_POST(self):
        if not self._authorized():
            self._error(401, 'operator token required')
            return
        path, _ = self._route()

        def handle():
            length = int(self.headers.get('Content-Length') or 0)
            if not 0 < length <= 16384:
                raise ValueError('body must contain 1–16384 bytes')
            body = self._body()
            if path == '/api/v1/tasks':
                self._json(201, {'task': self.console.submit(body.get('instruction'), body.get('request_id'))})
            elif path.startswith('/api/v1/tasks/') and path.endswith('/actions'):
                parts = path.strip('/').split('/')
                if len(parts) != 5:
                    raise ValueError('invalid task route')
                self._json(200, {'task': self.console.action(parts[3], body.get('action'))})
            elif path == '/api/v1/messages':
                self._json(200, self.console.message(body.get('text'), body.get('request_id')))
            else:
                self._error(404, 'not found')
        self._guarded(handle)


def make_server(host='127.0.0.1', port=8780, *, console=None, token=None):
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('G1 console binds to loopback; use SSH forwarding for remote access')
    handler = type('BoundG1Console', (Handler,), {'console': console or Console(),
                                                'token': token or secrets.token_urlsafe(24)})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, handler.token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8780)
    parser.add_argument('--mode', choices=['demo', 'readonly'], default='demo')
    parser.add_argument('--storage', type=Path, default=Path('.local-runtime/g1-console'))
    parser.add_argument('--camera-directory', type=Path)
    parser.add_argument('--telemetry-file', type=Path)
    parser.add_argument('--owner-file', type=Path)
    parser.add_argument('--agent-events-file', type=Path)
    parser.add_argument('--message-directory', type=Path, help='Current G1 run directory; enable durable operator inbox')
    parser.add_argument('--thermal-rule', choices=['native90', 'strict91'], default='native90',
                        help='display fallback when no fresh owner rule is available')
    parser.add_argument('--no-autoplay', action='store_true', help='disable automatic demo task playback')
    args = parser.parse_args()
    console = Console(mode=args.mode, storage=args.storage / args.mode, camera_directory=args.camera_directory,
                      telemetry_file=args.telemetry_file, owner_file=args.owner_file,
                      agent_events_file=args.agent_events_file, thermal_rule=args.thermal_rule,
                      demo_autoplay=not args.no_autoplay, message_directory=args.message_directory)
    server, token = make_server(port=args.port, console=console)
    console.start()
    print(f'G1 Console: http://127.0.0.1:{server.server_port}/?token={token}', flush=True)
    print(f'Mode: {args.mode}. Operator inbox: {bool(console.inbox)}. This HTTP service never executes robot commands or model requests.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        console.close()


if __name__ == '__main__':
    main()
