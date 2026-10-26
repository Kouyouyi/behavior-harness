"""Host-only Anthropic Messages endpoint for Claude Code. The agent never holds credentials.

The host-run Claude Code talks to ANTHROPIC_BASE_URL=http://127.0.0.1:<port> (this
relay, on loopback TCP; a Unix socket path is also accepted). The relay adds the
provider credential and enforces a model allowlist, a model-call cap, local
(client-executed) tools only, inline media only, bounded bodies, and provider
throttling retried before any byte reaches the client. Token usage is appended to a
host-side JSONL for the operator.

Providers:
- ``openrouter``: https://openrouter.ai/api/v1/messages with the OpenRouter key.
- ``anthropic``: https://api.anthropic.com/v1/messages with an API key (x-api-key).
- ``claude-login``: https://api.anthropic.com with the dedicated experiment account's
  long-lived ``claude setup-token`` token (harness/claude_cli/experiment_login.py),
  re-read for every request so a rotated token takes effect at once. The operator's
  personal Claude login (~/.claude) is refused.
"""
import gzip
from http.server import BaseHTTPRequestHandler
import http.client
import io
import json
import os
import re
import socket
import stat
import struct
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from harness.claude_cli.relay_common import CallCounter, LoopbackHTTPServer, UnixHTTPServer, retry_delay

MAX_BODY = 256*1024*1024
REQUEST_TIMEOUT_SECONDS = 120
RETRY_STATUSES = frozenset({429, 502, 503, 504, 529})
# Retry an overloaded / failed provider every few seconds (RETRY_MAX_DELAY_SECONDS) for up to a minute, then
# pass the error on: the agent CLI retries again with its own visible backoff. A 10-minute silent wait here
# looked like a hung session.
RETRY_WINDOW_SECONDS = 60
# A streaming response that sends nothing for this long has stalled (healthy streams went at most 31 s
# without a byte, stalled ones 119-334 s, 2026-10-09). The relay then waits no longer: before the response
# starts it retries, mid-stream it drops the connection so the agent CLI retries at once instead of after
# its own ~4.5 min watchdog. Non-streaming requests keep the long timeout.
STREAM_IDLE_SECONDS = 90
UPSTREAM_TIMEOUT_SECONDS = 600
STALLED_EVENT = (b'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", '
                 b'"message": "relay: the provider stream stalled; retry"}}\n\n')
MAX_OUTPUT_TOKENS = 128000
BUDGET_EXHAUSTED = b'Relay model-call budget exhausted; the operator must raise --max-model-calls.'
OAUTH_BETA = 'oauth-2025-04-20'
# model: the agent; auditor / fast: the models Claude Code's `sonnet` / `haiku` aliases resolve to
# (pinned in the agent's environment, see `aliases`), used by subagents and background tasks. The
# relay forwards ANY well-formed model id ('*' in a session's models): every Claude model the
# provider serves is available to the agent and its subagents. A 2026-10-09 probe of the Claude
# login found Opus 5.5, Sonnet 5.5, Haiku 5.5, Fable 5.1 and Haiku 4.5 all known to the API.
PROVIDERS = {
    'openrouter': {'base_url': 'https://openrouter.ai/api', 'model': 'anthropic/claude-opus-5.5',
                   'auditor': 'anthropic/claude-sonnet-5.5', 'fast': 'anthropic/claude-haiku-4.5',
                   'aliases': {'opus': 'anthropic/claude-opus-5.5', 'sonnet': 'anthropic/claude-sonnet-5.5',
                               'haiku': 'anthropic/claude-haiku-4.5'}},
    'anthropic': {'base_url': 'https://api.anthropic.com', 'model': 'claude-opus-5-5',
                  'auditor': 'claude-sonnet-5-5', 'fast': 'claude-haiku-5-5',
                  'aliases': {'opus': 'claude-opus-5-5', 'sonnet': 'claude-sonnet-5-5', 'haiku': 'claude-haiku-5-5'}},
    'claude-login': {'base_url': 'https://api.anthropic.com', 'model': 'claude-opus-5-5',
                     'auditor': 'claude-sonnet-5-5', 'fast': 'claude-haiku-5-5',
                     'aliases': {'opus': 'claude-opus-5-5', 'sonnet': 'claude-sonnet-5-5', 'haiku': 'claude-haiku-5-5'}},
}
ANY_MODEL = '*'  # a session's models list containing this lets every well-formed model id through
ROUTES = {'/v1/messages': True, '/v1/messages/count_tokens': False}  # path -> counts against the cap
FORWARDED_HEADERS = ('anthropic-version', 'anthropic-beta')
MODEL_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:/@-]{0,199}')


def _private_text(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'r') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError(f'{path} must be a private, owner-readable regular file')
        return stream.read(1 << 20)


def personal_login(path):
    """True for the operator's own Claude login, which experiments must never use."""
    path = Path(path).expanduser()
    personal = (Path.home() / '.claude').resolve()
    resolved = path.resolve()
    return (path.name == '.credentials.json' or resolved == personal
            or personal in resolved.parents or resolved == (Path.home() / '.claude.json').resolve())


class Credential:
    """Provider auth headers, resolved per request so a rotated token applies at once."""

    def __init__(self, provider, *, key_file=None):
        if provider not in PROVIDERS:
            raise ValueError(f'Unknown provider {provider!r}; choose from {sorted(PROVIDERS)}')
        self.provider, self.key_file = provider, key_file
        if provider == 'claude-login':
            if not key_file:
                raise ValueError('claude-login needs the experiment setup-token file; '
                                 'run scripts/setup_claude_experiment_login.sh')
            if personal_login(key_file):
                raise ValueError('claude-login refuses the personal Claude login; '
                                 'use the dedicated experiment token')
        self.headers()  # Fail at startup, not on the first model call.

    def _token(self):
        token = _private_text(self.key_file).strip()
        if token.startswith('{') and 'claudeAiOauth' in token:
            raise ValueError('claude-login refuses a Claude login credentials file; '
                             'use the dedicated experiment token')
        if not token or len(token) > 16384 or any(c.isspace() for c in token):
            raise ValueError('Invalid provider credential')
        return token

    def headers(self):
        if self.provider == 'anthropic':
            key = _private_text(self.key_file).strip() if self.key_file else os.environ.get('ANTHROPIC_API_KEY', '')
            if not key:
                raise ValueError('The anthropic provider needs --key-file or ANTHROPIC_API_KEY')
            return {'x-api-key': key}
        if self.provider == 'openrouter':
            key = _private_text(self.key_file).strip() if self.key_file else os.environ.get('OPENROUTER_API_KEY', '')
            if not key:
                raise ValueError('The openrouter provider needs --key-file or OPENROUTER_API_KEY')
            return {'Authorization': 'Bearer ' + key}
        return {'Authorization': 'Bearer ' + self._token()}


def _media_is_inline(value):
    """No URL or Files-API sources: the provider must never fetch on the agent's behalf."""
    if isinstance(value, list):
        return all(_media_is_inline(item) for item in value)
    if isinstance(value, dict):
        source = value.get('source')
        if isinstance(source, dict) and source.get('type') not in ('base64', 'text', 'content'):
            return False
        return all(_media_is_inline(item) for item in value.values())
    return True


def local_tools(tools):
    """Only client-executed tools; server tools (web search/fetch, code execution) are refused."""
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict) or tool.get('type') not in (None, 'custom'):
            return False
        if not isinstance(tool.get('name'), str):
            return False
    return True


def refusal(payload, allowed):
    """Why the relay refuses a request (a complete sentence for the client), or None to forward it.
    allowed: the session's model ids, or None for any well-formed id."""
    if not isinstance(payload, dict):
        return 'rax relay: the request body is not a JSON object.'
    model = payload.get('model')
    if not isinstance(model, str) or not MODEL_ID.fullmatch(model) or (allowed is not None and model not in allowed):
        enabled = 'any model' if allowed is None else ', '.join(sorted(allowed))
        return f'rax relay: model {str(model)[:80]!r} is not enabled for this session (enabled: {enabled}).'
    if any(key in payload for key in ('mcp_servers', 'container')):
        return 'rax relay: server-side MCP servers and containers are not allowed; use the local tools.'
    if not local_tools(payload.get('tools', [])):
        return 'rax relay: server-side tools are not allowed; only client tools are forwarded.'
    if not (_media_is_inline(payload.get('messages', [])) and _media_is_inline(payload.get('system', []))):
        return 'rax relay: images and documents must be inline (base64), not URLs or file ids.'
    return None


def acceptable(payload, allowed):
    return refusal(payload, allowed) is None


class UsageLog:
    """Parses usage from a Messages response (SSE or JSON) and appends one JSONL row."""

    def __init__(self, path):
        self.path, self.lock = (Path(path) if path else None), threading.Lock()

    def record(self, model, usage):
        if self.path is None or not usage:
            return
        row = {'time': time.time(), 'model': model}
        for key in ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'):
            if isinstance(usage.get(key), int):
                row[key] = usage[key]
        if isinstance(usage.get('cost'), (int, float)):
            row['cost_usd'] = usage['cost']
        self._write(row)

    def error(self, model, status, attempts, body):
        """One row per provider error passed to the client: status and error type, no prompt content."""
        if self.path is None:
            return
        row = {'time': time.time(), 'model': model, 'error_status': status, 'attempts': attempts}
        try:
            detail = json.loads(body).get('error') or {}
            row['error_type'] = str(detail.get('type', ''))[:100]
            row['error_message'] = str(detail.get('message', ''))[:300]
        except (ValueError, AttributeError):
            pass
        self._write(row)

    def stalled(self, model, phase, idle_s, attempts):
        """One row per upstream stream that went silent for idle_s and was given up."""
        if self.path is not None:
            self._write({'time': time.time(), 'model': model, 'error_status': 'stalled', 'phase': phase,
                         'idle_s': idle_s, 'attempts': attempts})

    def _write(self, row):
        with self.lock, open(self.path, 'a') as stream:
            stream.write(json.dumps(row) + '\n')


class StreamUsage:
    """Incremental SSE scan: message_start carries input usage, message_delta the output."""

    def __init__(self):
        self.buffer, self.usage = b'', {}

    def feed(self, block):
        self.buffer += block
        *lines, self.buffer = self.buffer.split(b'\n')
        for line in lines:
            if not line.startswith(b'data:') or b'usage' not in line:
                continue
            try:
                event = json.loads(line[5:])
            except ValueError:
                continue
            usage = (event.get('message') or {}).get('usage') if event.get('type') == 'message_start' \
                else event.get('usage')
            if isinstance(usage, dict):
                self.usage.update({k: v for k, v in usage.items() if v is not None})


def usage_reset(response):
    """Epoch at which an exhausted subscription usage window reopens; None for an ordinary 429."""
    if response.status != 429 or response.getheader('anthropic-ratelimit-unified-status') != 'rejected':
        return None
    try:
        return float(response.getheader('anthropic-ratelimit-unified-reset'))
    except (TypeError, ValueError):
        return None


class MessagesRelay:
    def __init__(self, path, credential, *, base_url, models, max_calls=4000, client_write_timeout=120,
                 usage_log=None, calls_file=None):
        allowed = set()
        for model in models:
            if model == ANY_MODEL:
                allowed = None  # every well-formed model id (checked per request)
                break
            if not isinstance(model, str) or not MODEL_ID.fullmatch(model):
                raise ValueError('Invalid operator model identifier')
            allowed.add(model)
        url = urlsplit(base_url)
        if url.scheme not in {'https', 'http'} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('Invalid operator model endpoint')
        if url.scheme == 'http' and url.hostname not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('Non-TLS endpoint is only allowed for local tests')
        self.url, self.credential, self.max_calls = url, credential, max_calls
        self.counter, self.lock = CallCounter(calls_file), threading.Lock()
        self.usage = UsageLog(usage_log)
        self.limited_until = 0.0  # Subscription usage window exhausted until this epoch.
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_):
                pass  # Never log headers, prompts, provider failures or credentials.

            def reject(self, status, body=b''):
                self.send_response(status)
                if body:
                    self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True

            def do_POST(self):
                upstream = None
                try:
                    self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)
                    parts = urlsplit(self.path)
                    if parts.path not in ROUTES or parts.query not in ('', 'beta=true') \
                            or self.headers.get('Transfer-Encoding'):
                        self.reject(403)
                        return
                    length = int(self.headers.get('Content-Length', '-1'))
                    if not 0 < length <= MAX_BODY:
                        self.reject(413)
                        return
                    body = self.rfile.read(length)
                    if self.headers.get('Content-Encoding') == 'gzip':
                        with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                            body = stream.read(MAX_BODY+1)
                    elif self.headers.get('Content-Encoding'):
                        self.reject(415)
                        return
                    if len(body) > MAX_BODY:
                        self.reject(413)
                        return
                    payload = json.loads(body)
                    reason = refusal(payload, allowed)
                    if reason:
                        self.reject(403, reason.encode())
                        return
                    counted = ROUTES[parts.path]
                    if relay.limited_until > time.time():
                        # Retrying would only wait out the window; let the launcher pause instead.
                        error = json.dumps({'type': 'error', 'error': {'type': 'rate_limit_error',
                                            'message': 'Usage limit reached; the session resumes after it resets'}}).encode()
                        self.send_response(429)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Content-Length', str(len(error)))
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        self.wfile.write(error)
                        self.close_connection = True
                        return
                    if counted:
                        maximum = payload.get('max_tokens')
                        if type(maximum) is not int or not 1 <= maximum <= MAX_OUTPUT_TOKENS:
                            self.reject(400)
                            return
                        with relay.lock:
                            if relay.counter.value >= relay.max_calls:
                                self.reject(402, BUDGET_EXHAUSTED)
                                return
                            relay.counter.increment()
                    try:
                        auth = relay.credential.headers()
                    except (OSError, ValueError, RuntimeError):
                        self.reject(401, b'Relay provider credential unavailable')
                        return
                    headers = {**auth, 'Content-Type': 'application/json', 'Accept-Encoding': 'identity',
                               'Accept': self.headers.get('Accept', 'application/json')}
                    for name in FORWARDED_HEADERS:
                        if self.headers.get(name):
                            headers[name] = self.headers[name]
                    if relay.credential.provider == 'claude-login':
                        betas = [b for b in headers.get('anthropic-beta', '').split(',') if b.strip()]
                        headers['anthropic-beta'] = ','.join(dict.fromkeys([*betas, OAUTH_BETA]))
                    headers.setdefault('anthropic-version', '2023-06-01')
                    connection = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
                    target = url.path.rstrip('/') + parts.path + (f'?{parts.query}' if parts.query else '')
                    deadline, attempt = time.monotonic()+RETRY_WINDOW_SECONDS, 0
                    idle = STREAM_IDLE_SECONDS if payload.get('stream') is True else UPSTREAM_TIMEOUT_SECONDS
                    while True:
                        upstream = connection(url.hostname, url.port, timeout=idle)
                        try:
                            upstream.request('POST', target, body, headers)
                            response = upstream.getresponse()
                        except TimeoutError:
                            relay.usage.stalled(payload.get('model'), 'before the response', idle, attempt + 1)
                            upstream.close()
                            if time.monotonic() >= deadline:
                                raise
                            attempt += 1
                            continue
                        if response.status not in RETRY_STATUSES:
                            break
                        if (reset := usage_reset(response)) is not None:
                            relay.limited_until = max(relay.limited_until, reset)
                            break
                        delay = retry_delay(attempt, response.getheader('Retry-After'))
                        if time.monotonic()+delay >= deadline:
                            break
                        upstream.close()
                        time.sleep(delay)
                        attempt += 1
                    if response.status != 200:
                        # Claude Code needs the error type (e.g. context too long) to react;
                        # provider error bodies are small JSON and carry no credential.
                        error = response.read(65536)
                        relay.usage.error(payload.get('model'), response.status, attempt + 1, error)
                        self.send_response(response.status if 400 <= response.status <= 599 else 502)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Content-Length', str(len(error)))
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        self.wfile.write(error)
                        self.close_connection = True
                        return
                    # A slow client must not truncate a valid response.
                    self.connection.settimeout(client_write_timeout)
                    self.send_response(200)
                    self.send_header('Content-Type', response.getheader('Content-Type', 'text/event-stream'))
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    streaming = 'text/event-stream' in (response.getheader('Content-Type') or '')
                    scan, whole, total = StreamUsage(), [], 0
                    while True:
                        try:
                            block = response.read1(65536)
                        except TimeoutError:
                            relay.usage.stalled(payload.get('model'), 'mid-stream', idle, attempt + 1)
                            # End with a retryable error, then reset (TCP): a cleanly ended stream could pass
                            # for a short answer.
                            if streaming:
                                self.wfile.write(STALLED_EVENT)
                                self.wfile.flush()
                            if self.connection.family != socket.AF_UNIX:
                                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
                            raise
                        if not block:
                            break
                        total += len(block)
                        if total > MAX_BODY:
                            break
                        if streaming:
                            scan.feed(block)
                        elif total <= 1 << 20:
                            whole.append(block)
                        self.wfile.write(block)
                        self.wfile.flush()
                    if counted:
                        usage = scan.usage
                        if not streaming and whole:
                            try:
                                usage = json.loads(b''.join(whole)).get('usage') or {}
                            except ValueError:
                                usage = {}
                        relay.usage.record(payload.get('model'), usage)
                except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
                    self.close_connection = True
                finally:
                    if upstream:
                        upstream.close()

            def do_GET(self):
                self.reject(403)
            do_CONNECT = do_GET

        if isinstance(path, tuple):
            self.server = LoopbackHTTPServer(path, Handler)
        else:
            self.server = UnixHTTPServer(str(path), Handler)
            os.chmod(path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def address(self):
        """(host, port) for a loopback relay; the socket path otherwise."""
        return self.server.server_address

    @property
    def calls(self):
        return self.counter.value

    def start(self):
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
