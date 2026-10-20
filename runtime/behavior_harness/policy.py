"""Cancellable BEHAVIOR websocket inference without replaying an in-flight request.

All heavy dependencies are lazy. The native BEHAVIOR msgpack protocol is kept:
receive server metadata, send optional reset (no ACK), send observations, receive
one ``action``. Server-side action chunks remain the server's responsibility.
"""

from __future__ import annotations

import math
import socket
import threading
import time
from typing import Any, Callable


class PolicyTransportError(RuntimeError):
    """Connection state is uncertain; stop the host episode, do not retry motion."""


class BoundedWebsocketPolicy:
    POLL_SECONDS = 0.25

    def __init__(self, host="127.0.0.1", port=8000, timeout_s=180, *, context_enabled=False,
                 connect_factory: Callable | None = None,
                 packer: Any = None, unpack: Callable | None = None,
                 tensor_factory: Callable | None = None):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if type(context_enabled) is not bool:
            raise ValueError("context_enabled must be a boolean")
        self.host, self.port = str(host), int(port)
        self.context_enabled = context_enabled
        self._visual_context = None
        self._context_bind_allowed = False
        self.timeout_s = float(timeout_s)
        self.stop_event = threading.Event()
        self.last_action = None
        self._ws = None
        self._server_metadata = None
        self._reset_pending = False
        self._closed = False
        self._failed = False
        self._lock = threading.RLock()
        self._connect_factory = connect_factory
        self._packer, self._unpack = packer, unpack
        self._tensor_factory = tensor_factory

    @property
    def server_metadata(self):
        return self._server_metadata

    def _check(self, deadline, stage):
        if self._closed:
            raise PolicyTransportError("VLA policy connection is closed")
        if self._failed:
            raise PolicyTransportError("VLA policy previously failed; no request will be retried")
        if self.stop_event.is_set():
            raise PolicyTransportError(f"VLA {stage} cancelled by stop request; no inference replay")
        if time.monotonic() >= deadline:
            raise PolicyTransportError(f"VLA {stage} timed out after {self.timeout_s:g}s; no inference replay")

    def _codec(self):
        if self._packer is None or self._unpack is None:
            from omnigibson.eval.utils.network_utils import Packer, unpackb
            self._packer = self._packer or Packer()
            self._unpack = self._unpack or unpackb

    @staticmethod
    def _close_socket(ws):
        # shutdown the transport first: close() alone may wait on a lock held by
        # sendall when the peer stopped consuming a large RGB observation.
        raw = getattr(ws, "socket", None)
        if raw is not None:
            try:
                raw.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                raw.close()
            except OSError:
                pass
        try:
            ws.close()
        except Exception:
            pass

    def _disconnect(self):
        ws, self._ws = self._ws, None
        if ws is not None:
            self._close_socket(ws)

    def _blocking(self, function, deadline, stage, cleanup=None):
        """Bound DNS/connect/send too; only socket work runs in a helper thread.

        A connect which finishes after cancellation closes its late socket. A
        blocked send is interrupted by _disconnect's socket shutdown. No helper
        retries calls, and no simulator/Torch operations run on these threads.
        """
        self._check(deadline, stage)
        done, abandoned, guard = threading.Event(), threading.Event(), threading.Lock()
        result = {}

        def worker():
            try:
                value = function()
            except BaseException as exc:
                with guard:
                    result["error"] = exc
            else:
                with guard:
                    if abandoned.is_set():
                        if cleanup is not None:
                            cleanup(value)
                    else:
                        result["value"] = value
            finally:
                done.set()

        threading.Thread(target=worker, name=f"behavior-vla-{stage}", daemon=True).start()
        try:
            while True:
                self._check(deadline, stage)
                if done.wait(min(self.POLL_SECONDS, max(0.0, deadline - time.monotonic()))):
                    self._check(deadline, stage)
                    break
        except BaseException:
            with guard:
                abandoned.set()
                if cleanup is not None and "value" in result:
                    cleanup(result.pop("value"))
            raise
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _receive(self, deadline, stage):
        while True:
            self._check(deadline, stage)
            try:
                value = self._ws.recv(timeout=min(self.POLL_SECONDS, max(0.001, deadline - time.monotonic())))
            except TimeoutError:
                # Only receive is repeated. The observation is never sent twice.
                continue
            self._check(deadline, stage)
            return value

    def _send(self, payload, deadline, stage):
        self._check(deadline, stage)
        data = self._packer.pack(payload)
        self._blocking(lambda: self._ws.send(data), deadline, stage)

    def _connect(self, deadline):
        if self._ws is not None:
            return
        if self._connect_factory is None:
            from websockets.sync.client import connect
            self._connect_factory = connect
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        uri = f"ws://{host}:{self.port}"
        while self._ws is None:
            self._check(deadline, "connect")
            try:
                # Availability is probed by the handshake itself; there is no
                # separate unbounded HTTP health request or inference replay.
                self._ws = self._blocking(
                    lambda: self._connect_factory(
                        uri, open_timeout=min(1.0, max(0.001, deadline - time.monotonic())),
                        close_timeout=self.POLL_SECONDS, compression=None, max_size=None,
                        ping_interval=60, ping_timeout=30, proxy=None),
                    deadline, "connect", cleanup=self._close_socket)
            except (OSError, TimeoutError):
                self._check(deadline, "connect")
                self.stop_event.wait(min(self.POLL_SECONDS, max(0.0, deadline - time.monotonic())))
        raw = self._receive(deadline, "metadata")
        if isinstance(raw, str):
            raise PolicyTransportError("VLA server returned text instead of binary metadata")
        metadata = self._unpack(raw)
        if not isinstance(metadata, dict):
            raise PolicyTransportError("VLA metadata must be a msgpack mapping")
        self._server_metadata = metadata
        if self.context_enabled:
            protocol = metadata.get("harness_context_protocol")
            if type(protocol) is not int or protocol != 1:
                raise PolicyTransportError("VLA server does not support harness context protocol 1")
        if self._reset_pending:
            self._send({"reset": True}, deadline, "reset")
            self._reset_pending = False

    def _action_tensor(self, value):
        if self._tensor_factory is None:
            import torch
            tensor = torch.as_tensor(value, dtype=torch.float32)
        else:
            tensor = self._tensor_factory(value)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != 23 or tensor.shape[0] < 1:
            raise PolicyTransportError("VLA action must have shape (batch, 23)")
        if not bool(tensor.isfinite().all()):
            raise PolicyTransportError("VLA action contains nonfinite values")
        return tensor.detach().cpu()

    def forward(self, obs: dict, *args, **kwargs):
        with self._lock:
            deadline = time.monotonic() + self.timeout_s
            try:
                self._check(deadline, "inference")
                outgoing = obs
                if self.context_enabled:
                    from .context_policy_service import WIRE_KEY
                    if WIRE_KEY in obs:
                        raise PolicyTransportError("Bind visual context with set_visual_context, not inside observations")
                    self._context_bind_allowed = False
                    outgoing = dict(obs)
                    if self._visual_context is not None:
                        outgoing[WIRE_KEY] = dict(self._visual_context)
                if "need_new_action" in obs and not bool(obs["need_new_action"]) and self.last_action is not None:
                    return self.last_action.clone()
                self._codec()
                self._connect(deadline)
                self._send(outgoing, deadline, "send")
                raw = self._receive(deadline, "inference")
                if isinstance(raw, str):
                    raise PolicyTransportError("VLA server returned an error text response")
                response = self._unpack(raw)
                if not isinstance(response, dict) or "action" not in response:
                    raise PolicyTransportError("VLA response has no action; inference will not be resent")
                action = self._action_tensor(response["action"])
                self.last_action = action.clone()
                return action
            except BaseException as exc:
                self._failed = True
                self.last_action = None
                self._disconnect()
                if isinstance(exc, (PolicyTransportError, KeyboardInterrupt, SystemExit)):
                    raise
                raise PolicyTransportError(f"VLA transport failed ({type(exc).__name__}); inference will not be replayed") from exc

    def set_visual_context(self, mapping):
        """Bind locally after reset; handshake and wire transmission wait for forward.

        Context versions are scoped to one episode/service, not to connections.
        Reset clears this client's mapping but does not restart server version IDs.
        """
        with self._lock:
            self._check(time.monotonic() + self.timeout_s, "context binding")
            if not self.context_enabled:
                raise PolicyTransportError("Visual language context is disabled")
            from .context_policy_service import validate_context
            context = validate_context(mapping)
            if not self._context_bind_allowed:
                if context == self._visual_context:
                    return
                raise PolicyTransportError("Visual context is frozen; reset before binding a new context")
            self._visual_context = context
            self._context_bind_allowed = False

    def reset(self):
        with self._lock:
            self.last_action = None
            self._visual_context = None
            self._context_bind_allowed = False
            deadline = time.monotonic() + self.timeout_s
            try:
                self._check(deadline, "reset")
                if self._ws is None:
                    self._reset_pending = True
                else:
                    self._codec()
                    self._send({"reset": True}, deadline, "reset")
                    self._reset_pending = False
                self._context_bind_allowed = self.context_enabled
            except BaseException as exc:
                self._failed = True
                self._disconnect()
                if isinstance(exc, (PolicyTransportError, KeyboardInterrupt, SystemExit)):
                    raise
                raise PolicyTransportError(f"VLA reset failed ({type(exc).__name__}); do not retry this episode") from exc

    def close(self):
        self._closed = True
        self.last_action = None
        self._reset_pending = False
        self._visual_context = None
        self._context_bind_allowed = False
        self._disconnect()
