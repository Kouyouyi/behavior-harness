"""Loopback RPC queue; all simulator operations execute on its owning thread."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import secrets
import signal
import threading
import traceback

from .contracts import ControlLimits, validate_decision


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


class HarnessService:
    def __init__(self, backend, output_dir, *, token=None, limits=None, stop_event=None):
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token = token or secrets.token_urlsafe(32)
        self.limits = limits or ControlLimits()
        self.pending = queue.Queue()
        self.receipts = {}
        self.lock = threading.Lock()
        self.observation = None
        self.closed = False
        self._finalized = False
        self.stop_event = stop_event or threading.Event()
        self.http = None

    def audit(self, event, **data):
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        with (self.output_dir / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def submit(self, request):
        if set(request) != {"request_id", "command", "payload"}:
            raise ValueError("request requires exactly request_id, command and payload")
        rid, command = request["request_id"], request["command"]
        if not isinstance(rid, str) or not 1 <= len(rid) <= 100 or not all(c.isalnum() or c in "-_" for c in rid):
            raise ValueError("invalid request_id")
        if command not in {"observe", "execute", "close"} or not isinstance(request["payload"], dict):
            raise ValueError("invalid command or payload")
        canonical = json.dumps(request, sort_keys=True, allow_nan=False)
        with self.lock:
            if rid in self.receipts:
                if self.receipts[rid]["request"] != canonical:
                    raise ValueError("request_id already bound to a different command")
                return self.public_receipt(rid)
            if self.closed or (self.stop_event.is_set() and command != "close"):
                raise ValueError("session is stopping or closed")
            self.receipts[rid] = {"request": canonical, "status": "queued", "request_id": rid}
            self.pending.put(request)
            return self.public_receipt(rid)

    def public_receipt(self, rid):
        return {k: v for k, v in self.receipts[rid].items() if k != "request"}

    def _save_receipt(self, rid):
        atomic_json(self.output_dir / "receipts" / f"{rid}.json", self.public_receipt(rid))

    def observe(self):
        value = self.backend.observe()
        value.setdefault("capabilities", {})["limits"] = asdict(self.limits)
        self.observation = value
        atomic_json(self.output_dir / "latest_observation.json", value)
        return value

    def execute_request(self, request):
        rid, command, payload = request["request_id"], request["command"], request["payload"]
        self.receipts[rid]["status"] = "executing"
        self._save_receipt(rid)
        self.audit("request_started", request_id=rid, command=command, payload=payload)
        simulation_entered = False
        terminal_fields = {}
        try:
            if command == "observe":
                if payload:
                    raise ValueError("observe takes no parameters")
                simulation_entered = True
                result = {"observation": self.observe()}
            elif command == "close":
                if set(payload) - {"reason"}:
                    raise ValueError("close only accepts reason")
                simulation_entered = True
                result = self.backend.close(reason=str(payload.get("reason", "user_stop")))
                self.closed = True
            else:
                if self.observation is None:
                    raise ValueError("observe before acting")
                obs = self.observation
                robot, caps = obs["robot"], obs.get("capabilities", {})
                validated = validate_decision(
                    payload, current_observation_id=obs["observation_id"], limits=self.limits,
                    trunk_qpos=robot["trunk_qpos"], trunk_limits=robot["trunk_limits"],
                    phase=obs["phase"], remaining_steps=obs["remaining_steps"],
                    head_enabled=bool(caps.get("gimbal_enabled", False)),
                    head_state=robot.get("head_gimbal"),
                )
                simulation_entered = True
                if validated["action"] == "stop":
                    result = self.backend.close(reason=validated["stop_reason"])
                    self.closed = True
                elif validated["action"] == "observe":
                    result = {"observation": self.observe()}
                else:
                    execution = self.backend.execute(validated)
                    result = {"execution": execution, "observation": self.observe()}
            terminal_fields = {"status": "completed", "result": result}
            self.audit("request_completed", request_id=rid, result=result)
        except Exception as exc:
            terminal_fields = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            self.audit("request_failed", request_id=rid, error=str(exc), traceback=traceback.format_exc())
            # Validation failures do not end an otherwise unchanged episode. Runtime
            # failures may have partially executed: the receipt is never replayed.
            if simulation_entered or not isinstance(exc, ValueError):
                self.stop_event.set()
        finally:
            # Publish a terminal status only after its durable receipt exists.
            # A client may immediately lose the connection during simulator close.
            final_receipt = {**self.public_receipt(rid), **terminal_fields}
            atomic_json(self.output_dir / "receipts" / f"{rid}.json", final_receipt)
            with self.lock:
                self.receipts[rid].update(terminal_fields)

    def start_http(self, port=0):
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, status, data):
                encoded = json.dumps(data, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def authorized(self):
                supplied = self.headers.get("Authorization", "")
                return secrets.compare_digest(supplied, "Bearer " + service.token)

            def do_GET(self):
                if not self.authorized():
                    return self.reply(401, {"error": "unauthorized"})
                if self.path == "/health":
                    return self.reply(200, {"ready": not service.closed, "stopping": service.stop_event.is_set()})
                rid = self.path.removeprefix("/commands/")
                with service.lock:
                    receipt = service.public_receipt(rid) if rid in service.receipts else None
                self.reply(200 if receipt else 404, receipt or {"error": "unknown request"})

            def do_POST(self):
                if not self.authorized():
                    return self.reply(401, {"error": "unauthorized"})
                if self.path == "/stop":
                    service.stop_event.set()
                    return self.reply(202, {"status": "stopping"})
                if self.path != "/commands" or self.headers.get("Content-Type") != "application/json":
                    return self.reply(400, {"error": "use JSON POST /commands"})
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 65536:
                        raise ValueError("invalid body length")
                    request = json.loads(self.rfile.read(size))
                    self.reply(202, service.submit(request))
                except (ValueError, TypeError) as exc:
                    self.reply(400, {"error": str(exc)})

        self.http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        return self.http.server_address[1]

    def run(self):
        error = None
        try:
            while not self.closed and not self.stop_event.is_set():
                try:
                    request = self.pending.get(timeout=0.2)
                except queue.Empty:
                    continue
                self.execute_request(request)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.finalize(reason="stop_requested" if self.stop_event.is_set() else "service_exit", error=error)

    def finalize(self, *, reason, error=None):
        """Persist terminal state and stop HTTP before potentially native exit."""
        if self._finalized:
            return
        close_error = None
        try:
            if not self.closed:
                result = self.backend.close(reason=reason)
                self.audit("session_closed", result=result)
        except Exception as exc:
            close_error = exc
            error = error or f"{type(exc).__name__}: {exc}"
            self.audit("session_close_failed", error=str(exc), traceback=traceback.format_exc())
        self.closed = True
        # execute_request has already persisted the terminal command receipt.
        # Startup failures have no command, so service_status is their receipt.
        status = {"status": "failed" if error else "closed", "reason": reason}
        if error:
            status["error"] = error
        atomic_json(self.output_dir / "service_status.json", status)
        try:
            if self.http is not None:
                try:
                    self.http.shutdown()
                finally:
                    self.http.server_close()
        finally:
            self._finalized = True
            # OG may terminate the entire process here, including via C++ exit.
            # Nothing essential may depend on Python resuming after this call.
            self.backend.shutdown_runtime()
        if close_error is not None:
            raise close_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    from .backend import BehaviorBackend
    config["stop_event"] = threading.Event()
    backend = BehaviorBackend(config)
    service = HarnessService(backend, config["output_dir"], stop_event=config["stop_event"])
    try:
        backend.start()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: service.stop_event.set())
        if config.get("run_mode", "assisted") == "baseline":
            backend.execute({"action": "handoff", "reasoning": "host starts VLA-only baseline"})
        port = service.start_http(args.port)
        service.observe()
        atomic_json(service.output_dir / "ready.json", {
            "url": f"http://127.0.0.1:{port}", "token": service.token,
            "pid": __import__("os").getpid(), "run_mode": config.get("run_mode", "assisted"),
        })
        service.run()
    except BaseException as exc:
        if not service._finalized:
            service.audit("startup_or_service_failed", error=str(exc), traceback=traceback.format_exc())
            service.finalize(reason="startup_or_service_error", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
