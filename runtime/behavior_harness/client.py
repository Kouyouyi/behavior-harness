"""Idempotent command submission and receipt polling; no motion retries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
import uuid


class CommandTimeout(TimeoutError):
    """Execution is unknown; query request_id instead of submitting another move."""


class HarnessClient:
    def __init__(self, ready_file, *, timeout_s=600):
        self.output_dir = Path(ready_file).resolve().parent
        ready = json.loads(Path(ready_file).read_text())
        self.url, self.token = ready["url"], ready["token"]
        self.timeout_s = timeout_s
        self.last_request_id = None

    def request(self, method, path, body=None):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={
            "Authorization": "Bearer " + self.token, "Content-Type": "application/json",
        })
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)

    def status(self, request_id):
        if not request_id or not all(c.isalnum() or c in "-_" for c in request_id):
            raise ValueError("invalid request_id")
        try:
            return self.request("GET", "/commands/" + request_id)
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            # close may have shut down HTTP after persisting the final receipt.
            # A disk read reconciles this exact command; it never resends motion.
            path = self.output_dir / "receipts" / f"{request_id}.json"
            if path.exists():
                return json.loads(path.read_text())
            raise

    def call(self, command, payload=None, *, request_id=None):
        rid = request_id or uuid.uuid4().hex
        self.last_request_id = rid
        self.request("POST", "/commands", {"request_id": rid, "command": command, "payload": payload or {}})
        return self.wait(rid)

    def wait(self, request_id):
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            receipt = self.status(request_id)
            if receipt["status"] == "completed":
                return receipt["result"]
            if receipt["status"] == "failed":
                raise RuntimeError(f"Request {request_id} failed; do not replay motion: {receipt['error']}")
            time.sleep(0.3)
        raise CommandTimeout(f"Request {request_id} still unconfirmed; inspect the same receipt before any new motion")

    def stop(self):
        return self.request("POST", "/stop", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", required=True)
    parser.add_argument("command", choices=["observe", "execute", "close", "stop", "status"])
    parser.add_argument("--decision", help="JSON decision file")
    parser.add_argument("--request-id")
    args = parser.parse_args()
    client = HarnessClient(args.ready)
    if args.command == "stop":
        value = client.stop()
    elif args.command == "status":
        value = client.status(args.request_id)
    else:
        payload = json.loads(Path(args.decision).read_text()) if args.decision else {}
        value = client.call(args.command, payload, request_id=args.request_id)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
