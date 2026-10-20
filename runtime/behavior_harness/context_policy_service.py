"""Optional RGB-language experiment; not used by the normal harness launcher.

Run with the OpenPI interpreter and this workspace on PYTHONPATH. ``--help`` and
imports need only the standard library. Actual serving explicitly loads the same
official checkpoint, B1K wrapper, and websocket server as serve_b1k.py.

After a websocket reset, the first observation may carry __harness_context__:
{version: 1, context_id: "handoff-one", source_observation_id:
 "step-00000150-obs-000015", source_step: 150, source_kind: "rgb",
 text: "The radio is on the low cabinet below the television, directly ahead."}
Subsequent observations may omit it, or repeat the identical mapping. A different
version needs another reset and a fresh source observation. Context is single-env.
Use one service process and audit file per episode. Version/ID/source high-water
marks last for that service's lifetime; reconnecting does not restart version 1.
Without active context, official prompt truncation remains unchanged; with context,
the complete task + discrete proprio token sequence must fit within 200 tokens.

The source ID is an audit reference, not proof that the caller used only RGB.
The caller must enforce the public observation boundary; this service cannot
authenticate image provenance from a text claim. Numeric locations, scene-state
fields, simulator identifiers, and coordinate terminology are rejected here.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path
import re
import time


WIRE_KEY = "__harness_context__"
MAX_TEXT_CHARS = 240
MAX_TEXT_WORDS = 40
MAX_TOKENS = 200
_CONTEXT_KEYS = {"version", "context_id", "source_observation_id", "source_step", "source_kind", "text"}
_OBSERVATION_ID = re.compile(r"step-(\d{8,12})-obs-(\d{6,12})\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,95}\Z")
_FORBIDDEN = re.compile(
    r"\b(world|global|coordinate\w*|pose\w*|aabb|quaternion\w*|radian\w*|"
    r"meter\w*|metre\w*|centimeter\w*|centimetre\w*|ground\s+truth|"
    r"simulator|privileged|object\s+scope|reward|success|ignore|system\s+prompt|"
    r"position\w*|axis|xyz|[xyz]|joint\w*|state|zero|one|two|three|four|five|six|"
    r"seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million)\b", re.I)


class ContextError(ValueError):
    """Invalid context must stop this request before an action is returned."""


def validate_context(value):
    if type(value) is not dict or set(value) != _CONTEXT_KEYS:
        raise ContextError("Context must contain exactly the declared protocol fields")
    if type(value["version"]) is not int or not 1 <= value["version"] <= 2**31 - 1:
        raise ContextError("version must be a positive integer")
    if type(value["context_id"]) is not str or not _IDENTIFIER.fullmatch(value["context_id"]):
        raise ContextError("context_id must be a bounded identifier, not a path")
    source = value["source_observation_id"]
    match = _OBSERVATION_ID.fullmatch(source) if type(source) is str else None
    if match is None or int(match[2]) < 1:
        raise ContextError("source_observation_id must identify a saved harness observation")
    step = value["source_step"]
    if type(step) is not int or step < 0 or step != int(match[1]):
        raise ContextError("source_step must match source_observation_id")
    if value["source_kind"] != "rgb" or type(value["source_kind"]) is not str:
        raise ContextError("Only RGB-derived context is supported")
    text = value["text"]
    if type(text) is not str or not 1 <= len(text) <= MAX_TEXT_CHARS or text != text.strip():
        raise ContextError("text must be nonempty, trimmed, and at most 240 characters")
    if len(text.split()) > MAX_TEXT_WORDS or not re.fullmatch(r"[A-Za-z ,.'!?;:\-]+", text):
        raise ContextError("Use short English visual descriptions without numbers, IDs, or structured data")
    if not re.search(r"[A-Za-z]", text) or _FORBIDDEN.search(text):
        raise ContextError("Context must not contain privileged state, coordinates, or result claims")
    return dict(value)


class ContextSession:
    def __init__(self, original_prompt, audit_path):
        if not isinstance(original_prompt, str) or not original_prompt.strip():
            raise ValueError("An original task prompt is required")
        self.original_prompt = original_prompt
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        # Each service must use its own file, never append to another experiment.
        with self.audit_path.open("x", encoding="utf-8"):
            pass
        self.phase = 0
        self.context = None
        self._open = False
        self._max_version = 0
        self._seen_ids = set()
        self._last_source = None
        self._first_applied = False
        self.infer_count = 0
        self.token_count = None
        self.audit("service_created", **self.prompt_record())

    @property
    def effective_prompt(self):
        if self.context is None:
            return self.original_prompt
        return self.original_prompt + "\nCurrent visual context: " + self.context["text"]

    def prompt_record(self):
        return {"original_prompt": self.original_prompt, "effective_prompt": self.effective_prompt,
                "context": self.context, "language_context_active": self.context is not None,
                "assistance_type": "visual_localization_and_language_context" if self.context else "none"}

    def audit(self, event, **fields):
        row = {"event": event, "unix_time": time.time(), "phase": self.phase, **fields}
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reset(self):
        self.phase += 1
        self.context = None
        self._open = True
        self._first_applied = False
        self.audit("reset", **self.prompt_record())

    def accept(self, obs):
        supplied = validate_context(obs[WIRE_KEY]) if WIRE_KEY in obs else None
        if self._open:
            if supplied is not None:
                source = _OBSERVATION_ID.fullmatch(supplied["source_observation_id"])
                source_order = (int(source[1]), int(source[2]))
                if supplied["version"] <= self._max_version or supplied["context_id"] in self._seen_ids:
                    raise ContextError("A new phase requires a new, increasing context version and ID")
                if self._last_source is not None and source_order <= self._last_source:
                    raise ContextError("A new context requires a newer source observation")
                self._max_version = supplied["version"]
                self._seen_ids.add(supplied["context_id"])
                self._last_source = source_order
            self.context = supplied
            self._open = False
            self.audit("phase_bound", **self.prompt_record())
        elif supplied is not None and supplied != self.context:
            raise ContextError("Context is frozen; send reset before changing it")
        return {key: value for key, value in obs.items() if key != WIRE_KEY}


class _BudgetEncoder:
    """Check the actual SentencePiece result before official code can truncate it."""
    def __init__(self, encoder, on_count, enforce=lambda: True):
        self.encoder, self.on_count, self.enforce = encoder, on_count, enforce

    def encode(self, *args, **kwargs):
        tokens = self.encoder.encode(*args, **kwargs)
        self.on_count(len(tokens))
        if self.enforce() and len(tokens) > MAX_TOKENS:
            raise ContextError(f"Full task + discrete proprio prompt has {len(tokens)} tokens; limit is 200")
        return tokens

    def __getattr__(self, name):
        return getattr(self.encoder, name)


def install_token_guard(policy, session, *, token_transform_class=None):
    """Reuse the policy's loaded tokenizer; neither model nor tokenizer is loaded here."""
    if token_transform_class is None:
        from openpi.transforms import TokenizePrompt
        token_transform_class = TokenizePrompt
    composite = policy._input_transform
    transforms = list(composite.transforms)
    indices = [i for i, transform in enumerate(transforms) if isinstance(transform, token_transform_class)]
    if len(indices) != 1:
        raise ContextError("Expected exactly one official TokenizePrompt transform")
    index = indices[0]
    transform = transforms[index]
    if not transform.discrete_state_input or getattr(transform.tokenizer, "_max_len", None) != MAX_TOKENS:
        raise ContextError("This experiment requires pi05 discrete proprio with a 200-token budget")
    tokenizer = copy.copy(transform.tokenizer)
    if not callable(getattr(getattr(tokenizer, "_tokenizer", None), "encode", None)):
        raise ContextError("Unsupported tokenizer: cannot enforce the full token budget")
    tokenizer._tokenizer = _BudgetEncoder(tokenizer._tokenizer,
                                          lambda count: setattr(session, "token_count", count),
                                          enforce=lambda: session.context is not None)
    transforms[index] = dataclasses.replace(transform, tokenizer=tokenizer)
    policy._input_transform = dataclasses.replace(composite, transforms=transforms)


class _AuditedPolicy:
    def __init__(self, policy, session):
        self.policy, self.session = policy, session

    def reset(self):
        return self.policy.reset()

    def infer(self, obs, **kwargs):
        session = self.session
        session.infer_count += 1
        session.token_count = None
        fields = {"infer_index": session.infer_count, **session.prompt_record()}
        session.audit("infer_started", **fields)
        try:
            result = self.policy.infer(obs, **kwargs)
            if session.token_count is None:
                raise ContextError("The strict tokenizer guard was not invoked; refusing the action")
        except Exception as exc:
            session.audit("infer_rejected_or_failed", **fields, token_count=session.token_count,
                          error=f"{type(exc).__name__}: {exc}")
            raise
        first = session.context is not None and not session._first_applied
        session.audit("infer_completed", **fields, token_count=session.token_count, first_context_infer=first)
        session._first_applied = session._first_applied or first
        return result


def make_context_wrapper(policy, *, robot, original_prompt, audit_path, action_horizon=16,
                         max_len=32, wrapper_class=None, token_transform_class=None):
    """Wrap a loaded policy. Dependency injection exists only for CPU fake tests."""
    if wrapper_class is None:
        from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
        wrapper_class = B1KPolicyWrapper
    session = ContextSession(original_prompt, audit_path)
    install_token_guard(policy, session, token_transform_class=token_transform_class)

    class ContextWrapper(wrapper_class):
        def reset(self):
            super().reset()
            session.reset()

        def process_input(self, obs):
            inputs = super().process_input(obs)
            if session.context is not None:
                if len(inputs) != 1:
                    raise ContextError("Language context supports exactly one environment")
                for item in inputs:
                    item["prompt"] = session.effective_prompt
            return inputs

        def act(self, obs):
            try:
                return super().act(session.accept(obs))
            except Exception as exc:
                session.audit("request_failed", error=f"{type(exc).__name__}: {exc}")
                raise

    wrapper = ContextWrapper(policy=_AuditedPolicy(policy, session), robot=robot,
                             text_prompt=original_prompt, control_mode="receding_horizon",
                             action_horizon=action_horizon, max_len=max_len)
    wrapper.context_session = session
    return wrapper


def capability_metadata(original=None):
    return {**(original or {}), "harness_context_protocol": 1,
            "harness_context": {"wire_key": WIRE_KEY, "reset_required": True, "phase_frozen": True,
                                "single_environment": True, "single_controller_connection": True,
                                "max_text_chars": MAX_TEXT_CHARS,
                                "max_text_words": MAX_TEXT_WORDS, "max_prompt_tokens": MAX_TOKENS,
                                "service_scope": "single_episode", "version_scope": "service_lifetime",
                                "context_free_token_overflow": "official_truncation",
                                "context_token_overflow": "reject",
                                "source_kind": "rgb", "provenance_verified_by_server": False,
                                "assistance_type": "visual_localization_and_language_context"}}


def make_context_server(wrapper, *, host, port, metadata, server_class=None):
    """Reuse the official protocol, allowing only one controller connection.

    Disconnect clears the old episode context, so a later context-free client
    never inherits a previous client's language. Each new connection must still
    send an explicit reset before binding a context.
    """
    if server_class is None:
        from openpi.serving.websocket_b1k_server import WebsocketPolicyServer
        server_class = WebsocketPolicyServer

    class SingleControllerServer(server_class):
        _context_client_active = False

        async def _handler(self, websocket):
            if self._context_client_active:
                await websocket.close(code=1013, reason="This experiment allows one controller connection")
                return
            self._context_client_active = True
            try:
                await super()._handler(websocket)
            finally:
                try:
                    wrapper.reset()
                    wrapper.context_session._open = False
                    wrapper.context_session.audit("controller_disconnected")
                finally:
                    self._context_client_active = False

    return SingleControllerServer(wrapper, host=host, port=port, metadata=metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audit", required=True, help="New JSONL file for this independent experiment")
    parser.add_argument("--robot", default="b1k/R1Pro")
    parser.add_argument("--task", default="b1k/turning_on_radio")
    parser.add_argument("--config", default="pi05_b1k")
    parser.add_argument("--repo-id", default="turning_on_radio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    args = parser.parse_args(argv)
    # Runtime imports are intentionally after argument parsing. Tests do not load
    # weights, initialize a GPU, or download a tokenizer.
    from openpi.configs.tasks import TASK_REGISTRY
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config
    config = get_config(args.config)
    if not config.model.pi05 or not config.model.discrete_state_input or config.model.max_token_len != MAX_TOKENS:
        raise ContextError("Only pi05 with a 200-token discrete-state input is supported")
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=args.repo_id,
                                                                robot_config_name=args.robot))
    bucket, task = args.task.split("/", 1)
    prompt = TASK_REGISTRY[bucket][task]
    policy = create_trained_policy(config, args.checkpoint, default_prompt=prompt)
    wrapper = make_context_wrapper(policy, robot=args.robot, original_prompt=prompt, audit_path=args.audit,
                                   max_len=config.model.action_horizon)
    server = make_context_server(wrapper, host=args.host, port=args.port,
                                 metadata=capability_metadata(policy.metadata))
    server.serve_forever()


if __name__ == "__main__":
    main()
