"""One-shot multimodal Codex planning with no robot or simulator dependencies.

Uses the user's existing Codex authentication in place. Never reads/copies auth
files, resumes a thread, retries a failed decision, or executes a returned action.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import Any
import uuid

from .prompts import (DECISION_SCHEMA, GIMBAL_SENSOR_FRAMING_INSTRUCTIONS,
                      LANGUAGE_CONTEXT_INSTRUCTIONS, MANIPULATION_PROPRIO_INSTRUCTIONS,
                      PLANNER_INSTRUCTIONS, build_prompt, gimbal_sensor_framing_preferred)


class PlannerError(RuntimeError):
    """No decision is safe to execute; diagnostic files remain in run_dir."""

    def __init__(self, message: str, run_dir: Path | None = None):
        self.run_dir = run_dir
        super().__init__(message + (f" (logs: {run_dir})" if run_dir else ""))


class PlannerTimeout(PlannerError):
    pass


# Disable tools at registration, not just through natural-language instructions.
# --ignore-user-config excludes configured MCPs, hooks, plugins and developer text;
# supported default model preferences are copied as scalar CLI overrides below.
_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "code_mode_host", "code_mode", "code_mode_only",
    "apps", "plugins", "remote_plugin", "hooks", "memories", "skill_search",
    "computer_use", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
    "in_app_browser", "image_generation", "view_image", "multi_agent", "multi_agent_v2",
    "goals", "sleep_tool", "tool_suggest", "workspace_dependencies", "artifact",
    "agent_message_board", "send_message_to_user_async", "send_async_message",
)
_PREFERENCE_KEYS = ("model", "model_provider", "model_reasoning_effort",
                    "model_reasoning_summary", "model_verbosity")


def validate_planner_transport(value: Any, *, mode: str = "assisted") -> str:
    if type(value) is not str or value not in ("auto", "chatgpt_https"):
        raise ValueError("planner_transport must be auto or chatgpt_https")
    if value != "auto" and mode != "assisted":
        raise ValueError("Nondefault planner transport is only allowed for assisted episodes")
    return value


def _default_preferences() -> dict[str, str]:
    """Read only model preferences, never credentials or arbitrary user overrides."""
    config = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
    if not config.is_file():
        return {}
    try:
        raw = config.read_text(encoding="utf-8")
        try:
            import tomllib
        except ImportError:  # Python 3.10 simulator environments do not include tomllib.
            top = raw.split("\n[", 1)[0]
            parsed = {}
            for line in top.splitlines():
                match = re.match(r"\s*(\w+)\s*=\s*(\"(?:[^\"\\]|\\.)*\"|'[^']*')\s*(?:#.*)?$", line)
                if match:
                    value = match[2]
                    parsed[match[1]] = json.loads(value) if value.startswith('"') else value[1:-1]
        else:
            parsed = tomllib.loads(raw)
    except (OSError, ValueError) as exc:
        raise PlannerError("Cannot read Codex model preferences") from exc
    if parsed.get("model_provider", "openai") != "openai":
        raise PlannerError("This planner requires the existing OpenAI Codex provider; custom providers are not inherited")
    return {key: parsed[key] for key in _PREFERENCE_KEYS if isinstance(parsed.get(key), str)}


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_planner_decision(decision: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """Check transport/schema semantics; physical bounds belong to the executor."""
    fields = DECISION_SCHEMA["properties"]
    if not isinstance(decision, dict) or set(decision) != set(fields):
        raise ValueError("Decision must contain exactly the required schema fields")
    for key, shape in fields.items():
        value = decision[key]
        kind = shape["type"]
        valid = {
            "string": isinstance(value, str), "boolean": type(value) is bool,
            "number": type(value) in (int, float) and math.isfinite(value),
            "integer": type(value) is int, "array": isinstance(value, list),
        }[kind]
        if not valid:
            raise ValueError(f"Invalid type or nonfinite value for {key}")
    action = decision["action"]
    if action not in fields["action"]["enum"]:
        raise ValueError("Unknown action")
    if decision["observation_id"] != observation.get("observation_id"):
        raise ValueError("Decision refers to a stale or different observation")
    if not decision["reasoning"].strip():
        raise ValueError("Decision is missing its visual rationale")
    if any(type(q) not in (int, float) or not math.isfinite(q) for q in decision["trunk_qpos"]):
        raise ValueError("trunk_qpos must contain finite numbers")
    if len(decision["trunk_qpos"]) != (4 if action == "look" else 0):
        raise ValueError("look requires four trunk joints; other actions require an empty array")
    if action != "move_base" and any(decision[key] != 0 for key in ("vx_mps", "vy_mps", "yaw_rps")):
        raise ValueError("Only move_base may set base velocities")
    if action != "look_head" and any(decision[key] != 0 for key in ("head_yaw_rad", "head_pitch_rad")):
        raise ValueError("Only look_head may set head angles")
    if action == "look_head" and observation.get("capabilities", {}).get("gimbal_enabled") is not True:
        raise ValueError("look_head requires an enabled independent-head capability")
    if action in ("observe", "handoff", "stop"):
        if decision["steps"] != 0:
            raise ValueError("Nonmoving decisions must have zero steps")
    elif decision["steps"] <= 0:
        raise ValueError("Moving/VLA decisions require positive steps")
    if action == "handoff" and (not decision["target_visible"] or not decision["manipulation_ready"]):
        raise ValueError("Handoff requires visible target and manipulation readiness")
    if action == "handoff" and observation.get("phase") != "gpt":
        raise ValueError("Handoff requires GPT phase")
    if action == "handoff" and observation.get("capabilities", {}).get("language_context_enabled") is True:
        from .context_policy_service import validate_context

        validate_context({"version": 1, "context_id": "planner-validation",
                          "source_observation_id": observation["observation_id"],
                          "source_step": observation.get("step"), "source_kind": "rgb",
                          "text": decision["reasoning"]})
    if action == "run_vla" and observation.get("phase") != "vla":
        raise ValueError("run_vla requires VLA phase")
    if ((action == "stop" and not decision["stop_reason"].strip()) or
            (action != "stop" and decision["stop_reason"] != "")):
        raise ValueError("stop_reason must be nonempty only for stop")
    return decision


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError("Duplicate JSON key")
        data[key] = value
    return data


def _assert_no_tool_events(stdout: str) -> dict[str, Any]:
    started = completed = False
    reconnect_events = warning_items = 0
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item", {})
        # The CLI emits startup warnings as error items (e.g. a deliberately
        # disabled code-mode host). These are diagnostics, not executed tools.
        if isinstance(item, dict) and item.get("type") not in (None, "agent_message", "reasoning", "error"):
            raise ValueError(f"Planner emitted a forbidden tool/event item: {item.get('type')}")
        if event.get("type") == "turn.failed":
            raise ValueError("Codex reported a failed turn")
        if completed:
            raise ValueError("Codex emitted events after its final completed turn")
        if isinstance(item, dict) and item.get("type") == "error":
            warning_items += 1
        kind = event.get("type")
        if kind == "turn.started":
            if started:
                raise ValueError("Codex emitted more than one turn")
            started = True
        elif kind == "error":
            # Codex emits nonterminal transport reconnects as top-level errors.
            # Accept only this known diagnostic, and only if this same turn
            # subsequently completes. All other errors remain terminal here.
            message = event.get("message", "")
            if (not started or not isinstance(message, str) or
                    not re.match(r"^Reconnecting\.\.\.\s+(?:\d+/\d+(?:\s|$)|waiting for network \(Connection failed:)", message)):
                raise ValueError("Codex reported a non-recoverable error")
            reconnect_events += 1
        elif kind == "turn.completed":
            if not started:
                raise ValueError("Codex completed a turn that was not started")
            completed = True
    if not completed:
        raise ValueError("Codex turn did not complete; no decision may be executed")
    return {"cli_reconnect_events": reconnect_events,
            "cli_recovered_after_reconnect": reconnect_events > 0,
            "cli_warning_items": warning_items}


class CodexPlanner:
    def __init__(self, output_dir: str | Path, model: str | None = None,
                 timeout_s: float = 180, codex_bin: str = "codex", *, transport: str = "auto"):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.output_dir = Path(output_dir).resolve()
        self.model = model
        self.timeout_s = timeout_s
        self.codex_bin = codex_bin
        self.transport = validate_planner_transport(transport)
        self.last_run_dir: Path | None = None

    def _command(self, run_dir: Path, workspace: Path, images: list[Path], *,
                 language_context_enabled=False, manipulation_proprio_enabled=False,
                 gimbal_sensor_framing_preferred=False) -> list[str]:
        command = [self.codex_bin, "exec", "--ignore-user-config", "--ignore-rules",
                   "--strict-config", "--ephemeral", "--skip-git-repo-check",
                   "--sandbox", "read-only", "--color", "never", "--json",
                   "--cd", str(workspace), "--output-schema", str(run_dir / "schema.json"),
                   "--output-last-message", str(run_dir / "decision.raw.json")]
        config: dict[str, Any] = _default_preferences()
        if self.model is not None:
            config["model"] = self.model
        config.update({"approval_policy": "never", "web_search": "disabled",
                       "project_doc_max_bytes": 0, "skills.include_instructions": False,
                       "mcp_servers": {}, "plugins": {},
                       "include_apps_instructions": False,
                       "developer_instructions": PLANNER_INSTRUCTIONS + (
                           LANGUAGE_CONTEXT_INSTRUCTIONS if language_context_enabled else "") + (
                           MANIPULATION_PROPRIO_INSTRUCTIONS if manipulation_proprio_enabled else "") + (
                           GIMBAL_SENSOR_FRAMING_INSTRUCTIONS if gimbal_sensor_framing_preferred else "")})
        for key, value in config.items():
            # JSON scalar/string representations are valid TOML values. Empty tables
            # intentionally contain no MCP/plugin configuration.
            command.extend(["-c", f"{key}={json.dumps(value, ensure_ascii=False)}"])
        if self.transport == "chatgpt_https":
            # This opt-in uses the existing ChatGPT sign-in. Never copy auth or
            # accept caller-supplied endpoints, tokens, or authentication headers.
            command.extend([
                "-c", 'model_provider="openai-https"',
                "-c", 'model_providers.openai-https={name="OpenAI",'
                'base_url="https://chatgpt.com/backend-api/codex",wire_api="responses",'
                'requires_openai_auth=true,supports_websockets=false}',
            ])
        for feature in _DISABLED_FEATURES:
            command.extend(["--disable", feature])
        command.extend(["--enable", "skip_host_skill_discovery"])
        for path in images:
            command.extend(["--image", str(path)])
        command.append("-")
        return command

    def decide(self, observation: dict[str, Any], image_paths: list[Path],
               history: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(observation.get("observation_id"), str) or not observation["observation_id"]:
            raise PlannerError("A current nonempty observation_id is required")
        images = [Path(path).resolve() for path in image_paths]
        if not images or any(not path.is_file() for path in images):
            raise PlannerError("At least one existing current RGB image is required")
        if observation.get("capabilities", {}).get("language_context_enabled") is True:
            cameras = observation.get("cameras", {})
            current_rgb = {Path(camera["path"]).resolve() for role, camera in cameras.items()
                           if role in {"head", "left_wrist", "right_wrist"}
                           and camera.get("source") == "robot_sensor" and camera.get("color_space") == "RGB"
                           and camera.get("step") == observation.get("step") and camera.get("path")}
            if any(path not in current_rgb for path in images):
                raise PlannerError("Language context requires attachments from the current robot RGB observation")
        run_dir = self.output_dir / (time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10])
        run_dir.mkdir(parents=True, exist_ok=False)
        self.last_run_dir = run_dir
        start = time.monotonic()
        metadata: dict[str, Any] = {"status": "started", "observation_id": observation["observation_id"],
                                   "timeout_s": self.timeout_s, "retry_count": 0,
                                   "retry_count_scope": "harness_cli_restarts"}
        if self.transport != "auto":
            metadata["planner_transport"] = self.transport
        prefer_gimbal = gimbal_sensor_framing_preferred(observation)
        if prefer_gimbal:
            metadata["gimbal_sensor_framing_preferred"] = True
        _write_json(run_dir / "metadata.json", metadata)
        try:
            prompt = build_prompt(observation, images, history)
            (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            _write_json(run_dir / "schema.json", DECISION_SCHEMA)
            # A fresh directory outside the checkout avoids project instructions and
            # gives no implicit scene/source context. No credentials are placed here.
            with tempfile.TemporaryDirectory(prefix="behavior-planner-") as temp:
                command = self._command(run_dir, Path(temp), images,
                                        language_context_enabled=observation.get("capabilities", {}).get(
                                            "language_context_enabled") is True,
                                        manipulation_proprio_enabled=observation.get("capabilities", {}).get(
                                            "manipulation_proprio_enabled") is True,
                                        gimbal_sensor_framing_preferred=prefer_gimbal)
                _write_json(run_dir / "command.json", command)
                with (run_dir / "stdout.jsonl").open("w", encoding="utf-8") as stdout, \
                     (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout,
                                               stderr=stderr, text=True, start_new_session=True,
                                               cwd=temp)
                    try:
                        process.communicate(prompt, timeout=self.timeout_s)
                    except subprocess.TimeoutExpired as exc:
                        # Kill the CLI's entire child group, not only its Node wrapper.
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=3)
                        metadata["returncode"] = process.returncode
                        raise PlannerTimeout("Codex visual decision timed out; no action was returned or retried", run_dir) from exc
                    except BaseException:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait(timeout=3)
                        raise
                metadata["returncode"] = process.returncode
                if process.returncode != 0:
                    raise PlannerError(f"Codex exited with status {process.returncode}; see stderr.log", run_dir)
            metadata.update(_assert_no_tool_events((run_dir / "stdout.jsonl").read_text(encoding="utf-8")))
            raw_path = run_dir / "decision.raw.json"
            if not raw_path.is_file():
                raise PlannerError("Codex did not write a decision", run_dir)
            decision = json.loads(raw_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
            validate_planner_decision(decision, observation)
            _write_json(run_dir / "decision.json", decision)
            metadata["status"] = "ok"
            return decision
        except BaseException as exc:
            metadata["status"] = "timeout" if isinstance(exc, PlannerTimeout) else "error"
            metadata["error_type"] = type(exc).__name__
            # Exception type and our own validation diagnostics contain no auth data.
            metadata["error"] = str(exc)
            if isinstance(exc, (PlannerError, KeyboardInterrupt, SystemExit)):
                raise
            raise PlannerError(f"Invalid/unavailable Codex decision: {exc}", run_dir) from exc
        finally:
            metadata["elapsed_s"] = round(time.monotonic() - start, 6)
            _write_json(run_dir / "metadata.json", metadata)
