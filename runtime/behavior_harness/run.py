"""Run a visual GPT control loop or a matched VLA-only baseline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from .client import HarnessClient
from .planner import CodexPlanner, validate_planner_transport


def validate_planner_timeout(value):
    """Bound one CLI invocation's total wait; never add an outer retry."""
    if type(value) not in (int, float) or not 0 < value <= 600 or not math.isfinite(value):
        raise ValueError("planner_timeout_s must be a finite number greater than 0 and at most 600 seconds")
    return value


def decision(action, obs, **kwargs):
    value = {"action": action, "observation_id": obs["observation_id"],
             "reasoning": "host-controlled VLA-only baseline", "target_visible": False,
             "manipulation_ready": False, "vx_mps": 0.0, "vy_mps": 0.0, "yaw_rps": 0.0,
             "head_yaw_rad": 0.0, "head_pitch_rad": 0.0, "steps": 0,
             "trunk_qpos": [], "stop_reason": ""}
    value.update(kwargs)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--mode", choices=["assisted", "baseline"], default="assisted")
    parser.add_argument("--model", default=None, help="Default: user's configured Codex model")
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--planner-timeout", type=float, default=None,
                        help="Total wait for one planner CLI call, up to 600 seconds; config or default 240")
    parser.add_argument("--planner-transport", choices=["auto", "chatgpt_https"], default=None,
                        help="Optional ChatGPT-signed-in HTTPS transport; default uses Codex transport")
    parser.add_argument("--vla-chunk", type=int, default=200)
    args = parser.parse_args()
    if args.max_turns < 1 or args.vla_chunk < 1:
        parser.error('max-turns and vla-chunk must be positive')
    ready = Path(args.ready).resolve()
    metadata = json.loads(ready.read_text())
    if metadata.get("run_mode") != args.mode:
        raise ValueError("Runner mode must match the service run_mode; use a fresh same-seed episode")
    output = ready.parent
    config_path = output / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    agent_initial_only = config.get("agent_initial_only", False)
    if type(agent_initial_only) is not bool:
        raise ValueError("agent_initial_only must be a boolean")
    vla_locked = False
    transport = validate_planner_transport(config.get("planner_transport", "auto"), mode=args.mode)
    if args.planner_transport is not None:
        if "planner_transport" in config and args.planner_transport != transport:
            raise ValueError("Runner planner transport must match the saved episode configuration")
        transport = validate_planner_transport(args.planner_transport, mode=args.mode)
    planner_timeout = validate_planner_timeout(config.get("planner_timeout_s", 240))
    if args.planner_timeout is not None:
        if "planner_timeout_s" in config and args.planner_timeout != planner_timeout:
            raise ValueError("Runner planner timeout must match the saved episode configuration")
        planner_timeout = validate_planner_timeout(args.planner_timeout)
    client = HarnessClient(ready, timeout_s=1800)
    planner_options = {"transport": transport} if transport != "auto" else {}
    planner = (CodexPlanner(output / "planner", model=args.model, timeout_s=planner_timeout,
                            **planner_options) if args.mode == "assisted" else None)
    history = []
    started = time.time()
    summary = {"mode": args.mode, "status": "running", "started_unix": started,
               "planner_timeout_s": planner_timeout, "agent_initial_only": agent_initial_only}
    if transport != "auto":
        summary["planner_transport"] = transport
    try:
        obs = client.call("observe")["observation"]
        for turn in range(args.max_turns):
            if obs["terminal"]["done"] or obs["remaining_steps"] <= 0:
                summary["result"] = client.call("close", {"reason": obs["terminal"].get("termination_reason") or "budget_exhausted"})
                summary["status"] = "completed"
                break
            if args.mode == "baseline" or vla_locked:
                plan = decision("run_vla", obs, steps=min(args.vla_chunk, obs["remaining_steps"], 300))
                if vla_locked:
                    plan["reasoning"] = "Initial assistance completed; VLA retains control until the episode ends."
            else:
                images = [Path(obs["cameras"][name]["path"]) for name in ("head", "left_wrist", "right_wrist")]
                plan = planner.decide(obs, images, history)
            receipt = client.call("execute", plan)
            if agent_initial_only and plan["action"] == "handoff":
                if receipt.get("observation", {}).get("phase") != "vla":
                    raise RuntimeError("Initial handoff did not enter VLA phase")
                vla_locked = True
                summary["initial_handoff_step"] = receipt["observation"]["step"]
            execution = receipt.get("execution", {})
            row = {"turn": turn, "observation_id": obs["observation_id"], "step": obs["step"],
                   "decision": plan, "execution": execution}
            history.append(row)
            with (output / "agent_decisions.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"turn": turn, "action": plan["action"], "step": receipt.get("observation", {}).get("step"),
                              "reasoning": plan["reasoning"]}, ensure_ascii=False), flush=True)
            if plan["action"] == "stop":
                summary.update(status="agent_stopped", result=receipt)
                break
            obs = receipt["observation"]
            if obs["terminal"]["done"] or obs["remaining_steps"] <= 0:
                summary.update(status="completed", result=client.call("close", {
                    "reason": obs["terminal"].get("termination_reason") or "budget_exhausted"}))
                break
        else:
            summary.update(status="turn_budget_exhausted", result=client.call("close", {"reason": "agent_turn_budget"}))
    except BaseException as exc:
        summary.update(status="error", error=f"{type(exc).__name__}: {exc}", last_request_id=client.last_request_id)
        # Stop is cancellation, never a replay/reset of an uncertain motion command.
        try:
            summary["stop_receipt"] = client.stop()
        except Exception as stop_error:
            summary["stop_error"] = str(stop_error)
        raise
    finally:
        summary["elapsed_seconds"] = time.time() - started
        (output / "runner_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
