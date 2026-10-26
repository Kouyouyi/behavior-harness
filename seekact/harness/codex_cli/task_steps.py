"""Bounded code execution between visual checkpoints through the existing MCP.

No robot SDK, publisher or native lifecycle operations. The bridge owns one
instance per UI turn; durable request IDs prevent motion replay after reconnect.
"""
import json
import math
from pathlib import Path
import time

from services.g1_sonic.vla import atomic
from services.g1_sonic.locomotion import validate as validate_move

NAME = "g1_task_step"
SCHEMA = dict(name=NAME, description=(
    "Fast Mickey workflow: observe returns live image+owner; move/prepare submit once, "
    "poll in Python and return the next image; grasp submits exactly 750 steps and "
    "returns an early visual checkpoint after 2 seconds; check waits briefly and "
    "returns a fresh image; pause stops only this workflow's active upper-level job. "
    "confirm_hold records visual success from two distinct fresh source frames after "
    "the action ended. Always inspect images. 600-second task budget, no native stop. "
    "For observe/check/pause send only action; optional request_id/frame_id/evidence "
    "are accepted as descriptive metadata and never authorize extra motion. "
    "Mutations require unique request_id, frame_id and truthful evidence; move requires "
    "path_clear=true, prepare hands_empty=true and arm_clear=true, grasp "
    "target_reachable=true and approach_clear=true. Observe first. "
    "New operator messages interrupt waiting and pause the owned job."),
    inputSchema=dict(type="object", additionalProperties=False, required=["action"],
        properties=dict(
            action=dict(type="string", enum=["observe", "move", "prepare", "grasp",
                                            "check", "pause", "confirm_hold"]),
            request_id=dict(type="string"), frame_id=dict(type="string"),
            evidence=dict(type="string"), direction=dict(type="string"),
            duration_s=dict(type="number"), yaw_rate_deg_s=dict(type="number"),
            path_clear=dict(type="boolean"), hands_empty=dict(type="boolean"),
            arm_clear=dict(type="boolean"), target_reachable=dict(type="boolean"),
            approach_clear=dict(type="boolean"),
            verification_frames=dict(type="array", items=dict(type="string"),
                                     minItems=2, maxItems=2))))


class TaskSteps:
    def __init__(self, connection, run, session, batch, guard, *, clock=time.monotonic,
                 sleep=time.sleep):
        self.ctx, self.session, self.guard = connection, session, guard
        self.clock, self.sleep = clock, sleep
        self.path = Path(run)/"ui-agent-turns"/batch/"compiled-task.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["session"] != session:
                raise ValueError("Compiled workflow owner changed")
        else:
            status_path = Path(run)/"ui-agent-status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            self.state = dict(session=session, started=status.get("started_monotonic") or clock(),
                              requests={}, frames={}, used_frames=[], active=None, history=[])
            self.save()

    def save(self):
        atomic(self.path, self.state)

    def value(self, tool, **args):
        return self.ctx.value(tool, **args)

    def owner(self):
        state = self.value("g1_vla_status")
        if state["session_id"] != self.session:
            raise ValueError("Owner changed; old task cannot resume")
        return state

    def remaining(self):
        elapsed = self.clock()-self.state["started"]
        if elapsed < 0:
            raise ValueError("Clock changed; old task cannot resume")
        return max(0., 600-elapsed)

    def result(self, active):
        if active["kind"] == "move":
            return self.value("g1_move", action="result", job_id=active["job_id"])
        return self.value("g1_vla_result", job_id=active["job_id"])

    def update(self):
        active = self.state["active"]
        if not active:
            return None
        result = self.result(active)
        if result.get("job_id") != active["job_id"]:
            raise ValueError("Result job mismatch")
        if result.get("session_id", self.session) != self.session:
            raise ValueError("Result session mismatch")
        if "completed" in result:
            self.state["active"] = None
            self.state["last_result"] = result
            self.state["history"].append(result)
            self.save()
        return result

    def pause_owned(self):
        active = self.state["active"]
        if not active:
            return
        owner = self.owner()
        field = "locomotion" if active["kind"] == "move" else "vla"
        current = (owner.get(field) or {}).get("active_job") or {}
        # Never pause another caller's job. A submitted pending request can still
        # be ours; a pause/stop has priority over that pending request in the owner.
        if current and current.get("id") != active["job_id"]:
            raise ValueError("Another job is active; refusing to pause it")
        if active["kind"] == "move":
            reply = self.value("g1_move", action="stop")
        else:
            reply = self.value("g1_vla", action="pause")
        self.state["last_pause_request"] = reply
        self.save()
        deadline = self.clock()+4
        while self.clock() < deadline and self.state["active"]:
            self.update()
            if self.state["active"]:
                self.sleep(.1)

    def wait(self, seconds):
        deadline = self.clock()+seconds
        while self.state["active"] and self.clock() < deadline:
            self.guard()
            pending = self.value("g1_operator_messages", action="read").get("pending", [])
            if pending or self.remaining() <= 0:
                self.pause_owned()
                return "new_operator_message" if pending else "task_budget_exhausted"
            result = self.update()
            if result and "completed" in result:
                if not result["completed"] or (
                        result.get("command") and not result.get("settled")):
                    return "action_interrupted_review_required"
                return "action_complete_review_image"
            self.sleep(.2)
        return "action_running_review_image" if self.state["active"] else "ready"

    def output(self, reason, **extra):
        report = dict(reason=reason, remaining_seconds=self.remaining(),
                      target_seconds=[180, 300], max_seconds=600,
                      active=self.state["active"], last_result=self.state.get("last_result"),
                      visual_grasp_verified=self.state.get("visual_grasp_verified", False),
                      **extra)
        images = []
        try:
            observed = self.ctx.call("g1_observe")
            obs = json.loads(next(b["text"] for b in observed["content"] if b["type"] == "text"))
            if observed.get("isError"):
                raise ValueError(obs)
            self.state["frames"][obs["frame_id"]] = obs
            # A bounded history is enough for two-frame verification.
            self.state["frames"] = dict(list(self.state["frames"].items())[-12:])
            self.save()
            report["observation"] = obs
            images = [b for b in observed["content"] if b["type"] == "image"]
        except Exception as exc:
            report["observation_error"] = str(exc)
            # An unobserved running grasp must not wait for another model turn.
            try:
                self.pause_owned()
                report["active"] = self.state["active"]
            except Exception as pause_error:
                report["pause_error"] = str(pause_error)
        try:
            report["owner"] = self.owner()
            report["operator_messages"] = self.value("g1_operator_messages", action="read")
        except Exception as exc:
            report["state_error"] = str(exc)
        return dict(content=[dict(type="text", text=json.dumps(report, ensure_ascii=False)), *images])

    def frame(self, ident):
        obs = self.state["frames"].get(ident)
        if not obs or not 0 <= self.clock()-obs["received_monotonic"] <= 30:
            raise ValueError("Use a fresh image from g1_task_step before deciding")
        return obs

    def call(self, args):
        self.guard()
        action = args.get("action")
        if action not in SCHEMA["inputSchema"]["properties"]["action"]["enum"]:
            raise ValueError("Unknown compiled action")
        allowed = {"action"}
        if action in ("observe", "check", "pause"):
            # Some clients carry the preceding checkpoint's metadata forward.
            # These fields are inert for read/pause: never dispatch another run.
            allowed |= {"request_id", "frame_id", "evidence"}
        if action in ("move", "prepare", "grasp"):
            allowed |= {"request_id", "frame_id", "evidence"}
            allowed |= {"move": {"direction", "duration_s", "yaw_rate_deg_s", "path_clear"},
                        "prepare": {"hands_empty", "arm_clear"},
                        "grasp": {"target_reachable", "approach_clear"}}[action]
        if action == "confirm_hold":
            allowed |= {"verification_frames", "evidence", "request_id", "frame_id"}
        if set(args)-allowed:
            raise ValueError("Unexpected compiled action arguments")
        self.owner()
        self.update()
        if action in ("observe", "check", "pause"):
            if action == "pause":
                self.pause_owned()
            reason = self.wait(2 if action == "check" else 0)
            return self.output(reason)
        if self.remaining() <= 0:
            self.pause_owned()
            return self.output("task_budget_exhausted")
        if self.state["active"]:
            raise ValueError("Previous job active; check/pause and inspect before continuing")
        owner = self.owner()
        if (owner["phase"] != "ready" or owner.get("fault")
                or owner["vla"].get("active_job")
                or (owner.get("locomotion") or {}).get("active_job")):
            raise ValueError("Owner unavailable or busy")
        if self.value("g1_operator_messages", action="read").get("pending"):
            return self.output("acknowledge_new_operator_message_first")
        evidence = args.get("evidence")
        if not isinstance(evidence, str) or not 1 <= len(evidence.strip()) <= 1800:
            raise ValueError("Describe actual visual evidence")
        if action == "confirm_hold":
            ids = args.get("verification_frames")
            if not isinstance(ids, list) or len(ids) != 2:
                raise ValueError("Two visual confirmation frames required")
            first, second = [self.frame(i) for i in ids]
            if not first.get("source_frame_id") or first["source_frame_id"] == second.get("source_frame_id"):
                raise ValueError("Two distinct camera source frames required")
            self.state.update(visual_grasp_verified=True, verification_frames=ids,
                              verification_evidence=evidence)
            self.save()
            return self.output("visual_hold_confirmed")
        ident = args.get("request_id")
        if not isinstance(ident, str) or not 1 <= len(ident) <= 80:
            raise ValueError("Unique request_id required")
        previous = self.state["requests"].get(ident)
        if previous:
            if previous["arguments"] != args:
                raise ValueError("Request ID reused with different arguments")
            return self.output("request_already_recorded_no_replay", request=previous)
        if any(r["phase"] == "submitting" for r in self.state["requests"].values()):
            raise ValueError("Earlier submission outcome unknown; inspect before a new task")
        self.frame(args.get("frame_id"))
        if args["frame_id"] in self.state["used_frames"]:
            raise ValueError("New motion requires a new reviewed frame")
        required = {"move": ("path_clear",), "prepare": ("hands_empty", "arm_clear"),
                    "grasp": ("target_reachable", "approach_clear")}[action]
        if any(args.get(k) is not True for k in required):
            raise ValueError("Required scene review: "+", ".join(required))
        if action == "move":
            move = {k: args[k] for k in ("direction", "duration_s", "yaw_rate_deg_s",
                                        "frame_id", "evidence") if k in args}
            command = validate_move("run", **move)
            tool, params, wait = "g1_move", dict(action="run", **command), command["duration_s"]+15
        elif action == "prepare":
            tool, params, wait = "g1_vla", dict(action="prepare"), 18
        else:
            if not owner["vla"].get("initialized"):
                raise ValueError("Prepare required after movement/thermal interruption")
            tool, params, wait = "g1_vla", dict(action="run", task="mickey", steps=750,
                                              evidence=evidence), 2
        if self.remaining() < (30 if action == "grasp" else wait+2):
            return self.output("insufficient_task_budget_no_new_action")
        self.state["requests"][ident] = dict(arguments=args, phase="submitting")
        self.state["used_frames"].append(args["frame_id"])
        self.state["visual_grasp_verified"] = False
        self.save()  # Durable before dispatch: an ambiguous reply is never replayed.
        reply = self.value(tool, **params)
        active = dict(kind="move" if action == "move" else "vla", job_id=reply["job_id"])
        self.state["active"] = active
        self.state["requests"][ident].update(phase="submitted", job_id=reply["job_id"])
        self.save()
        try:
            reason = self.wait(wait)
            if action != "grasp" and self.state["active"]:
                self.pause_owned()
                reason = "step_timeout_review_required"
            return self.output(reason)
        except Exception:
            self.pause_owned()
            raise
