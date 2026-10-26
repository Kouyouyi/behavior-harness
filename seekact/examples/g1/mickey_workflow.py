"""Compile the 2026-10-20 task's tool orchestration, keeping visual decisions explicit.

Uses only the supplied RAX Context. Does not start hardware or replay a route.
review(stage, observation) must inspect the supplied live image and return
{frame_id, evidence, ... stage-specific booleans ...}. No default approval.
"""
import math
import time


class CheckpointRequired(RuntimeError):
    """The caller must inspect the scene before doing any more work."""


class MickeyWorkflow:
    def __init__(self, ctx, review, *, clock=time.monotonic, sleep=time.sleep):
        self.ctx, self.review = ctx, review
        self.clock, self.sleep = clock, sleep
        self.session_id = None
        self.records = []

    def _owner(self):
        state = self.ctx.value("g1_vla_status")
        session = state["session_id"]
        if self.session_id is not None and session != self.session_id:
            raise CheckpointRequired("Owner session changed; create a new workflow")
        self.session_id = session
        return state

    def _ready(self):
        state = self._owner()
        if (state.get("phase") != "ready" or state.get("fault")
                or state["vla"].get("active_job")
                or (state.get("locomotion") or {}).get("active_job")):
            raise CheckpointRequired("Owner unavailable or busy; inspect before continuing")
        return state

    def _look(self, stage, required=()):
        observation = self.ctx.value("g1_observe")
        decision = self.review(stage, observation)
        if not isinstance(decision, dict):
            raise CheckpointRequired(f"{stage}: missing visual review")
        if decision.get("frame_id") != observation["frame_id"]:
            raise CheckpointRequired(f"{stage}: review does not match current frame")
        evidence = decision.get("evidence")
        if not isinstance(evidence, str) or not 1 <= len(evidence.strip()) <= 1800:
            raise CheckpointRequired(f"{stage}: describe what the image shows")
        received = observation.get("received_monotonic")
        if (type(received) not in (float, int) or not math.isfinite(received)
                or not 0 <= self.clock() - received <= 30):
            raise CheckpointRequired(f"{stage}: observation expired; observe again")
        self.records.append(dict(stage=stage, observation=observation, decision=decision))
        if any(decision.get(key) is not True for key in required):
            raise CheckpointRequired(f"{stage}: required review: {', '.join(required)}")
        return observation, decision

    def _result(self, kind, job):
        if kind == "move":
            return self.ctx.value("g1_move", action="result", job_id=job)
        return self.ctx.value("g1_vla_result", job_id=job)

    def _wait(self, kind, job, timeout):
        deadline = self.clock() + timeout
        while self.clock() < deadline:
            self._owner()
            result = self._result(kind, job)
            if result.get("job_id") != job:
                raise CheckpointRequired("Result job identity mismatch")
            if "session_id" in result and result["session_id"] != self.session_id:
                raise CheckpointRequired("Result owner identity mismatch")
            if "completed" in result:
                self.records.append(dict(stage="result", result=result))
                return result
            self.sleep(.1)
        # Do not retry a submission with an uncertain outcome or shut down native.
        raise CheckpointRequired(f"{kind} job {job} timed out; inspect active job")

    @staticmethod
    def _complete(result, *, move=False):
        if result.get("completed") is not True or (move and result.get("settled") is not True):
            raise CheckpointRequired(f"Action did not complete: {result.get('reason')}")

    def move(self, direction, duration_s=2., yaw_rate_deg_s=15.):
        """One reviewed segment, automatic polling, then a fresh observation.

        Actual translation remains unknown. Never multiply speed by duration to
        claim measured distance. Choose the next segment from the new image.
        """
        directions = {"forward", "backward", "left", "right", "forward_left",
                      "forward_right", "backward_left", "backward_right",
                      "rotate_left", "rotate_right", "step_in_place"}
        turning = direction in {"rotate_left", "rotate_right"}
        if direction not in directions:
            raise ValueError("Unsupported direction")
        if (type(duration_s) not in (int, float) or not math.isfinite(duration_s)
                or not .1 <= duration_s <= (2 if turning else 3)):
            raise ValueError("Invalid segment duration")
        if turning and (type(yaw_rate_deg_s) not in (int, float)
                        or not math.isfinite(yaw_rate_deg_s)
                        or not 1 <= yaw_rate_deg_s <= 30
                        or duration_s * yaw_rate_deg_s > 30):
            raise ValueError("Invalid turn reference")
        self._ready()
        obs, decision = self._look("before_move", ("path_clear",))
        self._ready()
        args = dict(action="run", direction=direction, duration_s=duration_s,
                    speed=.3, frame_id=obs["frame_id"], evidence=decision["evidence"])
        if turning:
            args["yaw_rate_deg_s"] = yaw_rate_deg_s
        job = self.ctx.value("g1_move", **args)["job_id"]
        result = self._wait("move", job, duration_s + 18)
        self._complete(result, move=True)
        return dict(result=result, observation=self.ctx.value("g1_observe"),
                    task_success_verified=False)

    def prepare(self):
        """Prepare opens BOTH hands; never infer that an uninitialized VLA is empty."""
        self._ready()
        self._look("before_prepare", ("hands_empty", "arm_clear"))
        self._ready()
        job = self.ctx.value("g1_vla", action="prepare")["job_id"]
        result = self._wait("vla", job, 20)
        self._complete(result)
        return dict(result=result, observation=self.ctx.value("g1_observe"))

    def grasp(self):
        """Run at most 15 s, pause on a visual hold/uncertainty, verify two frames.

        The review callback is synchronous. Slow image review can miss early-stop
        opportunities; the owner's 750-step bound remains in force.
        """
        if not self._ready()["vla"].get("initialized"):
            raise CheckpointRequired("Prepare required after movement or thermal interruption")
        _, decision = self._look("before_grasp", ("target_reachable", "approach_clear"))
        if not self._ready()["vla"].get("initialized"):
            raise CheckpointRequired("VLA preparation invalidated during review")
        job = self.ctx.value("g1_vla", action="run", task="mickey", steps=750,
                             evidence=decision["evidence"])["job_id"]
        deadline, next_look = self.clock() + 35, self.clock() + 2
        paused = False
        try:
            while self.clock() < deadline:
                self._owner()
                result = self._result("vla", job)
                if result.get("job_id") != job:
                    raise CheckpointRequired("Result job identity mismatch")
                if result.get("session_id", self.session_id) != self.session_id:
                    raise CheckpointRequired("Result owner identity mismatch")
                if "completed" in result:
                    break
                if self.clock() >= next_look:
                    _, check = self._look("during_grasp")
                    if check.get("held") is True or check.get("continue") is not True:
                        self._pause_own_job(job)
                        paused = True
                        result = self._wait("vla", job, 5)
                        break
                    next_look = self.clock() + 2
                self.sleep(.1)
            else:
                raise CheckpointRequired("Grasp result timeout")
        except Exception as error:
            # Best effort only for our own active VLA job, never another caller's.
            try:
                self._pause_own_job(job)
            except Exception as pause_error:
                raise CheckpointRequired(
                    f"{error}; pause could not be confirmed: {pause_error}") from error
            raise
        self.records.append(dict(stage="grasp_result", result=result))
        if result.get("completed") is not True and not (
                paused and result.get("reason") == "agent_pause"):
            raise CheckpointRequired(f"Grasp interrupted: {result.get('reason')}")
        self._ready()
        first, _ = self._look("verify_hold", ("held",))
        self.sleep(.2)
        self._ready()
        second, _ = self._look("verify_hold_again", ("held",))
        if (not first.get("source_frame_id") or
                first["source_frame_id"] == second.get("source_frame_id")):
            raise CheckpointRequired("Hold verification requires two distinct camera frames")
        return dict(result=result, visual_grasp_verified=True,
                    verification_frames=[first["frame_id"], second["frame_id"]],
                    task_success_verified=result.get("task_success_verified", False))

    def _pause_own_job(self, job):
        state = self._owner()
        active = state["vla"].get("active_job") or {}
        if active.get("id") == job:
            pause = self.ctx.value("g1_vla", action="pause")["job_id"]
            result = self._wait("vla", pause, 5)
            self._complete(result)
