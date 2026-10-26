"""Offline orchestration checks; never imports a robot backend."""
import pytest

from examples.g1.mickey_workflow import CheckpointRequired, MickeyWorkflow


class FakeContext:
    def __init__(self):
        self.now = 100.
        self.calls = []
        self.jobs = {}
        self.active = None
        self.initialized = True
        self.session = "session"
        self.frames = 0
        self.failure = None
        self.pending = False
        self.reuse_source = False

    def value(self, tool, **args):
        self.calls.append((tool, args))
        if tool == "g1_vla_status":
            return dict(session_id=self.session, phase="ready", fault=None,
                        vla=dict(initialized=self.initialized, active_job=self.active),
                        locomotion=dict(active_job=None))
        if tool == "g1_observe":
            self.frames += 1
            return dict(frame_id=str(self.frames), received_monotonic=self.now,
                        source_frame_id="same" if self.reuse_source else str(self.frames))
        if tool == "g1_vla_result" or (tool == "g1_move" and args["action"] == "result"):
            job = args["job_id"]
            if self.pending and self.jobs[job].get("action") == "run":
                return dict(job_id=job, status="pending_or_unknown")
            return self.jobs[job]
        action = args["action"]
        job = f"job{len(self.jobs)}"
        if action == "pause":
            previous = self.active["id"]
            self.jobs[previous].update(completed=False, reason="agent_pause")
            self.pending = False
            self.active = None
        elif action == "run" and tool == "g1_vla" and self.pending:
            self.active = dict(id=job)
        if tool == "g1_move":
            self.initialized = False
        if action == "prepare":
            self.initialized = True
        self.jobs[job] = dict(job_id=job, session_id=self.session, action=action,
                              completed=not self.failure, reason=self.failure or "complete",
                              settled=True, task_success_verified=False)
        return dict(job_id=job)

    def sleep(self, seconds):
        self.now += seconds


def approved(stage, obs):
    return dict(frame_id=obs["frame_id"], evidence="Test fixture image assessment",
                path_clear=True, hands_empty=True, arm_clear=True,
                target_reachable=True, approach_clear=True, held=True)


def workflow(ctx, review=approved):
    return MickeyWorkflow(ctx, review, clock=lambda: ctx.now, sleep=ctx.sleep)


def mutations(ctx):
    return [(t, a) for t, a in ctx.calls
            if t in ("g1_move", "g1_vla") and a.get("action") not in ("result", "status")]


def test_move_invalidates_preparation_and_cannot_skip_reprepare():
    ctx = FakeContext()
    flow = workflow(ctx)
    assert flow.move("right")["result"]["settled"]
    with pytest.raises(CheckpointRequired, match="Prepare required"):
        flow.grasp()
    flow.prepare()
    result = flow.grasp()
    assert result["visual_grasp_verified"]
    assert result["task_success_verified"] is False
    assert result["verification_frames"][0] != result["verification_frames"][1]
    assert mutations(ctx)[-1][1]["steps"] == 750


@pytest.mark.parametrize("bad", ["hands_empty", "arm_clear", "frame_id", "stale"])
def test_prepare_never_opens_hands_without_current_review(bad):
    ctx = FakeContext()

    def review(stage, obs):
        result = approved(stage, obs)
        if bad == "stale":
            ctx.now += 31
        else:
            result[bad] = False
        return result

    with pytest.raises(CheckpointRequired):
        workflow(ctx, review).prepare()
    assert not mutations(ctx)


def test_failed_move_is_not_retried():
    ctx = FakeContext()
    ctx.failure = "movement_settle_timeout"
    with pytest.raises(CheckpointRequired, match="movement_settle_timeout"):
        workflow(ctx).move("right")
    assert len(mutations(ctx)) == 1


def test_owner_change_blocks_next_action():
    ctx = FakeContext()
    flow = workflow(ctx)
    flow.move("right")
    ctx.session = "replacement"
    with pytest.raises(CheckpointRequired, match="session changed"):
        flow.prepare()
    assert len(mutations(ctx)) == 1


def test_early_visual_hold_pauses_own_job_and_checks_two_frames():
    ctx = FakeContext()
    ctx.pending = True
    result = workflow(ctx).grasp()
    assert [a["action"] for _, a in mutations(ctx)] == ["run", "pause"]
    assert result["result"]["completed"] is False
    assert result["result"]["reason"] == "agent_pause"
    assert result["visual_grasp_verified"] is True
    assert result["task_success_verified"] is False


def test_uncertain_hold_does_not_become_success():
    ctx = FakeContext()
    ctx.pending = True

    def review(stage, obs):
        return approved(stage, obs) | {"held": False, "continue": False}

    with pytest.raises(CheckpointRequired, match="held"):
        workflow(ctx, review).grasp()
    assert [a["action"] for _, a in mutations(ctx)] == ["run", "pause"]


def test_same_camera_frame_is_not_two_confirmations():
    ctx = FakeContext()
    ctx.reuse_source = True
    with pytest.raises(CheckpointRequired, match="distinct camera"):
        workflow(ctx).grasp()


def test_thermal_interruption_is_not_reprepared_or_retried():
    ctx = FakeContext()
    ctx.failure = "ankle_temperature_priority"
    with pytest.raises(CheckpointRequired, match="ankle_temperature_priority"):
        workflow(ctx).grasp()
    assert [a["action"] for _, a in mutations(ctx)] == ["run"]


@pytest.mark.parametrize("direction,duration,rate", [
    ("right", 4, 15), ("rotate_left", 2, 20), ("rotate_left", 1, float("nan")),
])
def test_invalid_motion_never_reaches_service(direction, duration, rate):
    ctx = FakeContext()
    with pytest.raises(ValueError):
        workflow(ctx).move(direction, duration, rate)
    assert ctx.calls == []
