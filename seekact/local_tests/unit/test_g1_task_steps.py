"""Compiled tool orchestration against a fake MCP, with no network or hardware."""
import json
import io
import time
import pytest

from harness.codex_cli.task_steps import TaskSteps
from harness.codex_cli import ui_agent
from harness.codex_cli.ui_bridge import serve
from services.g1_sonic.vla import atomic


class Context:
    def __init__(self):
        self.now = 100.
        self.calls = []
        self.frames = 0
        self.jobs = {}
        self.active = None
        self.initialized = True
        self.pending = []
        self.session = "session"
        self.run_pending = False
        self.failure = None
        self.observe_failure = False
        self.message_at = None

    def sleep(self, seconds):
        self.now += seconds
        if self.message_at and self.now >= self.message_at:
            self.pending = [dict(id="new", text="暂停")]

    def call(self, tool, **args):
        if self.observe_failure:
            raise RuntimeError("camera unavailable")
        assert tool == "g1_observe"
        self.frames += 1
        obs = dict(frame_id=f"{self.frames:032x}", source_frame_id=f"source{self.frames}",
                   received_monotonic=self.now)
        return dict(content=[dict(type="text", text=json.dumps(obs)),
                             dict(type="image", mimeType="image/png", data="fixture")])

    def value(self, tool, **args):
        self.calls.append((tool, args))
        if tool == "g1_vla_status":
            return dict(session_id=self.session, phase="ready", fault=None,
                        vla=dict(initialized=self.initialized,
                                 active_job=self.active if self.active and self.active["kind"] == "vla" else None),
                        locomotion=dict(active_job=self.active if self.active and self.active["kind"] == "move" else None))
        if tool == "g1_operator_messages":
            return dict(pending=self.pending)
        if tool == "g1_vla_result" or (tool == "g1_move" and args["action"] == "result"):
            result = self.jobs[args["job_id"]]
            if "completed" in result:
                self.active = None
            return result
        action = args["action"]
        if action in ("pause", "stop"):
            old = self.active["id"]
            self.jobs[old].update(completed=False, reason="agent_pause")
        job = f"job{len(self.jobs)}"
        kind = "move" if tool == "g1_move" else "vla"
        self.active = dict(id=job, kind=kind)
        result = dict(job_id=job, session_id=self.session, task_success_verified=False)
        if action != "run" or tool == "g1_move" or not self.run_pending:
            result.update(completed=not self.failure, reason=self.failure or "complete")
        if tool == "g1_move" and action == "run":
            self.initialized = False
            result.update(command=args, settled=True)
        if action == "prepare":
            self.initialized = True
        self.jobs[job] = result
        return dict(job_id=job)


def make(tmp_path, ctx=None):
    ctx = ctx or Context()
    steps = TaskSteps(ctx, tmp_path, "session", "batch", lambda: None,
                      clock=lambda: ctx.now, sleep=ctx.sleep)
    return ctx, steps


def report(value):
    return json.loads(value["content"][0]["text"])


def observe(steps):
    return report(steps.call(dict(action="observe")))["observation"]["frame_id"]


def args(steps, action, **kw):
    return dict(action=action, request_id=action, frame_id=observe(steps),
                evidence="Inspected current fixture scene", **kw)


def mutations(ctx):
    return [(t, a) for t, a in ctx.calls
            if t in ("g1_move", "g1_vla") and a["action"] not in ("result", "status")]


def test_step_waits_and_returns_image_without_agent_polling(tmp_path):
    ctx, steps = make(tmp_path)
    reply = steps.call(args(steps, "move", direction="right", duration_s=2, path_clear=True))
    assert report(reply)["reason"] == "action_complete_review_image"
    assert reply["content"][1]["type"] == "image"
    assert len(mutations(ctx)) == 1
    with pytest.raises(ValueError, match="Prepare required"):
        steps.call(args(steps, "grasp", target_reachable=True, approach_clear=True))


def test_request_dedup_survives_bridge_restart(tmp_path):
    ctx, steps = make(tmp_path)
    action = args(steps, "prepare", hands_empty=True, arm_clear=True)
    steps.call(action)
    _, resumed = make(tmp_path, ctx)
    assert report(resumed.call(action))["reason"] == "request_already_recorded_no_replay"
    assert len(mutations(ctx)) == 1
    with pytest.raises(ValueError, match="different arguments"):
        resumed.call(action | {"evidence": "changed"})


@pytest.mark.parametrize("mutation", [{"hands_empty": False}, {"arm_clear": False},
                                     {"frame_id": "nonexistent"}])
def test_required_current_scene_checks_before_prepare(tmp_path, mutation):
    ctx, steps = make(tmp_path)
    action = args(steps, "prepare", hands_empty=True, arm_clear=True) | mutation
    with pytest.raises(ValueError):
        steps.call(action)
    assert not mutations(ctx)


def test_new_message_pauses_running_grasp_without_restarting(tmp_path):
    ctx, steps = make(tmp_path)
    ctx.run_pending = True
    ctx.message_at = ctx.now + .5
    reply = steps.call(args(steps, "grasp", target_reachable=True, approach_clear=True))
    assert report(reply)["reason"] == "new_operator_message"
    assert [a["action"] for _, a in mutations(ctx)] == ["run", "pause"]
    assert report(reply)["operator_messages"]["pending"]


def test_budget_prevents_new_motion_but_keeps_observation_available(tmp_path):
    ctx, steps = make(tmp_path)
    action = args(steps, "prepare", hands_empty=True, arm_clear=True)
    ctx.now += 601
    assert report(steps.call(action))["reason"] == "task_budget_exhausted"
    assert report(steps.call(dict(action="observe")))["observation"]
    assert not mutations(ctx)


def test_early_checkpoint_pause_and_visual_confirmation(tmp_path):
    ctx, steps = make(tmp_path)
    ctx.run_pending = True
    reply = report(steps.call(args(steps, "grasp", target_reachable=True, approach_clear=True)))
    assert reply["active"]
    assert ctx.now < 103
    paused = report(steps.call(dict(action="pause")))
    first = paused["observation"]["frame_id"]
    second = observe(steps)
    confirmed = report(steps.call(dict(action="confirm_hold", verification_frames=[first, second],
                                       request_id="verification", frame_id=second,
                                       evidence="Both real frames show a suspended held toy")))
    assert confirmed["visual_grasp_verified"]
    assert not confirmed["last_result"]["task_success_verified"]
    assert [a["action"] for _, a in mutations(ctx)] == ["run", "pause"]


def test_read_pause_metadata_never_dispatches_an_extra_run(tmp_path):
    ctx, steps = make(tmp_path)
    ctx.run_pending = True
    steps.call(args(steps, "grasp", target_reachable=True, approach_clear=True))
    metadata = dict(request_id="checkpoint", frame_id="stale-metadata",
                    evidence="A description only, not a new movement request")
    assert report(steps.call(dict(action="check", **metadata)))["active"]
    assert report(steps.call(dict(action="pause", **metadata)))["active"] is None
    steps.call(dict(action="observe", **metadata))
    assert [a["action"] for _, a in mutations(ctx)] == ["run", "pause"]


def test_cannot_confirm_hold_from_one_source_frame(tmp_path):
    ctx, steps = make(tmp_path)
    frame = observe(steps)
    with pytest.raises(ValueError, match="distinct"):
        steps.call(dict(action="confirm_hold", verification_frames=[frame, frame], evidence="held"))


def test_ambiguous_submit_is_durable_and_not_replayed(tmp_path):
    ctx, steps = make(tmp_path)
    action = args(steps, "prepare", hands_empty=True, arm_clear=True)
    original = ctx.value

    def fail(tool, **kw):
        if tool == "g1_vla":
            raise ConnectionError("reply lost")
        return original(tool, **kw)

    ctx.value = fail
    with pytest.raises(ConnectionError):
        steps.call(action)
    ctx.value = original
    _, resumed = make(tmp_path, ctx)
    assert report(resumed.call(action))["reason"] == "request_already_recorded_no_replay"
    with pytest.raises(ValueError, match="outcome unknown"):
        resumed.call(args(resumed, "prepare", hands_empty=True, arm_clear=True) | {"request_id": "new"})


def test_observation_failure_pauses_only_owned_grasp(tmp_path):
    ctx, steps = make(tmp_path)
    ctx.run_pending = True
    action = args(steps, "grasp", target_reachable=True, approach_clear=True)
    ctx.observe_failure = True
    reply = report(steps.call(action))
    assert "observation_error" in reply
    assert [a["action"] for _, a in mutations(ctx)] == ["run", "pause"]


def test_ui_prompt_uses_compiled_steps_with_finite_budget():
    assert "g1_task_step" in ui_agent.FAST_TASK_INSTRUCTIONS
    assert "87度不能硬编码" in ui_agent.FAST_TASK_INSTRUCTIONS


def test_real_bridge_lists_compiled_tool_and_forwards_its_image(tmp_path):
    ctx = Context()
    ctx.request = lambda method, params: {"tools": []}
    atomic(tmp_path/"ankle_status.json", dict(session_id="session"))
    atomic(tmp_path/"ui-agent-status.json",
           dict(phase="running", batch="batch", updated_at=time.time()))
    requests = [
        dict(id=1, method="tools/list"),
        dict(id=2, method="tools/call",
             params=dict(name="g1_task_step", arguments=dict(action="observe")))]
    sink = io.StringIO()
    serve(ctx, tmp_path, "session", "batch",
          io.StringIO("".join(json.dumps(r)+"\n" for r in requests)), sink)
    listing, observed = [json.loads(line)["result"] for line in sink.getvalue().splitlines()]
    assert listing["tools"][0]["name"] == "g1_task_step"
    assert any(t["name"] == "g1_replay_route" for t in listing["tools"])
    assert observed["content"][1]["type"] == "image"
    assert not mutations(ctx)
