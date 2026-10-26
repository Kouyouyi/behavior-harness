---
name: autonomous-control
description: Run and debug Python robot control, then turn what worked by hand into a step-wise controller that automates the motions while your own perception checks the scene at the decisive points, and test it on the real scene with operator resets. Use for closed-loop scripts, switching between your script and direct control, and test runs.
---

# Autonomous control

Follow the workflow in AGENTS.md. The robot and Gemini are external services,
reachable only through MCP. The deliverable is a controller that completes the task
from a fresh scene with **code doing the motions and measurements and your perception
checking the scene where it matters**.

## Keep your perception in the loop

A real scene is not a fixed layout: people move objects, objects shift or tip when
touched, the light changes, the base slips. Code written against yesterday's scene
fails quietly on today's. So do not aim for blind full automation:

- **Code moves and measures; you look and decide.** Each step returns its measurements
  and the `frame_id`s it captured. At the decisive points (before the first contact,
  before closing or opening a gripper, before releasing or placing, after a base move,
  after any pause) look at the frames yourself{% if gemini %}, or ask Gemini a specific
  question about them,{% endif %} and only then run the next step.
- **Steps take perception as input.** A step that acts on an object takes the object's
  pose (or the pixels you chose) as a parameter that you have just verified, or runs its
  own localization and returns what it found for you to confirm. It never reuses
  positions from before the last motion or pause.
- **Checkpoints, not guesses.** When a check is ambiguous (the object is not where
  expected, depth is missing, a measurement is near its threshold), the step stops,
  leaves the robot safe and returns `ok: False` with `why`, the frames and what to look
  at, instead of carrying on.
- **Chain without looking only what has proved reliable.** Run steps back to back
  without your review only after their code checks have caught the failures you saw in
  testing; keep your review at every point where a mistake would drop, crush or knock
  something.
- Record in `memory/RUNS.md` where you looked, what you saw and what you decided, so the
  checkpoints that matter become visible.

### Who perceives what

- **Semantically rich perception by subagents**: what the objects are and which one is
  the target, whether the gripper is around the object, whether it slipped or tipped,
  whether the scene still matches what the step assumes, why a step failed. These are
  judgements from images, not rules to hard-code.{% if subagent_cap %} Hand them to a
  subagent (the Agent tool) with the frame paths and one precise question:
  - **structured perception on a fast model** (`model: haiku`): checks with a fixed
    answer form — is X visible (yes / no), where (pixel), how many, is the path clear,
    are both fingertips below the edge. We recommend the faster model for these: they are
    frequent, small and well specified. Use one at every checkpoint; run several in
    parallel when you need several answers.
  - **open-ended understanding on a stronger model** (`sonnet` or `opus`): what is going
    on in this frame, what changed between these two, why did this grasp fail.

  Their answers are evidence, not decisions: you decide, and when a subagent and the
  geometry disagree, observe again before acting. A perception subagent only looks:
  tell it not to call motion tools.{% else %} This session has no
  subagents: make these judgements yourself from the images{% if gemini %} or with
  Gemini{% endif %}.{% endif %}
- Each step's code returns the frames and the semantic question for its checkpoint, so
  the check can be delegated without you re-reading the code.

### Document what each subagent does

Every checkpoint in your controller is written down where the next reader (you after
compaction, a later session, the skill library) finds it without reverse-engineering the
code. A step that ends at a checkpoint returns, next to `ok`, `measured` and `why`:

```python
"checkpoint": {
    "subagent": "haiku",                    # fast model for structured checks; sonnet / opus for diagnosis
    "frames": ["runtime/frames/<id>/head_camera.jpg"],
    "question": "Are both fingertips below the box's top edge, one at each end face?",
    "answer": "yes | no, plus the pixel of each fingertip",
    "pass_if": "yes for both fingertips",
    "on_fail": "back off 3 cm outward, re-localize the end faces, retry once; then stop",
}
```

and `code/<project>/README.md` has a **Perception checkpoints** table with one row per
checkpoint: step, subagent type, frames, question, expected answer, pass criterion, and
what you do on fail. Write the question so a subagent that knows nothing else about the
task can answer it from the frames alone; say what to look at, not what you hope to hear.

## Start from your successful manual run

Read the saved successful strategy and decisive frames in `memory/`. Create a project
under `code/<project>/` and turn one useful maneuver into a Python function that
receives `ctx` and explicit inputs. Test it on the current scene using the ordinary
Python example below. Inspect its final images and logs, correct it, and repeat before
composing the full `main(ctx)`.

Separate reusable motion logic from scene-specific inputs: reacquire object locations
from current observations instead of embedding one layout's positions.

## Turn successful control into automated scripts

After succeeding through direct robot control, reproduce the decisions that made it
work in an observation–decision–action script, step by step, keeping the checkpoints
above. Use
current observations to adapt to new layouts, rather than blindly replaying one run's
actions.

- Hard-code reusable motion functions and decision rules on the MCP tools
  (`{{ robot.move }}`, `{{ robot.move_joints }}`{% if robot.gripper %}, `{{ robot.gripper }}`{% endif %}) when suitable;
  [robot-interface](../robot-interface/SKILL.md) has the frames and limits.
{% if gemini %}
- **For perception in code, read the [gemini skill](../gemini/SKILL.md) before you write
  any detector.** Gemini identifies, points at and segments objects in a few seconds;
  use its helper module `vlm.py` and keep geometry (depth, kinematics) in your code.
{% endif %}
- Practise and debug functions on the current scene, using direct robot calls to
  rearrange objects into the test conditions you need where that is safe, rather than
  asking for a reset every time.

Combine these approaches as needed, and test the resulting closed-loop script on the
current scene before the end-to-end test runs below.

## Model the invariant motion; perceive what changes

Write the controller so a new layout, a changed scene, or a new task needs new
**perception and parameters**, not new motion code.

- **Separate the invariant from the variable.** The motion recipe is what stays the same
  across scenes: approach along the object's surface normal from a standoff, the grasp
  offsets and contact strategy, the lift and carry posture, the place-and-release
  sequence. Express it **in the object's or the target's frame** (relative offsets,
  directions, clearances), never as base-frame coordinates from one run. What changes —
  where the object is, its size and orientation, the support surface height, obstacles,
  which arm can reach — is measured fresh by perception on every run and fed in as
  inputs.
- **A perception layer with a fixed output.** One function turns the current scene into a
  small scene description the motions consume, for example `{target: {pose, size, kind},
  support: {height, edge}, obstacles: [...], free_space: ...}`. Geometry comes from depth
  and the tools{% if subagent_cap %}; semantic fields (which object is the target, its kind,
  whether the support is clear) come from perception subagents{% endif %}. A new environment
  or object type plugs in a different perception implementation or object description;
  the motions do not change.
- **Plug-in structure for new tasks.** Keep three layers in separate modules: perception
  (scene -> description), motion primitives (description + parameters -> robot motion,
  each with its checks), and a short task script that chains them. A new task is a new
  task script, plus new primitives only for motions that do not exist yet; reuse
  primitives from the [skill library](../continual-learning/SKILL.md) where they exist.
- **Parameters as data.** Object descriptions (size ranges, grasp side, how to recognise
  it), tolerances and per-environment settings live in a small config dict or file, not
  scattered constants, each with the range it was tested on. A scene outside a tested
  range is a checkpoint: stop and look rather than extrapolate.
- **Choose at run time, not at writing time.** Decide which arm, which approach side,
  whether to drive closer or lean, and whether the torso must help from the perceived
  scene (reach and posture checks with the robot's planning tools, see
  [robot-interface](../robot-interface/SKILL.md)), so the same code handles the object
  to the left, to the right, nearer or further.
- **Plan with perception-based heuristics, not brute-force search.** Do not grid-search
  stances, routes, approach angles or IK seeds. Look at the scene, then pick a few
  candidates by reasoning: face the object along its free side, approach straight along
  the surface normal, keep the path through floor you can see is clear, prefer the arm on
  the object's side. Check those few with depth and the robot's planning tools
  ([robot-interface](../robot-interface/SKILL.md)) and take the first that passes. If none pass, observe again or move the
  viewpoint rather than widening the search. A heuristic you can explain from an image is
  easier to verify, debug and reuse than a search result.

## Pressure-test across layouts

Do not stop at one success. The operator can reset the scene to new layouts on
request; reacquire objects and adapt approach poses instead of reusing layout-specific
coordinates. Vary what the invariants claim not to depend on — position, orientation,
distance, side, distractors nearby, lighting — and record which variations each run
covered; a variation that breaks the code shows which quantity was not perceived. Refresh observations after anything moved. Record each tested layout,
outcome, failure and fix in memory; you stepping in is debugging, not an autonomous
success. Add observation-driven recovery for expected perception or grasp failures
(re-observe, re-localize, retry from a safe height). Do not claim untested
generalization.

## Code you can read and resume

- Split the pipeline into steps, each with an observable precondition and completion
  check.
- Give each step a function `fn(ctx, ...)` that checks its preconditions from fresh
  observations, acts, verifies the result, and returns a small dict with `ok`, what it
  measured and, on failure, why. It returns instead of improvising when the scene is
  outside what it was tested on, leaving the robot in a safe state (no held object
  dropped mid-air, arm lifted clear, nothing left moving).
- Make the pipeline resumable at any step, and give every motion that can fail a
  fallback.
- Keep tunable values as named constants at the top of a module, with the range you
  tested.

`controller.py:main(ctx)` chains the same steps end to end and stops at the first one
that fails, returning a report. `code/<project>/README.md` says how to run the whole
controller and each step from ordinary Python, with their preconditions; record
reusable tips as [experience-memory](../experience-memory/SKILL.md) describes.

## Ordinary Python

Python runs in this workspace with normal files, stdout and stderr. Write reusable
functions against a supplied context `ctx`; the entrypoint is the same for single
steps and full runs:

```python
# code/stack/controller.py
def main(ctx):
    reply = ctx.value("{{ robot.observe }}")
    # Your observation–decision–action loop, e.g.:
    # {{ robot.motion_example }}
{% if robot.grasp_example %}
    # {{ robot.grasp_example }}
{% endif %}
    return {"ok": True}


if __name__ == "__main__":
    import json
    from api.runtime import Context
    with Context() as session:
        print(json.dumps(main(session), indent=1))
```

Run it from the workspace root (`python code/stack/controller.py`, with stdout and
stderr redirected to a log under `output/runs/<timestamp>/`). `main(ctx)` and its
helpers use the supplied context unchanged, passing `ctx` down: no new `Context()`.
`Context()` connects to the same robot service as your direct tool calls. Return
whenever you want to inspect images and choose new actions yourself; the scene keeps
the script's actions.

Robot calls run one at a time, but a script does not hold the robot: other calls can
come between its calls. Run one controller at a time, and stop it before acting
directly. Interrupting Python does not cancel a motion already dispatched: after an
interruption or an ambiguous motion error, check `{{ robot.status }}` and a fresh
observation before continuing.

## MCP and files

`ctx.call(tool_name, **arguments)` uses the live tool schemas and returns MCP content
blocks; `ctx.value(...)` decodes the JSON text block. Images remain image blocks.
Inspect current tool schemas (`MCP_SESSION.md`) instead of inventing arguments. Calls
reporting an MCP error raise `RuntimeError` with the error JSON (stage, reason,
`robot_state` where the arm stopped); handle refusals in code rather than blindly
retrying the same motion. Observations save full-resolution frames under
`{{ vocab.frames }}<frame_id>/`; save script replies, decisions and prints to files under
`output/` for debugging.

## End-to-end test runs

There is no formal evaluation: you judge the controller by running it end to end on
the real scene.

1. Bring the robot to a neutral state ({{ robot.neutral }}), then call `request_operator` with kind `reset` and say what you need, e.g.
   "please put the cubes back at new positions on the table". Wait for the answer.
2. Commit the controller version (`git` inside `code/<project>/`) and run it once,
   unattended: no direct robot calls while it runs. If it fails, read its report, log
   and frames, fix the code and go back to step 1; do not hand-finish a run and count
   it as a success.
3. Record each run in `memory/RUNS.md`: code version, scene description, outcome,
   duration, failure step and cause.

Aim for several consecutive clean successes on varied layouts before calling it done.
Use `request_operator` kind `help` when you are blocked by something outside your
control (an object out of reach, a robot fault, unclear task wording).
