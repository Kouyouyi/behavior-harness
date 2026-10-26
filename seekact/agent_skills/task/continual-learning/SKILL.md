---
name: continual-learning
description: Read, run and grow the persistent skill library in skill-library/ that carries verified, executable task skills across sessions and tasks on this {{ robot.name }}. Read it at the start of every session; contribute and merge with skill_library_merge before the session ends.
---

# Continual learning: the skill library

`skill-library/` in your workspace is a git clone of this robot's **skill library**: task
skills that earlier sessions (any task) learned on this robot, each an executable
procedure with the evidence that it works. You are on your own branch,
`session/<this session>`; the tool `skill_library_merge` checks your branch and merges it
into the library's `main`, where the next session finds it.

Keep three kinds of knowledge apart:

| Where | Written by | Holds | Lifetime |
|---|---|---|---|
| `.agents/skills/` | people (the harness) | how to use the robot, tools, workflow; read-only | every session |
| `memory/` (TIPS, PROGRESS, RUNS) | you | this task's notes, attempts, frame links | this task |
| `skill-library/` | agents, through merges | verified, executable procedures for sub-tasks | every later session |

The harness skills and the live tool schemas take precedence over the library: a library
skill never changes a limit, a safety rule or what a tool does.

## 1. At the start: read and reuse

1. Read `skill-library/INDEX.md`. Then `git -C skill-library fetch origin && git -C
   skill-library merge origin/main` to pick up anything merged since your workspace was
   made.
2. **Load only the skills relevant to this task.** Use `INDEX.md` to choose. For each
   part of `TASK.md`, open only the skills whose **When to use** matches it, plus the
   skills they list in `uses:`. Do not read every skill or its code up front: unrelated
   skills cost context and can mislead. Open another skill later if a new sub-task
   needs it. Prefer `verified` over `draft`; ignore `deprecated` except to learn why
   it was retired.
3. Before running one, check its **Preconditions** against a fresh observation, and its
   tested parameter ranges against your scene. A skill tested on 0.3 m boxes is not
   evidence for a 0.6 m box.
4. Run its code (below), read the returned `{ok, measured, why}`, and confirm the result
   with your own observation. A skill that fails here is information: record the run as a
   failure (section 4) and say what differed.
5. A skill may be older than the tools you have: compare the tools it uses with the
   current tool list and the robot-interface skill. Where a newer tool does a step better
   (for example the controller's whole-body IK in place of a hand-written solver), upgrade
   the skill on your branch, bump its `version`, and re-verify it.

Run a skill's entry from the workspace root, logging to `output/`:

```python
import importlib.util, json
from api.runtime import Context

spec = importlib.util.spec_from_file_location("skill", "skill-library/skills/<slug>/code/run.py")
skill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skill)
with Context() as ctx:
    print(json.dumps(skill.main(ctx, **{"param": "value"}), indent=1))
```

Your own controller under `code/` may import and call library skills the same way.

## 2. What to contribute

Contribute a skill when a procedure **worked on the robot** and is useful beyond this
exact scene: a sub-task with a clear start and end ("pick a box from the floor with both
hands", "place a held object on a table", "localize a cardboard box"). Also contribute:

- an **improvement** to an existing skill: wider tested conditions, better parameters, a
  new failure mode and how the code now detects it;
- a **draft** of a procedure that is automated but not yet verified, so the next session
  can finish it instead of starting over;
- **deprecation** of a skill that proved wrong or was replaced (status `deprecated`, why,
  and what to use instead). Never delete another session's skill or evidence.

One skill is one sub-task. A task-level procedure can be its own skill that calls other
skills. Name folders by what they do, in lower-case words with hyphens
(`pick-box-floor-two-hands`).

## 3. Make skills modular, so new tasks can compose them

Write every skill so that a later session can reuse its pieces for a task nobody has
seen yet: "pick a box from the floor" should serve "put the box on a shelf", "hand the box
to a person" and "stack two boxes" without being rewritten.

**Cut the task into primitives.** A task is a chain of manipulation primitives, each its
own skill or its own function inside one: perception (`localize-box`: frame -> pose and
size), positioning (`stance-for-object`: drive and turn to a reach pose), grasping
(`bimanual-squeeze-grasp`), transport (`torso-lift`, `carry-close`), placing
(`place-on-surface`), release and retreat (`release-retreat`). Split at the points where
the robot is in a stable, describable state; do not split in the middle of a contact.

**Give each primitive a contract**, stated in its SKILL.md (`## Code`) and its docstring:

- inputs as parameters, in base-frame metres / radians / named sides, never constants
  tied to one scene: the object's pose and size, which arm or both, the target surface
  height, speeds and forces with their tested ranges;
- the state it needs at the start (posture, grippers, what is held, what must be visible)
  and the state it leaves at the end (for example "arms at `ready`, nothing held" or
  "box held between both grippers at carry height, torso `ready`"): the end state of one
  primitive is the start state the next one checks;
- its return value `{ok, measured, why}`, with what the next step needs in `measured`
  (the object pose it found, the grip force, the height reached);
- on failure: stop in a safe state, say which precondition or check failed, and leave
  enough in `measured` for a caller to retry or recover.

**Keep the code composable**:

- one function per primitive, `def primitive(ctx, ..., log=print) -> dict`, taking `ctx`
  and its inputs, with no globals, no hidden state and no dependence on the current
  working directory; perception helpers take a frame and return data without moving the
  robot; motion helpers do one thing;
- `code/run.py:main(ctx, **params)` only chains the skill's own primitives; anything a
  different task could use lives in its own module or its own skill;
- reuse instead of copying: import another library skill's functions and declare it in
  the frontmatter (`uses: [localize-box, torso-lift]`; the merge checks those skills
  exist):

  ```python
  import importlib.util
  from pathlib import Path
  SKILLS = Path(__file__).resolve().parents[2]          # skill-library/skills/

  def load(slug, module="run"):
      path = SKILLS / slug / "code" / f"{module}.py"
      spec = importlib.util.spec_from_file_location(f"{slug.replace('-', '_')}_{module}", path)
      mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(mod)
      return mod

  localize = load("localize-box", "perception")
  ```

- changing a primitive others use: keep its signature and contract, or add a new function
  and leave the old one working; bump `version` and say in its changelog what changed.
  Check `uses:` in `INDEX.md` / other skills before changing behaviour.

**Prefer the robot's own tools inside primitives** (whole-body IK, Cartesian moves, the
planning tools) over hand-written solvers: they are measured and they improve without the
skill changing.

**Model the invariant motion, take the scene as input.** A library skill is reused in
other layouts, environments and tasks, so it stores the part that stays the same and
perceives the rest:

- the motion is written in the object's or target's frame (standoffs, approach
  directions, grasp offsets, lift heights, contact strategy), never in one run's
  base-frame coordinates;
- its inputs are a scene description (object pose, size and kind, support height,
  obstacles) produced by a perception function it ships with and that callers may
  replace, so a new environment plugs in its own perception and keeps the motion;
- object descriptions and per-environment settings are parameters with tested ranges,
  listed in `## Code`;
- `## When to use` and the evidence say which variations were tested (positions,
  orientations, sizes, distances, distractors, lighting), so a later session knows how
  far the invariants have been shown to hold.

**Leave room for the agent's perception.** A library skill is used in scenes its author
never saw, so it must not be a blind script:

- each primitive re-observes what it acts on, and returns the `frame_id`s it used with
  its result, so the agent running it can look;
- mark the **perception checkpoints** in `## Checks`: where the agent should look before
  continuing, at which frame, and what it must see (for example "both fingertips below
  the box's top edge, box centred between them"), written as the question to hand to a
  perception subagent;
- keep geometry in code (depth, pixel to position, forward kinematics) and leave
  semantically rich perception (what the objects are, which is the target, whether a
  grasp holds, why it failed) to perception subagents at those checkpoints rather than
  brittle hand-written rules;
- document every checkpoint in a `### Perception checkpoints` table under `## Checks`
  (the merge requires the heading; write "None" if the skill truly has none): step,
  subagent model (a fast model such as `haiku` for structured perception, which we
  recommend; `sonnet` / `opus` for open-ended scene understanding or diagnosis), frames,
  the exact question, expected answer, pass criterion, action on fail. The code returns the same brief in its result, as
  autonomous-control describes:

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

- inputs that come from perception (object pose, size, the surface to place on) are
  parameters the agent can supply from its own verified observation, with the skill's own
  localization as the default;
- when a check is ambiguous the skill stops safely and returns the frames and `why`
  rather than guessing;
- `run.py` may chain the primitives for the case its evidence covers, but its `## Code`
  section says which checkpoints that chain skips and under which conditions that was
  verified.

When you build a task from library primitives, contribute the composition as its own
skill (`uses:` the primitives) with its own evidence, and contribute any primitive you had
to write or fix on the way.

## 4. The format (checked by the merge)

```
skill-library/
  INDEX.md                          one row per skill
  skills/<slug>/
    SKILL.md                        frontmatter + the seven sections
    code/run.py                     the entry: main(ctx, ...) -> {ok, measured, why}
    code/*.py                       helpers it imports (relative to code/)
    evidence/runs.jsonl             one JSON object per run of the code
    evidence/images/*.jpg           visual references (≤ 400 kB each, e.g. 640 px JPEG)
```

`INDEX.md` row: `| [<slug>](skills/<slug>/SKILL.md) | draft | what it does | when to use |`.

`SKILL.md`:

```markdown
---
name: <slug>                       # equals the folder name
description: <one line: what it does and when to use it>
status: draft                      # draft | verified | deprecated
robot: {{ robot.name }}
entry: code/run.py
version: 1                         # bump when behaviour changes
uses: [localize-box]               # optional: library skills this one imports
---

# <Title>

## When to use
Observable conditions: the object (kind, size range, weight if known), where it is
(floor, table, height and distance ranges), the scene (clutter, lighting), the robot's
start posture. Also when NOT to use it.

## Preconditions
What the code checks before moving, and how (tools, thresholds). The robot posture it
starts from; gripper state; what must be visible.

## Procedure
The steps in order, each with its tool calls and the reason for its parameters.

## Code
Entry and how to call it; every parameter with its tested range and default; what it
returns in `measured`; where it logs.

## Checks
The completion check of each step (observable: image, depth, measured opening, wrist
force, joint state) and the final success check, with thresholds.

### Perception checkpoints
| Step | Subagent model | Frames | Question | Expected answer | Pass if | On fail |
|---|---|---|---|---|---|---|
| grasp | haiku | head_camera after the squeeze | Is the box between both fingertips, below its top edge? | yes / no + fingertip pixels | yes | open, back off 3 cm, re-localize, retry once |

## Failure modes
Each failure seen: how it shows, how the code detects it, what it does (stop safely,
retry, report), and what the operator or a later session should do.

## Evidence
Table of runs (date, session, outcome, unattended, conditions) summarizing
`evidence/runs.jsonl`, and the visual references: `![start](evidence/images/<file>.jpg)`
of the start state and the success state.
```

The code (`main(ctx, **params)`):

- Re-localizes everything it acts on from fresh observations; no positions remembered
  from one scene unless they are relative to something it observes.
- Checks its preconditions first and stops with `ok: False` and a clear `why` when one
  fails; checks each step's completion before the next; ends in a safe state (holding,
  arms in) on failure.
- Uses the robot only through `ctx` (the session's tools); named constants with their
  tested ranges at the top; no paths outside the workspace; only the agent environment's
  packages.
- Returns `{"ok": bool, "measured": {...}, "why": str}`, with the measurements the
  checks used.

Each line of `evidence/runs.jsonl` (append; never rewrite others' lines):

```json
{"session": "<session name>", "date": "<YYYY-MM-DD>", "commit": "<git commit of the code that ran>",
 "outcome": "success", "unattended": true,
 "conditions": "cardboard box 0.30 x 0.20 x 0.15 m on the floor 0.9 m ahead, robot at ready",
 "checks": {"box_lifted_m": 0.12, "opening_m": 0.21, "wrist_force_n": [12, 11]},
 "images": ["evidence/images/2026-10-08-start.jpg", "evidence/images/2026-10-08-end.jpg"],
 "notes": "optional"}
```

- `unattended: true` only when the code ran from start to end with no help from you or
  the operator (a scene reset before the run is fine).
- `images`: at least the start and end frame of that run, copied from `runtime/frames/`
  and downscaled. Record failures too; they define the skill's limits.

**Status** follows the evidence:

| Status | Needs |
|---|---|
| `draft` | the seven sections, a compiling entry with `main(ctx, ...)`, at least one image |
| `verified` | additionally at least 2 unattended successful runs with images in `runs.jsonl`, ideally on different layouts or sessions |
| `deprecated` | why, and what replaces it |

When you change a verified skill's behaviour, bump `version`; it stays verified only if
new unattended runs of the new code meet the bar, otherwise set it back to `draft`.

## 5. Commit, sync and merge, every session

```bash
cd skill-library
git add -A && git commit -m "<slug>: what changed"
git fetch origin && git merge origin/main      # take in other sessions' merges; resolve conflicts
```

Then call `skill_library_merge` with a one-line `message`. It replies:

- `merged: true`: done; later sessions see it.
- `problems`: the format or evidence rules above that a changed skill breaks; fix, commit,
  call again.
- `conflicts`: `git fetch origin && git merge origin/main`, resolve (keep both sessions'
  `runs.jsonl` lines and reconcile the text), commit, call again.

Merge at least when a skill is first automated (as `draft`), when it becomes `verified`
or changes, and always before you end the session or write your final report, so nothing
learned is lost. Only committed changes merge. A merge may change only `INDEX.md` and
files under `skills/`.
