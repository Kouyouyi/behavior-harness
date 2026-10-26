---
name: live-stream
description: The continuous read-only recording of the robot's cameras and joint state over the last 10 minutes. Use before acting after any pause (thinking, writing code, waiting), to see what changed while you were not looking, and to review what happened during a motion or a script run.
---

# Live stream: the world keeps changing

This is the real world. Between your observations people may move or remove objects,
things settle, slide or fall, and seconds to minutes pass while you think or write code.
A frame you looked at earlier describes the past, not the present.

`runtime/stream/` (read-only) holds the last 10 minutes, continuously:

- `<camera>/<UTC minute>/<epoch_ms>.jpg`: frames of {{ robot.cameras | join(', ') }};
  `<epoch_ms>_depth.png` about once per second where the camera has depth (uint16
  millimetres, 0 = invalid).
- `robot/<UTC minute>.jsonl`: the robot's joints and end-effector positions.
- `index.json`: the cameras, the newest frame time per camera and the measured rates.

Older data is deleted as time goes by; copy what you need to keep into `output/`.

## Two ways in

- **Directly, as a tool:** `{{ robot.stream }} {seconds_ago: [0, 30, 120]}` returns the
  frames from those moments (all cameras, or `cameras: [...]`), labelled with their age,
  plus the robot state then; `{{ robot.stream }} {sheet: {seconds: 120, count: 12}}`
  returns one contact sheet per camera over the last two minutes. No motion, no arming
  needed.
- **As files,** for code and bulk analysis: `python -m api.stream ...` below, or read the
  images under `runtime/stream/` directly.

## When to look

- **Before any motion that follows a pause**, check that the scene still matches what
  you planned against: `{{ robot.stream }}` with `seconds_ago: [0]` (or call
  `{{ robot.observe }}`, which also saves depth and camera geometry for localization).
  If something moved, re-localize before moving; never act on stale positions.
- **What changed while you were away**: `{{ robot.stream }}` with a `sheet` over the time
  you were away, or `python -m api.stream changes {{ robot.overview_camera }} --seconds 300`
  to list the moments the image changed and then `{{ robot.stream }} {seconds_ago: [...]}`
  at those moments. Compare with `python -m api.stream robot --seconds 120`: changes while
  the robot was still were caused by something else (a person, a falling object).
- **After a motion or a script run**, review the frames during it (a slip, a knocked
  object, an arm that stopped early) instead of guessing from the end state.
- In controllers, use `from api import stream` to check that the scene is quiet before
  committing to a motion, and to keep evidence frames of each step.

Metric geometry still comes from `{{ robot.observe }}` frames and
`{{ robot.pixel }}` ([object-localization](../object-localization/SKILL.md)); the stream is
for timing, change and review. Look at a few selected frames or sheets, never whole
minutes of images.
