---
name: base-navigation
description: Drive the RB-Y1's differential-drive base safely - prepare the body, look along the path, make short moves closed on wheel odometry and re-localize afterwards. Use before every rby1_base_move.
---

# Base navigation

The base has two driven wheels: it drives forward / backward and turns in place; it
cannot move sideways. `rby1_base_move` closes its loop on the wheel odometry, which slips
on smooth floors and drifts over time, so you only know where the robot really ended up by
looking again. Short moves, a check before each, a fresh observation after each.

There is no obstacle sensor behind this tool: the backend does not stop for obstacles.
Seeing the path clear is your job, from head-camera images.

## Before each move

1. Arms `ready` or `park`, torso `ready` or `zero` ([whole-body-control](../whole-body-control/SKILL.md));
   nothing held out.
2. Look along the path: `rby1_look_at` a point about 1.5 m away in the travel direction at
   z = 0.3 (for a turn, look to that side first), `{{ robot.observe }}`, and check the image
   for people, cables, low objects and table edges. The camera cannot see behind the
   robot: drive backward only along a path you saw before, and at most 0.3 m.

## The move

`rby1_base_move {forward?, turn?, max_linear?, max_angular?}`: straight `forward` (m, +
forward, − backward), then an in-place `turn` (rad, counter-clockwise +). The backend refuses to drive with the torso
below `mid`, the chest leaning, or a hand stretched out (more than 0.72 m forward or to the
side): bring the body in first. Caps: 0.15 m/s,
0.3 rad/s, 1 m and π rad per call. Prefer ≤ 0.5 m per call and one of the two per call.
The reply gives `measured_by_odometry` (`forward`, `sideways`, `turn`) and the new
`odometry`; a stuck base (no odometry change for 2 s) or the deadline is a handled stop.

## After each move

Every base-frame position from before is stale. Observe, find the target again, and
re-localize it ([object-localization](../object-localization/SKILL.md)); compare its new
position with the expected one to estimate the real displacement before the next move.
