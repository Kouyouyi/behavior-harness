---
name: whole-body-control
description: Coordinate the RB-Y1's base, torso, head and both arms for one task - choosing where to stand, the torso height and lean for the work height, where to look, which arm, and the order of moves. Use whenever a task needs more than one body part or a target is not comfortably in reach.
---

# Whole-body control

Treat the robot as one kinematic chain: the base sets where the shoulders are on the
floor, the torso sets their height and lean, the head sets what the camera sees, and the
arm reaches. Decide the whole sequence first, then execute it stage by stage with an
observation between stages.

## Torso (computed from the robot's URDF)

Six joints: `torso_0` and `torso_4` roll sideways and stay 0 (the upright rule);
`torso_1`, `torso_2`, `torso_3` pitch; `torso_5` turns the chest (yaw). With
`torso_1 = a`, `torso_2 = −2a`, `torso_3 = a` the chest stays upright and only lowers:

| a | `torso` positions (rad) | Shoulder height | Camera height (head 0) |
|---|---|---|---|
| 0 (`zero`) | `[0, 0, 0, 0, 0, 0]` | 1.37 m | 1.52 m |
| 30° (`ready`) | `[0, 0.524, −1.047, 0.524, 0, 0]` | 1.28 m | 1.43 m |
| 45° | `[0, 0.785, −1.571, 0.785, 0, 0]` | 1.17 m | 1.32 m |
| 60° (`low`) | `[0, 1.047, −2.094, 1.047, 0, 0]` | 1.02 m | 1.17 m |
| 69° (`lowest`, the cap) | `[0, 1.2, −2.4, 1.2, 0, 0]` | 0.92 m | 1.08 m |

The torso squats at most to `torso_1 = 1.2` rad: at 1.28 torso_2 met a real end stop
before its model limit, and pushing on it faulted the robot (every joint dropped). For
anything lower than the `lowest` row reaches, drive closer and reach down with the arm.

Leaning: raise `torso_3` above `a` by `b` and the chest tilts forward by `b` (shoulders
move forward about 0.07 m per 10°, the camera pitches down by `b`); at `a` = 30°:
`b` = 20° puts the shoulders 0.13 m forward at 1.25 m. Turning: `torso_5` (+ left)
swings both shoulders about the chest. Each arm reaches at most 0.69 m from its
shoulder (`out of reach` beyond that), so the shoulder height sets the work height.

Arm presets: `ready` (right `[−45, −30, 0, −90, 0, 45, 0]°`, left mirrored: forearms
forward at chest height, the vendor's demo pose), `half` (halfway from hanging to
`ready`), `park` (elbows bent, hands at the sides, out of the camera's view), and the
stretched postures `forward`, `forward_up`, `overhead`, `forward_out`, `side` (see the
posture library below). Torso presets: `zero`, `half`, `ready` (30°), `mid` (45°), `low`
(60°), `lowest` (69°, the squat cap). With the
arms hanging straight down (the rest pose, all zero) joints 1 and 3 sit on their limits:
that pose is not a valid target, and leaving it towards `ready` takes `half` first (one
call moves a joint at most 1.2 rad). Head: `head_0`
pans (+ left), `head_1` tilts (+ down); pan ±0.47 rad, tilt −0.30 … 1.52 rad. Use
`rby1_look_at` instead of computing these.

## Posture library (run on this robot, 2026-10-08)

Every row was driven through these tools at the backend's speeds with a person at the
e-stop (`scripts/rby1_large_motion.py`). Hands are the end-effector positions in `base`
(right arm; the left is the mirror image, y negated); load is the highest holding torque as
a fraction of that joint's rating (`rby1_status` → `safety.highest_torque_load`).

| Posture (preset) | How | Hands at (x, y, z) m | Load | Time |
|---|---|---|---|---|
| `ready` torso + arms | — | (0.57, −0.29, 1.29) | torso_3, shoulders ≈ 40 % | — |
| torso `mid` (45°) | `rby1_move_body` torso | z 1.18 | ≈ 40 % | 6.2 s from `ready` |
| torso `low` (60°) | torso, from `mid` | z 1.04 | torso_2 ≤ 54 % | 6.2 s from `mid` |
| torso `lowest` (69°, the squat cap) | torso, from `low` | z 0.94 | torso_2 42–47 % (no drift in 5 s) | 4 s from `low` |
| lean +0.15 at `ready` | torso_3 + 0.15 | (0.63, −0.29, 1.20) | torso_3 48 % | 2.5 s |
| lean +0.30 at `ready` | — | refused with the earlier 75 % load check; allowed now (check 130 %), not yet run on the robot | | |
| lean +0.15 at any height, arms `ready` | `[0, a, −2a, a + 0.15, 0, 0]` | hands 8 cm lower and 5 cm further forward than at that height upright: z 1.30 (`zero`), 1.20, 1.09, 0.95, 0.85 (`lowest`) | torso_3 49–56 % | 1.5 s from upright |
| + a hand 0.08 m forward while leaning +0.15 | `rby1_move_eef`, relative | ends within 1.4 mm | torso_3 ≤ 56 % | 3–4 s |
| the same reach with the torso `zero` | — | refused with the earlier 75 % load check; allowed now, not yet run | | |
| lean +0.3 at `mid` with the arms `park` | arms in first | hands at the sides, z 0.80 | ≤ 48 % | 2 s |
| chest yaw ±0.4 | torso_5 | hands swing ≈ 0.25 m sideways | ≤ 44 % | 5 s per 0.4 rad |
| arm `forward` | from `ready` | (0.69, −0.17, 1.33) | shoulder 55–64 % | 2 × 4.3 s |
| arm `forward_up` | from `forward` | (0.44, −0.01, 1.77) | shoulder ≤ 66 % | 4.9 s |
| arm `overhead` | from `forward_up` | (0.07, 0.05, 1.91) | ≤ 52 % | 3.7 s |
| arm `forward_out` | from `ready` | (0.61, −0.54, 1.27) | shoulder 53 % | 2 × 3.9 s |
| arm `side` (raised 60° sideways) | from `forward_out` | (0.10, −0.81, 0.94) | shoulder roll 52–64 % | 2 × 4.0 s |
| arm raised 90° sideways | — | **stopped by the torque guard** (left shoulder 81 %); the right peaked at 73 %. Do not raise an arm sideways above `side` | | |
| both arms `forward`, `ready` torso | both arms in one call | (0.69, ∓0.20, 1.34) | torso_3 ≈ 47 % | 2 × 4.3 s |
| table-height reach | torso `low` + arm `[-0.9, -0.15, 0, -0.6, 0, 0.3, 0]` | (0.62, −0.18, 0.75) | ≤ 54 % | 5.8 s |
| 0.1 m lower from there | — | refused: out of reach (0.73 m from the shoulder) | | |
| same arm with torso `lowest` | torso `lowest` + that arm | (0.62, −0.18, 0.65) | ≤ 56 % | 5.8 s |
| forearm roll ±1.0, wrist roll ±1.2 at `ready` | arm joints 4 / 6 | hand turns in place | ≤ 54 % | 4–6 s |

What this means for planning:
- Work heights: hands at about 1.3 m with `ready`, 1.2 m at `mid`, 1.0 m at `low`, 0.94 m
  at `lowest`; with the arm reaching down, 0.75 m (`low`) or 0.65 m (`lowest`) at 0.62 m
  in front. Anything lower or
  further: drive closer first (each 0.1 m closer buys roughly 0.1 m of extra depth).
- Reach: about 0.69 m forward of the chest, 1.9 m up (`overhead`), 0.81 m to the side
  (`side`, hand at 0.94 m high). Raising an arm higher out to the side overloads the
  shoulder.
- Large posture changes take two or three calls: one call moves a joint at most 1.2 rad,
  so `ready` → `forward_up` / `overhead` goes through `forward`, and `ready` → `side`
  through `forward_out`. Come back the same way.
- Leaning: tested on the robot up to +0.15 rad (torso_3 = torso_1 + 0.15) with the arms at
  `ready` at every height (a hand then reaches about 0.08 m further forward), and +0.3 with
  the arms tucked (`park`). The backend now allows more: with the arms at `ready` about +0.72
  rad at `zero` / `ready`, +0.68 at `mid`, +0.42 at `low` and +0.27 at `lowest` (the last
  two: the torso_3 joint limit); with the arms tucked up to +0.94 at `ready`. At +0.7 the
  hands are about 0.7 m ahead at 0.8–0.9 m high and the centre of mass is still 0.14 m
  behind the wheel axle. Beyond +0.15, lean in steps of 0.1 rad, with
  `rby1_check` first and `safety.highest_torque_load` after each step, and keep loads in
  the hands small: the hands move about 5 cm down and 3 cm forward per 0.1 rad. Lean for the last few centimetres of height and reach, after the squat:
  `lowest` + lean +0.15 puts the hands at 0.85 m without reaching down. Stretched arms and
  leaning add up on torso_3 and the shoulders: stretch one arm at a time while leaning.
- Before driving: torso at `mid` or higher, not leaning, hands in (`ready` / `park`); the
  backend refuses to drive otherwise.
- The right wrist reads 50–85 % of its small 10 N m rating during ordinary moves (more than
  the model predicts); brief peaks are normal. A payload in the right hand reaches the wrist
  guard first: carry it close to the body with the wrist near its `ready` angles.

## Sequence for reaching a target

1. **See it.** `{{ robot.observe }}`; if the target is off-centre, `rby1_look_at` it and
   observe again; localize ([object-localization](../object-localization/SKILL.md)).
2. **Check reach** from where you stand: `rby1_check` a `{{ robot.move }}` to the pre-pose
   0.10 m from the target (and to the target): refused with `out of reach` beyond 0.69 m
   from the shoulder. `rby1_fk` with `joints: {"torso": {"preset": "mid" | "low" |
   "lowest"}}` shows where the shoulders and hands would be at another torso height, and
   `hand_from_shoulder_m`; `rby1_check` the posture change itself. `rby1_optimal_control`
   (component `body`) is the robot controller's whole-body IK: it moves torso and arms
   together towards hand targets (measured: 10 cm down in 7.7 s within 0.2 mm, the torso
   taking 0.11 rad of it). Use it when the arm alone is short or both hands must move
   together; see robot-interface for the numbers.
   Beyond 0.69 m, or far below / above the shoulder:
   - target low → lower the torso (larger `a`, at most 1.2 rad), then lean forward with
     `torso_3`; still too low → drive closer rather than squatting further;
   - target far forward → lean (`torso_3` above `a`);
   - still out of reach → drive ([base-navigation](../base-navigation/SKILL.md)).
3. **Drive if needed**, body still: arms `ready` or `park`, torso `ready` or `zero`, then
   short base moves; afterwards observe and re-localize.
4. **Set the posture**: `rby1_move_body` with `torso` and `head` together, then
   `rby1_look_at` the target and observe. The torso moves at 0.1 rad/s (about 10 s per
   radian of its largest joint change) and one call changes a joint by at most 1.2 rad.
   Change it in steps of at most 0.3 rad and check `safety.highest_torque_load` after each.
5. **Reach**: `{{ robot.move }}` to the pre-pose in ≤ 0.25 m steps, observing after each;
   then the final approach (≤ 0.10 m). A self-collision refusal means go around: an
   intermediate arm posture (`ready`) first.
6. **Retract** the arm towards `ready` before any torso change or drive.

## Coordination rules

- One arm works, the other holds still. Pick the arm on the target's side (y > 0 left,
  y < 0 right).
- `rby1_move_body` sends all listed parts as one command timed to finish together; the
  slowest part (usually the torso) sets the pace. It never moves the base.
- The base never moves together with arms or torso; drive only with the body still.
- Any torso change moves the shoulders and the camera: re-localize after it.
