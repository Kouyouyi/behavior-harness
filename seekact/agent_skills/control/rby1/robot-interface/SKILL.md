---
name: robot-interface
description: Map of the Rainbow RB-Y1's MCP tools (sensing, joint-space, task-space, gripper, base), its frames, the limits the backend enforces and what a refusal or a handled stop means. Read first, before any robot action or robot code.
---

# Rainbow RB-Y1: robot interface

Real robot, people nearby, a person at the e-stop. Body: differential-drive wheeled base
(forward / backward and turning in place only), 6-joint torso, two 7-DoF arms with
two-finger grippers and wrist force-torque sensors, and a 2-joint head (pan, tilt) with
a ZED 2i stereo camera. You reach the robot only through these tools; the backend behind
them checks every command and cannot be changed from the workspace.

## Frames

- `base`: on the chassis, x forward, y left, z up (metres, radians, quaternions
  `[x, y, z, w]`). It moves with the robot: after `rby1_base_move` every position you
  measured before is stale.
- End effectors `ee_right` / `ee_left`: the gripper base. The fingers point along the
  end effector's −z (in the URDF, pads 0.073 m beyond it) and close along its x. This robot
  carries **custom round fingers that are longer than the URDF's**: use the measured
  values in [Custom fingers](#custom-fingers-measured), not the URDF offset. With all joints at
  zero (arms hanging) its axes are parallel to `base`. Every motion reply and
  `{{ robot.status }}` give `right_ee` / `left_ee` as `{position, quaternion, rpy_deg}`.
- Head camera: optical (z forward, x right, y down), from the head joints and a hand-eye
  calibration. Each saved frame's `meta.json` has `K` and `T_base_cam` for that frame.
- `odometry` (in every reply): the base pose `{x, y, yaw}` from wheel odometry since the
  robot started; it drifts and slips.


## Custom fingers (measured)

Both grippers carry hand-made black, curved fingers with rounded tips. They are in
neither the URDF nor `rby1_fk`. These values come from earlier sessions (box and bucket
pick-and-place, 2026-10-08/09), checked by projecting them into head-camera images and by
wrist-force contact:

| | Along the end effector's −z from `ee_*` | Notes |
|---|---|---|
| URDF pads (not these fingers) | 0.073 m | do not plan with this |
| Fingers closed (width ~0, ~0.5 A) | ball centre 0.195 m | the two tips form a "ball" of radius ~0.03 m, so they reach ~0.225 m; a pushed box face first touched ~0.05 m from the ball centre |
| Fingers open | middle of the round tips ~0.20 m | the tips reach about 5 cm further than the URDF predicts: open fingers touched a box edge 5 cm above the URDF-predicted height |

- Plan clearance and contact with these lengths. Planning with the URDF length (0.09 m)
  drove the fingertips into a chair bar (45 N on the wrist, `left_arm_4` at 159 %).
- Closed balls squeeze box-like objects between two hands. Open fingers pinch thin walls
  (a bucket rim, 3–11 mm) about 0.20 m along −z.
- The ball is ~0.35 m from the wrist centre: each newton of squeeze puts ~0.3 N·m on
  `*_arm_5`.
- Check once per session that the fingers in the image still look like this. If they
  differ, measure them again and record it in `memory/TIPS.md`.

## Tools by level

| Level | Tool | Use it for |
|---|---|---|
| Sense (no arming) | `{{ robot.status }}` | joints, end-effector poses, wrist forces, odometry, limits, EMO, arming, power / servo / control manager, gripper openings, `sensors` |
| | `{{ robot.observe }}` | the head image now; saves RGB, depth (rgbd), `meta.json` → `frame_id` |
| | `{{ robot.pixel }}` | base-frame 3-D points from pixels of a saved frame |
| | `{{ robot.stream }}` | frames and robot states from the last 10 min ([live-stream](../live-stream/SKILL.md)) |
| Plan (no arming, nothing moves) | `rby1_fk` | hand, camera and link poses, collisions, limits, joint load and centre of mass for the current or a hypothetical posture (`joints`: `{component: {positions \| preset, relative?}}`) |
| | `rby1_check` | dry run of any motion tool call from the current state: accepted or the refusal reason, joint goals, predicted hand poses, planned time, and what else stands in the way (`ready_to_move`: arming, EMO, controller) |
| Task space (arming) | `rby1_look_at` | aim the head camera at a base-frame point |
| | `{{ robot.move }}` | one end effector to a pose, ≤ 0.25 m per call, IK on that arm |
| | `rby1_base_move` | drive: straight `forward`, then turn in place `turn` ([base-navigation](../base-navigation/SKILL.md)) |
| Joint space (arming) | `{{ robot.move_joints }}` | one of `torso` (6), `right_arm`, `left_arm` (7), `head` (2) |
| | `rby1_move_body` | several of them in one command, finishing together ([whole-body-control](../whole-body-control/SKILL.md)) |
| Gripper (arming) | `{{ robot.gripper }}` | open, close (force-limited) or set an opening ([grasping](../grasping/SKILL.md)) |
| Read the controller (no arming) | `rby1_controller_info` | robot model and firmware, control-manager state, PID gains and parameters (read only), fault logs, the controller log |
| Controller modes (the robot's own solvers) | `rby1_cartesian`, `rby1_optimal_control` (whole-body IK), `rby1_joint_impedance`, `rby1_joint_group`, `rby1_jog` | the robot controller's own command modes, measured on this robot; see "The controller's own command modes" below |
| Always allowed | `rby1_hold` | stop the base, hold every joint and gripper (`mode: stop_command` also sends the controller's stop command) |

Plan before you move: these three are cheap (milliseconds) and are the backend's own
model and checks, so a plan that passes `rby1_check` passes the motion tool's planning
(arming, the EMO and the live guards still apply when it runs). Use `rby1_fk` (with a
hypothetical `joints` posture) and `rby1_check` of `{{ robot.move }}` to choose where to
stand and which torso height before driving or squatting, and `rby1_check` a large or
unfamiliar move before sending it. Scripts can use them the
same way through `ctx.value(...)`.

Choose the highest level that expresses the intent: `rby1_look_at` rather than head
joints, `{{ robot.move }}` for a hand position. Joint-space tools are for postures
(presets, torso, retracting) and small corrections. No motion planner is wired in: long
arm motions are a sequence of checked short moves through clear intermediate postures.

## Which tool for which motion (measured on this robot)

| Kind of motion | Prefer | Because |
|---|---|---|
| Hand to a point, one arm | `{{ robot.move }}` (now the controller's Cartesian command) or `rby1_cartesian` | ends within 0.1–0.2 mm; the hand moves along the straight line (within 1 mm); 5 cm in 2.4 s |
| Hand to a point the arm alone cannot reach well (low, far), or both hands together | `rby1_optimal_control` with component `body` | the controller moves torso and arms together; 4–5 cm targets end within 2–3.3 mm in 3.8 s, 10 cm within 0.2 mm in 7.7 s; paths within 1 mm of straight; the torso stays upright |
| Hand 10–25 cm, near objects | several moves of ≤ 5–10 cm with a look between | short moves keep the straight path verifiable from images |
| Posture change, retract, presets | `{{ robot.move_joints }}` / `rby1_move_body` | joints end within 0.0003 rad; predictable joint-space path; tested postures and what they reach: [whole-body-control](../whole-body-control/SKILL.md) "Posture library" |
| Torso height or lean | `rby1_move_body` with `torso` (and `head`) | torso ends within 0.0001 rad; `ready` → `mid` → `low` (0.26 rad each) takes 6.2 s per step |
| Pointing the camera | `rby1_look_at` | within 1° of the target direction, about 2–3 s |
| Changing where the robot stands | `rby1_base_move`, arms in, then observe | wheel odometry agreed with the camera within 1–4 cm over 0.3 m and turns to 0.5 rad within 0.01 rad, but every earlier base-frame position is stale |
| Holding an object | `{{ robot.gripper }}` close with a current limit | the fingers stop on the object and squeeze; close / open take 1.5 s |
| Anything unexpected | `rby1_hold`, then observe | a stop cancels within 0.1 s; the joints coast 0.02–0.04 rad and then hold |

Time budget: every motion call takes its planned time (largest joint change ÷ that part's
speed cap plus one ramp time, speed ÷ acceleration: 0.5 s for arms and torso, at least
0.5 s in all) plus about 0.8 s; `{{ robot.status }}` takes 10 ms and
`{{ robot.observe }}` 60 ms. Fewer, larger joint moves are faster than many tiny ones.

## What the backend enforces

Every limit below is checked before anything moves; a refusal names the limit. A refusal
is a wall, not a fault: change the approach, do not retry the same command.

| Rule | Value |
|---|---|
| Speed | arms 0.2 rad/s, torso 0.1 rad/s, head 0.3 rad/s, each with a gentle acceleration ramp (arms 0.4, torso 0.2 rad/s²); base 0.15 m/s, 0.3 rad/s |
| Per call | ≤ 1.2 rad on any joint; ≤ 0.25 m end-effector travel; ≤ 1 m base travel, ≤ π rad turn |
| Joint limits | URDF limits minus 0.1 rad (head 0.05) (in `{{ robot.status }}` → `limits`); real end stops can sit inside them, so keep routine targets 0.2 rad further in |
| Torso | upright rule: `torso_0 = torso_4 = 0` and `torso_2 = −2 · torso_1`; squat at most `torso_1 = 1.2` rad (`lowest`) (see whole-body-control) |
| Joint load | the gravity load on every joint along the path, from the vendor model × 1.5, stays below 130 % of its rated torque (the scaled model reads about 1.3 × the measured torso load, so this is about 100 % real) |
| Drive temperature | no motion while a drive is above 65 °C |
| Reach | target ≤ 0.69 m from that arm's shoulder (`link_*_arm_0`), else `out of reach` |
| Driving posture | `rby1_base_move` refuses with torso_1 > 0.8 (below `mid`), a chest lean (torso_3 − torso_1) over 0.05 rad, or a hand more than 0.72 m forward or out to the side |
| Height | both end effectors stay above z = 0.10 m along every arm or torso path |
| Self-collision | every joint path is checked against the vendor collision model; a path that makes a new contact is refused (moving out of an existing contact is allowed) |
| Contact | a wrist force rise above 60 N during an arm or torso motion stops it |
| Torque | a joint torque above 150 % of its rating stops the motion at once (also between calls). A joint already above it when a motion starts stops it only if it rises another 10 % of its rating, so moves that hold or ease a load are never locked out. Holding or carrying puts the wrists and elbow near 100 %; that is allowed, but a load that keeps a joint above its rating heats the drive (no motion above 65 °C), so carry close to the body and set loads down promptly |
| Blocked joint | an arm or torso joint still more than 0.004 rad from its target after the move is re-targeted where it is; the reply has `settled: false`, `blocked_rad` and a `warning` |
| Faults | a controller fault, a drive not ready or an EMO disarms the session; the fault is reset only after the operator arms again |

While a motion runs it is watched every 20 ms. Disarming, an EMO, joint state older than
0.2 s, a joint that makes no progress for 2 s, a wrist force spike, or the controller
ending the command early cancels it; the controller stops the joints 0.02–0.04 rad further
along and holds them there (measured). A joint torque over 150 % of its rating or a drive that is
not ready stops it too; after a torque stop the joints are re-targeted where they stopped.
`{{ robot.status }}` → `safety` shows the current fault (if any), the last safety events,
the most loaded joints and the hottest drive. The head pan settles about 0.015 rad (1°) short of
its target; everything else within 0.002 rad. The first armed motion of a
session also powers the drives on (`status` → `enable_steps`); they then hold where they
are.

## The controller's own command modes

These tools hand their targets to the robot controller's own solvers (one SDK command
builder each); **the controller solves the IK and the joint path**, not the backend. That
changes what can be checked:

- **No Cartesian arm move from a near-singular arm.** With an elbow almost straight
  (`*_arm_3` near 0), the controller faulted (MinorFault, session disarmed) on a 5 cm
  `rby1_cartesian` descent. The backend now refuses such a move (smallest arm-Jacobian
  singular value below 0.04; the faulting pose had 0.016, `ready` has 0.13). Bend the
  elbow with a joint move first, or use `rby1_move_eef` / `rby1_move_joints`.
- Before the command: only its **targets** (travel ≤ 0.25 m per link per call, torso
  link ≤ 0.1 m, rotation ≤ 1 rad, hands above z = 0.10 m, an arm alone within 0.69 m of
  its shoulder, joint targets inside the limits and the squat cap). `rby1_check` of these
  tools reports `checked: targets only`; it cannot predict the joints.
- While it runs, every 20 ms: arming, EMO, fresh state, drives ready, torque over 150 % of a
  rating, wrist force, and an envelope on the measured joints: a joint closing on its
  limit, the squat cap, the torso leaving upright (torso_0 / torso_4 beyond 0.05 rad), a
  hand below 0.10 m, a new self-collision. Any of these cancels the command and holds.
- It always ends with a joint-position hold at the measured pose, so nothing keeps
  pushing afterwards.
- You choose the mode, the targets and (Cartesian) a speed up to 0.1 m/s / 0.3 rad/s. You
  cannot set gains, stiffness, torque limits or solver settings: they are fixed by the
  operator's configuration, and unknown argument keys are refused.

Measured on this robot (2026-10-08, `scripts/rby1_controller_check.py`, from `ready`):

| Tool | SDK builder | Parts | What it does | Measured |
|---|---|---|---|---|
| `rby1_cartesian` | `CartesianCommandBuilder` | `right_arm` / `left_arm` (that arm), `torso`, `body` (torso + both arms) | drives `ee_right`, `ee_left` and/or `link_torso_5` to base-frame poses (`quaternion` omitted: keep orientation; `relative`); optional `joint_targets` pin joints; the upright rule is added as torso_0 = torso_4 = 0 targets for `torso` / `body` | 5 cm in 2.4 s, final error 0.1–0.2 mm, path within 0.4 mm of the straight line (arm) and 1 mm (body, torso joined by 0.06 rad); a 0.3 rad turn of the hand exact |
| `rby1_optimal_control` | `OptimalControlCommandBuilder` | `body`, `torso` | **whole-body IK**: one weighted problem over the torso and both arms: link pose targets (`translation_weight`, `rotation_weight`, 0–10), joint targets with weights, an optional `center_of_mass` target (base frame); the controller stops when the cost settles | right hand +5 cm: 3.8 s, 3.3 mm; left +5 cm up: 3.1 mm; both hands +4 cm: 2.3 / 2.4 mm; right hand 10 cm down: 7.7 s, 0.2 mm, torso 0.11 rad + arm 0.16 rad; paths within 1 mm of straight; torso roll ≤ 0.0003 rad |
| `{{ robot.move }}` | `CartesianCommandBuilder` (one arm) | one arm | as before, now solved by the controller | 3 cm in 2.4 s, 0.1–0.2 mm, within 0.2 mm of straight |
| `rby1_joint_impedance` | `JointImpedanceControlCommandBuilder` | one arm | a joint-space move (positions / preset / relative, same limits and path checks as `{{ robot.move_joints }}`) that yields to contact (stiffness 100 N m/rad, torque limit 60 % of rating) | 0.1 rad in 2.2 s, ends about 0.018 rad short: compliant, not exact |
| `rby1_joint_group` | `JointGroupPositionCommandBuilder` | torso | named torso joints only; the result must keep the upright rule and squat cap | 0.1 rad in 2.9 s, 0.1 mrad |
| `rby1_jog` | `JogCommandBuilder` | one joint | `absolute`, `relative` (≤ 0.3 rad) or `one_step`, at that part's speed; path checked like a joint move | wrist 0.1 rad in 2.2 s within 0.3 mrad; head 0.1 rad in 1.9 s within 8 mrad |
| `rby1_hold` `mode: stop_command` | `StopCommandBuilder` | all | the controller's stop in addition to cancelling | 0.2 s; motion works normally afterwards |

Not available, on purpose: Cartesian impedance ended 66 mm from a 3 cm target and link
impedance with the vendor's gains drove the left wrist to 202 % of its rating in 1.3 s (the
torque guard stopped it). Both stay off until the operator sets new gains and they are
measured again. Gravity compensation and every robot setting are never available.

How to choose: `rby1_cartesian` (or `{{ robot.move }}`) for accurate straight hand moves of
one arm; `rby1_optimal_control` (`body`) when the arm alone is short, for two-hand moves,
or when you want the torso to help (give a `center_of_mass` target to keep balance when
leaning); joint-space tools for postures and presets. Keep each controller-mode call to a
few centimetres near objects and observe between calls: the controller picks the joints,
so check the measured result in the reply.

## Reading errors

A failed tool returns its whole error: `{type, operation, stage, reason}`, plus
`robot_state` (measured, after the robot was held) when it failed during a motion.
`reason` is the complete message; for an unexpected error (an SDK or driver exception)
it ends with the innermost traceback frames.

| You see | Meaning | Do |
|---|---|---|
| `stage: validation` | refused before anything moved | fix the cause named in `reason` |
| `stage: robot_execution`, a backend reason (a guard, a disarm, a stop) with `robot_state` | handled stop: cancelled and holding at the measured `robot_state` | observe, find the cause, plan from `robot_state`; never resend the same target unchanged |
| `stage: robot_execution`, an SDK / driver message with a traceback, and `robot_state` | the robot's software failed during the motion; the backend then held the robot and measured it | read the message, `{{ robot.status }}` and `{{ robot.observe }}`, then continue from `robot_state`; tell the operator if it repeats |

## Arming

Motion tools refuse with `motion is not armed` until the operator arms the session. Ask
with `request_operator` (kind `help`), say what you will do, and wait for the answer.
Never try to work around arming or the e-stop.

## Safety rules

- Why the guards exist: on this robot a torso squat to `torso_1 = 1.28` put torso_2 on a
  real end stop inside its model limit; the drive kept pushing towards the unreachable
  target at 2.6 × its rated torque until it tripped, and every joint dropped (the arms
  fell). Never push: after `blocked_rad`, a torque stop or a force stop, back off and
  replan; never resend the same target.
- After a fault or an EMO: no motion and no recovery plan of your own. `request_operator`
  (kind `help`) with what happened, wait, and after the operator re-arms start with
  `{{ robot.status }}` and `{{ robot.observe }}`.
- Loads: `{{ robot.status }}` → `safety.highest_torque_load`. Above about 100 %, bring the
  arms or the object closer to the body (or set it down) before anything else; above 150 %
  motions stop.
- Tip-over: no base motion with an arm extended, an object held out or the torso low or
  leaning; both arms `ready` (or `park`) and torso `ready` or `zero` before driving.
- The torso is strong (hundreds of Nm): change it in steps and look after each; nothing
  and nobody between the arms and the body.
- Before an arm motion look at a fresh image of the space it sweeps (people, objects,
  the robot's own body and other arm); before driving look along the path.

## From code

`api.runtime.Context` reaches the same tools (`MCP_SESSION.md`), e.g.
`ctx.value('rby1_move_eef', arm='right', position=[x, y, z])`. Calls are serialized with
yours; check every reply and stop on the first unexpected one.
