# The RB-Y1 backend

The backend for the Rainbow Robotics RB-Y1 (model A): its tools, safety checks, the design
rules behind them, and what was measured on the robot. Setup lessons and the system map:
[docs/rby1/SETUP_NOTES.md](docs/rby1/SETUP_NOTES.md). Every tool below was characterized
on the robot, and the agent sandbox was audited; the bring-up order is the checklist below.

## How the robot is reached

```
agent (Claude Code, sandboxed) ── MCP ──> frontend (services.controller.frontend)
                                            └─ Rby1Backend (services/rby1/backend.py)
                                                 ├─ SdkRobot (robot.py) ── rby1_sdk gRPC ──> controller 192.168.30.1:50051
                                                 ├─ Gripper (gripper.py) ── rby1_sdk DynamixelBus ── /dev/rby1_gripper
                                                 ├─ ZedCamera (camera.py) ── pyzed ── head ZED 2i (USB)
                                                 └─ Recorder (recorder.py) ──> <session>/stream (last 10 min)
                                                      └─ Archiver (services/archive.py) ──> <session>/archive (whole session)
```

- The robot service runs on the robot's UPC (a Jetson, 192.168.30.2 on the robot network).
  The joint and wheel controllers run on the robot's controller at 192.168.30.1 (gRPC
  :50051; web UI :5173).
- `rby1-sdk` **0.8.3** matches the controller, which reports SDK commit `f3bb123`
  (`v0.8.3-3`). Keep the pair: upgrade the SDK only with the robot firmware.
- Robot service environment (`RAX_PYTHON`): Python 3.10 with `rby1-sdk==0.8.3`, numpy,
  scipy, opencv-python-headless, jinja2, tomli, pillow, pytest, imageio-ffmpeg (bundled
  ffmpeg for the stream archive), and pyzed 5.2.3 from the ZED SDK's own wheel.
- **Connecting changes nothing.** `create_robot_a(...).connect()` only opens the gRPC
  channel. Power, servo and the control manager (`unlimited_mode` off) are checked before
  every armed motion and brought up if off (`status` → `enable_steps`), after which the
  drives hold where they are. An emergency stop seen by the backend disarms the session:
  after releasing it the operator arms again on purpose. No robot parameter is written:
  helpers that persist acceleration scaling and cutoff frequencies with
  `set_parameter(..., write_db=True)` are not used.
- Kinematics, IK and self-collision come from the robot's own URDF
  (`get_robot_model()`) through `rby1_sdk.dynamics`; `services/rby1/model/rby1a.urdf`
  is the vendor model A for offline use. The head camera pose is `link_head_2` composed
  with a hand-eye calibration: the backend option `head_camera_extrinsic`, by default the
  one measured on our unit (`X_HEAD_CAMERA` in `services/rby1/kinematics.py`). Calibrate
  your own robot's head camera.

## Agent tools (backend `rby1`)

| tool | does |
|---|---|
| `rby1_status` | joints (torso 6, arms 7 + 7, head 2), end-effector poses (FK), wrist forces, wheel odometry, EMO, battery, power / servo / control manager, gripper openings, arming, limits |
| `rby1_observe` | head ZED 2i RGB (+ depth in rgbd sessions); saves the full-resolution frame, `K` and `T_base_cam` under `runtime/frames/<id>/` |
| `rby1_pixel_to_position` | head-camera pixels + depth to base-frame points |
| `rby1_move_joints` | one component to positions or a preset (`zero`, `ready`, `park`) as one timed joint-position command |
| `rby1_move_body` | several components in one command, finishing together |
| `rby1_move_eef` | one arm's `ee_*` to a base-frame pose: IK on the vendor model, then a joint move (path checked, not a straight line) |
| `rby1_look_at` | head pan / tilt to point the camera at a base-frame point |
| `rby1_base_move` | straight `forward`, then in-place `turn`, closed loop on wheel odometry (an SE(2) velocity stream) |
| `rby1_gripper` | open / close (current-limited) / width; the first call homes both grippers |
| `rby1_stream` | look back in the live recording (head camera ~10 Hz, depth 1 Hz, state 10 Hz, last 10 min) |
| `rby1_fk` | read-only: hand, camera and link poses, collisions, limits, joint load and centre of mass for the current or a hypothetical posture |
| `rby1_controller_info` | read-only controller queries: robot info, control manager (state, unlimited mode, time scale), PID gains and parameters (read), fault logs, the controller log |
| `rby1_check` | read-only dry run of a motion tool call: the same planning checks (`Rby1Backend._plan`, shared with `validate`), accepted or the reason, goals, predicted hands, planned time |
| `rby1_hold` | stop the base and hold every joint and gripper; `mode: stop_command` also sends the controller's StopCommand; always allowed |

The controller's own command modes (`services/rby1/controller_modes.py`), on by default since
their characterization on 2026-10-08 (backend option `controller_commands`; with it on,
`rby1_move_eef` uses the controller's Cartesian command instead of the backend's arm IK).
Measured with `scripts/rby1_controller_check.py` from `ready` (raw:
`runtime/characterization/controller_check_*.json`): Cartesian 5 cm in 2.4 s, 0.1-0.2 mm, path
within 0.4 mm (arm) / 1 mm (body) of straight; optimal control (whole-body IK) 4-5 cm in 3.8 s
within 2-3.3 mm, 10 cm down in 7.7 s within 0.2 mm with the torso taking 0.11 rad, torso roll
<= 0.0003 rad; joint impedance 0.1 rad in 2.2 s ending 0.018 rad short (compliant); jog and
joint group within 0.1-0.3 mrad (head 8 mrad); stop command 0.2 s. Disabled
(`controller_tools_disabled`): Cartesian impedance (66 mm from a 3 cm target) and link impedance
(vendor gains drove left_arm_5 to 202 % of its rating in 1.3 s; the torque guard stopped it).

| tool | SDK builder (used as shipped) | components |
|---|---|---|
| `rby1_cartesian` | `CartesianCommandBuilder`: link poses (`ee_right`, `ee_left`, `link_torso_5`), the controller solves IK and path | an arm, torso, body |
| `rby1_optimal_control` | `OptimalControlCommandBuilder`: weighted link / joint / centre-of-mass targets, whole-body IK on the controller | body, torso |
| `rby1_cartesian_impedance` | `CartesianImpedanceControlCommandBuilder`: compliant hand pose | an arm |
| `rby1_link_impedance` | `ImpedanceControlCommandBuilder`: hand held like a spring for `seconds` | an arm |
| `rby1_joint_impedance` | `JointImpedanceControlCommandBuilder`: compliant joint move | an arm |
| `rby1_joint_group` | `JointGroupPositionCommandBuilder`: named torso joints | torso |
| `rby1_jog` | `JogCommandBuilder`: one joint, absolute / relative / one step | any body or head joint |

Not wired, on purpose: `GravityCompensationCommandBuilder` and every setter of the robot
(see "The agent uses the SDK; it never modifies the robot" below). Gains and solver
settings of these modes are `DEFAULTS` from the vendor examples (stiffness 100 N m/rad,
damping ratio 1, torque limit 60 % of rating, optimal-control stop cost 2e-3, patience 10,
speed scaling 0.2; Cartesian at most 0.1 m/s, 0.3 rad/s). Before a command the backend
checks its targets (travel, rotation, hand height, arm reach, joint targets in limits, the
upright rule added as torso_0 = torso_4 = 0 targets); while it runs, every control period,
arming, EMO, fresh state, drives, torque, wrist force and an envelope on the measured joints
(approaching a limit, the squat cap, the upright torso, hand height, new self-collisions);
it ends with a joint hold at the measured pose. `rby1_check` covers these tools on targets
only, since the controller picks the joints.

Not wired in yet: a motion planner, and lidar odometry (for example the robot's Livox
MID-360 with DLIO), which would replace wheel odometry for the base.

Safety (outermost first): the EMO buttons and the operator; operator arming
(`rby1 arm` / `disarm`), re-read **every control tick** (50 Hz), with the EMO state
polled each tick; backend limits (URDF joint limits minus 0.1 rad (head 0.05), torso squat
torso_1 ≤ 1.2 rad, ≤ 1.2 rad per joint per call, arms 0.2 rad/s, torso 0.1 rad/s, head
0.3 rad/s with acceleration ramps, the torso-upright rule, end effectors above
z = 0.10 m, self-collision and the gravity load (≤ 130 % of rating, model × 1.5; 75 % until 2026-10-08) along the
path, a 60 N wrist force guard, a joint torque guard (over 150 % of the rating, at once; raised from 100 % and 40 N
on 2026-10-08 at the operator's request, joint limits unchanged; a joint already over it when a motion starts stops
the motion only if it rises another 10 % of its rating, after a squeezed box locked every move in `pnp_box`), base ≤ 0.15 m/s, 0.3 rad/s,
≤ 1 m per call); in-place re-targeting of a joint left short of its target; a 10 Hz
watchdog between calls; controller faults disarm and are reset only after the operator
re-arms; the controller's own limits. See "Fault of 2026-10-08" below.
All are `backend_options` (see `DEFAULTS` in `services/rby1/backend.py`).

## The agent uses the SDK; it never modifies the robot

Every robot tool maps its arguments onto public `rby1_sdk` calls used as shipped
(`services/rby1/robot.py` is the only module that sends commands). The backend writes no
solver, planner or controller of its own in their place: the SDK's whole-body IK is the
controller's `OptimalControlCommandBuilder`, its Cartesian IK the `CartesianCommandBuilder`.
The agent decides *what* the robot does (mode, targets, a speed below a cap), never *how
the controller behaves*:

- No tool writes or tunes the robot: no `set_parameter`, resets or factory resets, PID
  setters, power, servo, brakes, control-manager enable / disable / fault reset, time
  scale, Wi-Fi, system time, LEDs, odometry reset, home offsets or robot-model import.
- No controller-behaviour switches or gains as agent arguments: gravity compensation is
  not wired; impedance stiffness, damping and torque limits, and solver settings are
  backend options from the vendor examples, changed only by the operator through
  `backend_options`.
- Reading is allowed (`rby1_controller_info`: robot info, control-manager state,
  parameters and PID gains read-only, fault logs, the controller log).
- Unknown argument keys are refused, never ignored, so an agent cannot believe it tuned
  something.

## Use

See [README.md](README.md) for setup. In short:

```bash
ln -s $PWD/scripts/rby1 ~/.local/bin/rby1
"$RAX_PYTHON" scripts/configure_session.py --task tidy --task-file task.md   # --robot rby1 is the default
bash scripts/start_agent.sh --config <session>/operator-input.json
rby1 arm 30           # person at the e-stop; arms the newest session for 30 min
rby1 disarm           # any time: running motions stop and hold
```

Before a session, stop everything else that commands the robot or owns its devices on the
UPC: any other robot server, a gamepad teleop that writes straight to :50051 (a second
writer defeats the backend's checks), and any ZED publisher (the camera opens in one
process only). Read-only processes (lidar odometry, the dashboard telemetry) may stay.

## Bring-up checklist (from docs/REAL_ROBOT_ADAPTATION.md)

| # | Step | Where |
|---|---|---|
| 1 | Map the system | [docs/rby1/SETUP_NOTES.md](docs/rby1/SETUP_NOTES.md) |
| 2 | SDK versions; robot-service env; agent env without SDKs | `RAX_PYTHON`, `RAX_AGENT_PYTHON` (README) |
| 3 | Backend, registry, ROBOT_FACTS, telemetry, `scripts/rby1` | `services/rby1/`, `services/backends.py`, `services/workspace/setup.py` |
| 4 | Connect without moving | `scripts/rby1_motion_check.py` steps `status`, `disarmed` |
| 5 | Unit tests with fakes | `local_tests/unit/test_rby1_backend.py` |
| 6 | Supervised characterization | below |
| 7 | Observation (K, `T_base_cam`, depth, stream rates) | the floor plane checks `T_base_cam` + depth to 1.5 cm at 2.7 m (below) |
| 8 | Planner | none wired |
| 9 | Skills from measured numbers | robot-interface carries the selection table |
| 10 | Sandbox audit | run by an agent session: 192.168.30.1:50051 / :5173 refused (direct: network unreachable; proxy: 403 deny list), no `rby1_sdk` in the agent env, `/dev/rby1_gripper` and the ttyUSB devices absent in the sandbox, protected paths refused by the permission layer. Internet through the sandbox proxy timed out (PyPI) on that run |
| 11 | Supervised agent sessions with a person at the e-stop | the fault below came from one |

Settled on the robot: a cancelled command stops with the controller's deceleration and
the joints hold (below). Still open: what `unlimited_mode` relaxes (the backend keeps it
off), whether servo-on settles the joints (the robot was already servoed on), and the
odometry origin.

## Motion check and characterization (2026-10-07)

Run on the robot through the real backend (the agent's guards), operator at the e-stop:
`scripts/rby1_motion_check.py` (every step) and `scripts/rby1_characterize.py` (all
parts; raw JSON in `runtime/characterization/`). Controller version 0.8.0, rby1-sdk 0.8.3,
model A v1.0. Speeds are the backend defaults: arms 0.3 rad/s, torso 0.15 rad/s, head 0.5
rad/s; base capped at 0.1 m/s / 0.2 rad/s here.

| Tool / behaviour | Measured |
|---|---|
| `rby1_status` / `rby1_observe` call time | 8 ms / 63 ms |
| Joint move duration | planned time (step / speed cap, at least 0.5 s) + about 1.3 s with the controller's 1.0 s hold time; with 0.2 s (now the default) 1.48 s instead of 2.27 s for 0.2 rad, same error. Elbow / wrist 1.0 rad: 4.9 s (with 1.0 s hold) |
| Joint final error | arms and torso ≤ 0.0002 rad; head tilt ≤ 0.0015 rad; head pan 0.012–0.017 rad (settles about 1° short) |
| Whole body, rest → `half` → `ready` | 4.9 s each (3.5 s planned), ≤ 0.008 rad |
| Torso height (upright rule) 0.26 rad | 4.9 s, 0.0001 rad |
| `rby1_move_eef` final error | 0.2–1.8 mm (2–20 cm moves) |
| `rby1_move_eef` repeatability | 5 runs to the same +5 cm pose: spread 0.09 mm, mean error 0.65 mm |
| `rby1_move_eef` path vs. straight line | ≤ 3.4 mm (2 cm), ≤ 6.7 mm (5 cm), ≤ 8.4 mm (10 cm), 13–26 mm (20 cm) |
| `rby1_look_at` | 0.05–0.83° from the target direction, 1.8–3 s |
| Gripper | homing 8.4 s (first call); close / open 1.55 s, to < 1 mm / 0.1 m at 0.3 and 0.8 A |
| Camera | ZED 2i HD720, fx = fy = 532.5; depth 91–97 % valid; the floor 2.7 m ahead measured at z = −0.012…−0.016 m (truth 0) and repeatable to 5 mm; stream 8 fps (10 target), depth 1 Hz |
| Base 0.3 m | odometry 0.295 / −0.297 m; the camera (wall below the window) 0.333 / −0.309 m; 3.7 s at 0.1 m/s |
| Base turn 0.5 rad | odometry 0.493 rad; `base` swings 0.108 m sideways (it is 0.228 m behind the wheel axle), 3.2 s |
| Base 5 cm / 0.1 rad | 4.55 cm / 0.094 rad (arrival tolerance 5 mm / 0.01 rad) |
| `rby1 disarm` mid-motion | cancelled 0.08 s after the call; the controller decelerates 0.020 rad (head, 0.1 rad/s) / 0.040 rad (elbow, 0.3 rad/s) further, then holds: drift 0.0001 rad over 3 s |
| Refusals | every limit refuses before moving with its value in the text (`runtime/characterization/*_refusals.json`) |

Findings that changed the backend: the arms' rest pose sits on joints 1 and 3's limits (no
arm `zero` preset; a `half` preset); gripper homing must tolerate the encoders' one-count
jitter; base arrival tolerances 2 cm / 0.03 rad stopped 5 cm moves at 3.1 cm; the
controller's hold time 1.0 s cost 0.8 s per move for nothing; a hold command sent after a
cancel is cancelled with it (the SDK cancel cancels the current control) and would pull the
joints back if accepted, so stops only cancel and the controller holds.

## Fault of 2026-10-08 and the guards it added

An agent (`pnp_box`) squatted the torso to `[0, 1.28, −2.56, 1.28, 0, 0]` with the arms in
`ready`. The controller's fault log (`get_fault_log_list` + `download_file`) shows
torso_2 stopping at −2.554 rad against a real end stop (URDF limit −2.618), 6 mrad short of
its target; its drive then wound up to −697 N m / 46 A (2.6 × the 270 N m rating) for
1.2 s, tripped (state 516), the control manager went to MajorFault and every joint
dropped. The backend's next call reset the fault and re-enabled the robot on its own.

Fixed in the backend: no fault reset without a new operator arming; the squat limit;
larger limit margins; any arm / torso joint more than 4 mrad short after a move is
re-targeted where it is (`blocked_rad` + `warning` to the agent); a torque guard (over
the rating) during motion and in a 10 Hz watchdog between calls; a gravity-load check along the path;
slower speeds with ramps. In the agent's AGENTS.md (`robot.safety` in ROBOT_FACTS) and
skills: report faults and wait, stay 0.2 rad inside the limits, never push, keep loads
light, short slow steps.

`scripts/rby1_health_check.py` on the robot afterwards (raw:
`runtime/characterization/health_20261007T234227.json`): every arm and torso joint ±0.1 rad
and back ≤ 0.0003 rad error; peak velocity at the caps (arms ≈ 0.2, torso 0.10 rad/s,
torso ramp 0.2 rad/s²); `ready` from the tucked pose 6.2 s; drives ≤ 48 °C. Guards on the
hardware: the torque guard stopped a shoulder lift and the in-place re-target moved
0.1 mrad; the watchdog re-target moved 0.0 mrad; the blocked re-target after a finished
command moved 0.1 mrad. Holding torques read 1.1–1.6 × the vendor model; the right wrist
(right_arm_5) holds about 5 N m (50 % of 10) at `ready`, 2.4 × the model, so a payload in
the right hand nears its guard first.

## Large motions (2026-10-08)

`scripts/rby1_large_motion.py` (squat, stretch, reach, low; raw:
`runtime/characterization/large_motion_*.json`), after a head-camera clearance check (depth:
nothing within 1.1 m). Results, now the posture library in the whole-body-control skill and
the presets `mid` / `low` (torso) and `forward` / `forward_up` / `overhead` / `forward_out` /
`side` (arms):
- Squat `ready` → 45° → 60° (cap) and back: 6.2 s per 0.26 rad step, 0.1 mrad error,
  torso_2 ≤ 54 %; hands 1.29 → 1.18 → 1.04 m. Lean +0.15 fine; +0.30 refused by the load
  check (torso_3 76 %). Chest yaw ±0.4 swings the hands about 0.25 m.
- Arm stretches (each arm, then both forward): every posture ≤ 0.3 mrad; hand reach
  0.69 m forward, 1.91 m up, 0.81 m sideways at 60°. Raising the left arm 90° sideways hit
  the torque guard (left_arm_1 57 N m, 81 % of 70; right peaked 73 %): the arm stopped at
  77°, was re-targeted in place and came back down; 60° is the practical limit.
- Hand paths with `rby1_move_eef`: a 0.2 m square ends within 1.9 mm per corner with the
  path 14–18 mm off the straight line; a 0.15 m reach forward within 1.1 mm, 10 mm off.
- Table-height reach: torso `low` + arm forward-down puts the hand at z = 0.75 m, 0.62 m
  ahead; 0.1 m lower is out of reach (0.73 m from the shoulder).
- Deeper squat (squat cap raised from 1.05 to 1.2): torso_1 1.10 / 1.15 / 1.20 in steps,
  arms `ready`: each settled within 0.1 mrad, torso_2 measured / modelled torque 1.11 / 1.09 /
  1.13 (no contact signature), a 5 s hold at 1.2 drifted 117 → 126 N m (47 %) with no
  position change; hands at 0.94 m, and 0.65 m with the arm reaching down. 1.2 leaves
  0.15 rad on torso_2 to the stop measured at −2.554; torso preset `lowest`.
- The right wrist (right_arm_5, 10 N m) peaks at 75–82 % during ordinary moves while
  holding 13–60 % depending on the move before (force built up against friction; an
  in-place re-target released it to 1.3 N m). The first version of the torque guard
  (instant at 80 %) stopped a 0.2 m hand move on it; the guard is now 100 % of the rating
  at once, which still stops the fault mode (260 % for 1.2 s) within one tick.

## Leaning (2026-10-08)

`scripts/rby1_large_motion.py lean` (raw: `runtime/characterization/lean_*.json`): at each
torso height (`zero`, `ready`, `mid`, `low`, `lowest`) leans of +0.10 and +0.15 rad with the
arms at `ready` settled within 0.1 mrad in 1.5–2 s; +0.15 lowers the hands 8 cm and moves
them 5 cm forward (down to z = 0.85 m at `lowest`). torso_3 held 49–56 % of its rating (the
load check's model × 1.5 predicts about 71 %: conservative there). While leaning +0.15 each
hand reached 0.08 m forward and back within 1.4 mm, except with the torso `zero`, where the
load check refused the right-hand reach (torso_3 75 %). With the arms `park`, a +0.3 lean at
`mid` held at 48 %. The whole-body centre of mass (vendor model, 127.6 kg, 60 kg of it in
the base) stays within 0.06 m of the base origin in every posture tested, 0.17 m behind the
wheel axle. `rby1_base_move` now refuses with the torso below `mid`, leaning, or a hand
stretched out.

## The agent's Python and robot model

- Agent environment (`RAX_AGENT_PYTHON`, no robot SDK): numpy 2.2.6, scipy,
  OpenCV, pillow, matplotlib, pandas, scikit-learn, scikit-image, sympy, shapely, trimesh,
  pycollada, open3d 0.20, pinocchio (`pin` 4.1), yourdfpy, transforms3d, networkx, pyyaml,
  tqdm. Read-only in the sandbox; the agent adds packages with `pip install --target .pylib`
  (`.pylib` is on its `PYTHONPATH`). A session's `agent_python` overrides it; the launcher
  warns when neither is set, since the agent's scripts would then run in the robot service's
  interpreter, which has the SDK.
- `runtime/robot-models/rby1a/` (read-only, linked as `agent-workspace/robot-model/`): the
  vendor URDFs and meshes from the rby1-sdk checkout at the controller's commit (`SOURCE`),
  plus `model_v1.0.standard.urdf` / `model.standard.urdf` from
  `scripts/standardize_urdf.py` (capsules -> cylinder + 2 spheres, wheel efforts 0) for
  pinocchio / yourdfpy; pinocchio's FK on it matched rby1_sdk.dynamics on the original to
  1.4e-9 over 20 random postures.
- Internet: every host through the sandbox proxy except the robot controller
  (192.168.30.1).

## Head camera pose check (2026-10-08)

`T_base_cam` = FK(measured joints, base -> link_head_2) composed with the hand-eye
calibration (the built-in `kinematics.X_HEAD_CAMERA`), per frame. Checked without moving on 112 saved
frames (head pan -0.47..+0.46, tilt 0..1.3, torso_1 0..1.29) with a RANSAC floor-plane fit
per frame (0.8-2.5 m): floor 1.5 m ahead at -7 mm median, independent of head and torso
angles (<= 10 mm per rad), fit rms 4.4 mm; per frame 90 % within 46 mm and a plane tilt of
1.6 deg median (90 % < 4.1 deg): ZED depth noise and a small orientation error, which a
hand-eye re-calibration could reduce. The object-localization skill quotes these numbers.

## Skill library (agents' continual learning)

`runtime/skill-library/rby1/` (`services/skill_library.py`) is a git repository of task
skills that agents learn across sessions and tasks, separate from the read-only harness
skills. Each new session's workspace gets a clone in `agent-workspace/skill-library/` on
branch `session/<session>`; the agent reads `INDEX.md`, runs skills (`code/run.py:
main(ctx)`), and commits new or improved skills there. The harness tool
`skill_library_merge` (MCP frontend) checks the branch and merges it into `main`:

- format: `INDEX.md` row per skill; `skills/<slug>/SKILL.md` frontmatter (`name`,
  `description`, `status` draft / verified / deprecated, `robot`, `entry`) and the sections
  When to use, Preconditions, Procedure, Code, Checks, Failure modes, Evidence; an entry
  that compiles and defines `main(ctx, ...)`; `evidence/runs.jsonl` runs with session,
  date, commit, outcome, unattended, conditions, checks, images; at least one image
  (≤ 400 kB); `verified` needs 2 unattended successful runs with images;
- a branch may change only `INDEX.md` and `skills/`, and may not delete a skill
  (deprecate instead);
- the harness never runs git in the agent's clone (whose config and hooks the agent
  controls): it copies the object store without hooks, links or alternates, validates the
  commit's files as data (code is compiled, never run), and merges under a lock; on a
  conflict it aborts and tells the agent to merge `origin/main`.

The agent's protocol is the task skill `continual-learning`; AGENTS.md tells it to read
the library at the start and merge before it ends. Inspect or back up the library with
plain git (`git -C runtime/skill-library/rby1 log --stat`).

## Runtime storage and shipping

- `runtime/` is the artifact root (`RAX_ARTIFACT_ROOT` overrides it); put it on a disk
  with room. Sessions, frames, streams and logs stay there unless shipping is on.
- Every session is archived whole for replay: as each minute of
  the 10-minute live stream completes, it becomes `<session>/archive/head_camera/<UTC
  YYYYmmddTHHMM>.mp4` (H.264 via libx264, 10 fps nominal) with a `.json` sidecar of each frame's
  capture time (epoch ms), plus `archive/robot/<minute>.jsonl` (state at 10 Hz). Frame k was
  captured at `frames_ms[k]`, so the archive lines up with the transcript, `mcp.jsonl` and
  `model-usage.jsonl`. Depth is not archived. Measured: 5 s of encoding and 2.4 MB per minute
  (~145 MB/hour). The minute in progress is archived when the robot service stops; a minute
  not yet archived is kept up to 5 minutes past the window. ffmpeg comes from `imageio-ffmpeg`
  in the robot env when no system ffmpeg exists; option `stream_archive` turns it off.
- Shipping finished runs off the robot (`services/runtime_ship.py`) is off unless
  `RAX_SHIP_DEST=host:/dir` is set; then the launcher ships each run after it ends,
  and `rby1 ship [--dry-run]` does it by hand.

## Verify

```bash
env -u PYTHONPATH "$RAX_PYTHON" -m pytest -q    # simulated robot / mock backend only; never touches hardware
```
