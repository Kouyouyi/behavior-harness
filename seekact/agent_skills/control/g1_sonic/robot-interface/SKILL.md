---
name: robot-interface
description: Observe G1, execute bounded SONIC references or optional VLA tasks, and deploy the verified bilateral ankle-temperature response.
---

# G1 SONIC deployment and control

Use `docs/g1/DEPLOYMENT.md` for the current deployment defaults and commands.
Historical commissioning details belong in `docs/g1/EXPERIMENTS.md`, not in new
session prerequisites. Respect existing user authorization and site confirmations.

The operator's unified desktop entry is `bash scripts/start_g1_experiment.sh`;
`--check` starts nothing, and `--prepare-only` freezes configuration only.
The default launch opens two control terminals and Firefox, reuses verified
background services, and waits for standing before enabling UI dispatch. It does
not prepare VLA or grasp automatically. The UI task agent must never invoke this
launcher itself. Old owner cleanup requires a verified operator O exit record and
absence of native control; failed startup never authorizes automatic native shutdown.

## Deploy with ankle relief

Future real deployments must include the verified bilateral ankle response in the
single body owner. Use packaged `data/ankle_relief` tokens, gain 1.5, native90 alarm
semantics and a 10-second repeat interval after successful return. Any ankle pitch
or roll winding/driver temperature >=90 C latches that motor's alarm; it clears
below 85 C. Announce the corresponding side successfully before moving it. If
both sides need relief, serialize them. Never queue a movement for each native
five-second speech warning. Missing audio, failed return or controller faults do
not authorize automatic retries. Native speech itself is not an audio input: the
owner follows the same temperature hysteresis and supplies its own announcement.

Replay SONIC motion tokens, never recorded `action.wbc` motor commands. Keep one
body publisher and one BrainCo owner; optional VLA must yield to this owner.
Do not start the standalone RTC publisher alongside it. Holding-phase manual foot
adjustments do not lock the controller for a single speed spike; new movements
wait for settled feedback. Preserve active-motion, freshness and native protections.
Native O remains the site stop. Planner idle and a stopped upper-level process do
not mean motors are off. Prepare a new frozen run for code changes. New owner scripts clean retired desktop
windows with the frozen terminal manager. Retain model/camera services and the
current run; never close a live controller/owner or agent as terminal cleanup.

The user accepted the real response in E096. Logs contain two completed left
thermal actions; this does not quantify cooling effectiveness or ground clearance.
E109 subsequently reached 130 C winding temperature and reported a motor fault
during repeated bilateral relief. The return holds the demonstration first token,
not the pre-action stance; settled feedback does not establish unloading. See the
E109 postmortem before reusing this sequence as sustained thermal relief. The
10-second schedule is implemented, but its thermal effectiveness is not validated.
Raw temperature indices are casing [0], winding [1], despite the legacy field name.

No extra precision calibration or per-session demonstration is required solely
because SONIC has tracking residuals.

## Observe, act, verify

Operator sessions default to unlimited (`--lease-seconds 0`). Use
`g1_session_lease(action="status")` to inspect; `action="set", seconds=0` removes
the deadline; poll `action="result", job_id="..."` to confirm application.
Only set a finite duration (>=30 seconds, renewed from now) if the operator asks.
Unlimited status is `lease_unlimited=true, lease_remaining_seconds=null`, not expiry.
The owner can recover an expiry-only fault after fresh settled feedback, without
resuming old motion or sending O. Other faults cannot be cleared this way.
An older frozen owner without this interface requires an operator-controlled
native stop before replacing it; never hot-swap publishers or edit status as recovery.
This does not remove the user's 15-second VLA segment limit or native watchdogs.

- `g1_status`: measured 29-joint body state (Isaac order, rad/rad/s), camera and owner availability.
- `g1_observe`: fresh rectified 640×360 RGB-D, valid mask, K, saved frame and image.
- `g1_pixel_to_position`: saved-frame optical metres (x right, y down, z forward).
  Base coordinates require calibrated extrinsics; absent depth is an error.
- `g1_plan_arm_motion`, `g1_plan_reach`, `g1_plan_grasp_step`, `g1_plan_walk`:
  previews supported by the configured commissioning executor, not generic motion planners.
- `g1_execute_reference`: submit a single-use plan within 30 seconds;
  `g1_reference_result`: poll its measured result, then observe again.

BrainCo has six normalized actuator channels per hand, not Dex3 joint angles.
Reference completion, observed movement and task success are separate claims.
Inspect hand/object geometry before grasping and after a VLA segment; closed
fingers or a completed action budget do not establish a successful grasp.

## Optional household VLA tool

UI operator messages are available through `g1_operator_messages(action="read")`.
Read before new actions and at every 15-second segment boundary, including while
waiting for a handoff. Ordinary G1 tool results also include pending messages.
Acknowledge each current instruction with `action="ack", message_id="...",
reply="..."`; say what you will do, or why you cannot, in the public reply.
Acknowledgement is not execution. Pending instructions block new motions (never
pause); don't acknowledge without considering the user's change of intent.
With `harness.codex_cli.ui_agent` running, new UI messages automatically wake a
bounded Codex turn through the existing MCP socket. UI task submission is the
user instruction: do not require another "start" message, or stop after prepare
when grasping was requested. Read, acknowledge, observe, choose tools, await
results and inspect the scene. Read new messages while waiting and before ending.
The daemon remains idle between tasks; it never starts/stops native control.
The UI shows Agent 待命 / 执行中 / 离线 from a fresh dispatcher heartbeat.
The UI bridge also exposes `g1_task_step`: code submits and waits for one reviewed
move/prepare, returning a fresh image; grasp/check returns early visual checkpoints,
and pause holds only the workflow's own upper-level job. Prefer this compiled path
for Mickey tasks. Observe first, provide truthful scene checks and unique request
IDs, and verify holding in two distinct source frames.
For the user's replay-style right-side route request, prefer `g1_replay_route`:
preview shows the live image and current measurements; run executes eight
taught segments in one Python call, with current observations and measured movement checks.
Use start_ready for current starting area/heading suitability; start_matches is a legacy alias.
There is no historical image/depth comparison. Do not add those requirements.
Turn targets are relative to the current initial heading and the taught measured increments;
Travel residuals up to 10 degrees are logged and accepted; 5 degrees is diagnostic,
not a hard pass condition. Current route clearance must allow this coarse heading.
Larger residuals receive at most two bounded corrections. Less than 1 degree
improvement after a correction returns a recovery checkpoint without repeating it.
The cumulative heading goal is retained, including checks before translations.
These are commissioning values, not hardware-validated or official RAX tolerances.
Translations remain taught primitives,
not a position servo. Step results use ok/measured/why.
Current empty hands, whole right-side corridor and destination arm clearance must
be supported by current UI/visual evidence, including bounded turn corrections.
After prepare, route_complete_review_current_grasp returns the CURRENT frame.
Continue in the same UI turn: inspect target/hand geometry, optionally call
localize(frame_id,selections=[{x,y}]) for fresh optical-frame surface points through
g1_pixel_to_position, then g1_task_step grasp/check/pause/confirm_hold when reachable
and clear. Do not ask for a redundant start or finish after prepare alone.
Camera-to-body extrinsics are NOT calibrated; do not invent pelvis/level coordinates
or convert camera offsets directly to walking commands. Missing measurements or
failed motion returns a checkpoint; do not bypass it or replay from the endpoint.
At a recoverable heading checkpoint, inspect the current scene and continue the same
UI turn via g1_task_step when its preconditions hold. The recovery field reports the
next route index and remaining commands as planning context. Do not repeat a failed
weak turn, restart the whole route, or treat every checkpoint as the end of the task.
The composed route is not hardware-validated. Final lift still needs two-frame
visual verification. One route call has a 180-second limit; new UI messages
interrupt it through the existing upper-level pause mechanism.
Target 3–5 minutes, with a 600-second UI turn budget; expiry pauses only
upper-level work and never sends O, kills owner/native, or turns off motors.
Without that daemon, messages are only stored; an ended interactive Codex turn
is not automatically resumed. Unhandled messages expire after 120 seconds or an
owner session change. Claimed work is never automatically replayed after a crash.
Do not run an interactive control agent alongside the UI agent. Public responses
and MCP tool events appear in the UI; private model reasoning is not displayed.

Use `g1_hand(side="left"|"right"|"both", action="open"|"close", evidence="...")`
for hand actions, and poll `g1_hand_result(job_id)`. Observe contact/release geometry
first. Pause VLA or stop movement, then wait for settling. The existing paired
BrainCo owner blends selected hands over two seconds, preserves any unselected
hand target, and does not replace the body's current reference. It requires fresh
body/hand feedback and known paired hand targets (established by initial VLA
prepare). After locomotion it works without re-preparing VLA; do not reset the
body or open both hands merely to release one. Completion is not physical grasp
or release success. Observe afterward, and do not release a held object absent
the user's instruction and suitable placement/support. Two UI-requested right-hand
releases were verified on 2026-10-20; left/both and closing have offline tests only.
Evidence: `docs/g1/evidence/ui-hand-release-20261020/`.

`g1_move(action="run", direction="...", frame_id="...", evidence="...",
duration_s=2)` exposes forward/backward/left/right, four diagonals,
rotate_left/right and step_in_place. The operator fixes the speed reference at 0.3
(m/s for translation): omit `speed` or pass `0.3`; do not tune speed from experience.
Choose direction and duration from current observations. Rotation's heading rate
remains `yaw_rate_deg_s`, a separate quantity. Hand open/close and left/right/both
selection follow the operator's instruction; do not infer extra hand actions from
past trials. Get a fresh `g1_observe` frame first; evidence
must describe the actual path and hand/object state. The front camera cannot
establish rear clearance. Poll `g1_move(action="result", job_id="...")` and
observe again. See `docs/g1/LOCOMOTION.md` for speed, duration and turn limits.
`g1_move(action="stop")` requests planner idle and preserves native control;
it never sends O. Status/result/stop remain available with pending UI messages.
Movement and VLA/hand actions are serialized by the same owner. Movement holds
captured waist/arm references and existing hand targets, but does not guarantee
physical pose or grasp retention. It invalidates VLA preparation; hand tools remain
usable after settling. These locomotion primitives have simulation evidence,
not a completed loaded real-robot acceptance. They specify duration, not exact
step counts or measured distance. Prepare future integrated runs with
`--agent-tools` (enables VLA, hands and locomotion in one frozen owner).

Use `g1_vla_status` to read the configured model, six trained task names and owner
state. Deployment uses the exact `home_rtc_quality_bs64_8ep_final` bundle; do not
substitute a launcher alias. VLA is one toolbox option, not a replacement for
observation, reference planning or task verification.

1. `g1_vla(action="prepare")` blends to the model's initial body pose and opens
   both BrainCo hands over six seconds. Do not use prepare while holding an object.
2. Poll `g1_vla_result(job_id)`; then `g1_observe` and inspect the current scene.
3. `g1_vla(action="run", task="mickey", steps=750, evidence="...")` runs a bounded
   segment (50 Hz; 1–1500 steps). The operator's E115 update requests 750 steps
   (15 seconds) per segment before observation/intervention; protections may
   interrupt earlier. Evidence describes the actual current view,
   reachability and free path. Use `cola` for table cola; table Mickey is not a
   sofa-trained task. All six exact prompts are exposed in status.
4. Read the result and observe again. `step_budget_complete` is not grasp success.
   If the object is held, do not reset/open hands automatically.

`g1_vla(action="pause")` discards the RTC queue and holds the last body target;
it neither retracts nor opens fingers. No automatic task resume occurs after
thermal relief. A thermal interruption invalidates initial-pose readiness: inspect
the object and hands first, then decide how to proceed. Preparing again opens the
hands, so resolve any held object before doing so. Native O remains unchanged.

The standalone RTC keyboard publisher must not run alongside the integrated
owner. E115 completed one real MCP/VLA Mickey tabletop grasp: 750 steps at 50 Hz,
two subsequent images showed the toy lifted in the right hand, and the operator
confirmed taking support and pressing O. See EXPERIMENTS.md and the E115 report.
This does not establish repeated success or safe thermal motion while holding an
object. The model uses CPU threads=1, delay=8 and request interval=30. The operator subsequently prohibited all automatic O injection. The trial
terminal now only reports temperature (130 C), stale-state, owner-fault and
150-second warnings; none sends O. Native protections remain unchanged.

## Operator-exclusive native shutdown

The user explicitly prohibits agent-initiated O or equivalent native control shutdown, including timers, stale telemetry, owner faults and software temperature thresholds. Only the operator may manually stop native control, or explicitly authorize a specific shutdown first. The robot’s own native/hardware protection remains independent. Pause/discard VLA action queues and report faults without injecting O, killing the native process, or treating a STOP_O file as authorization. Never tell the user that an automatic 150-second O will occur.
