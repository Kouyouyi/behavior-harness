# RB-Y1 setup notes

What the robot and its UPC look like (read through the SDK, not changed), and the lessons
from earlier robot stacks that shaped the backend.

## 1. System map

| Part | Where | Notes |
|---|---|---|
| Robot controller | 192.168.30.1 | gRPC :50051 (`rby1_sdk`), web UI :5173. Reports SDK commit `f3bb123` (v0.8.3-3). Unauthenticated: anything on the network can command it. |
| UPC | 192.168.30.2 on the robot network | Jetson AGX Orin 64 GB, L4T R36.5, aarch64. Check first that the robot-network NIC transmits (`ping 192.168.30.1`): on our unit the r8168 NIC needed a driver fix before it did. |
| Grippers | `/dev/rby1_gripper` → a ttyUSB (FTDI, a udev rule gives the stable name) | Two Dynamixels, protocol 2.0, 2 Mbaud, ID 0 right, ID 1 left, powered by the tool flanges at 12 V. One process owns the bus. |
| Master arm | `/dev/rby1_master_arm` → a ttyUSB | The teleoperation device; not used here. |
| Head camera | ZED 2i, USB | Mounted on `link_head_2`; its hand-eye calibration is the backend option `head_camera_extrinsic`. One process opens it. |
| Lidar | Livox MID-360 on the robot network | Not used by this backend. |

The model: `Model_A`, 24 DoF in `robot_joint_names` order: two wheels, `torso_0..5`,
`right_arm_0..6`, `left_arm_0..6`, `head_0..1`. Differential drive (wheel base 0.53 m,
wheel radius 0.1 m, centre of rotation 0.228 m ahead of `base`). The `base` frame's
origin is on the floor (wheel centres at z = 0.1 m).

Read through the SDK (`get_robot_info`, `get_state`): model A v1.0, controller version
0.8.0 / SDK 0.8.3, one emergency-stop input `emo1`, power rails 5 / 12 / 24 / 48 V (48 V =
battery, 57.5 V at 88 %). Brakes on `torso_0..5` and `right/left_arm_0..5`; none on the
wrists (`*_arm_6`), the head or the wheels. The controller's own URDF is `RBY1_A_v1.0`,
the same model as `services/rby1/model/rby1a.urdf`. The raw wrist F/T readings carry
offsets of tens of newtons at rest, so the backend's force guard uses the rise over each
motion's start.

## 2. Environments

- The robot service's environment (`RAX_PYTHON`) must carry the `rby1-sdk` version that
  matches the controller's firmware (0.8.3 for controller 0.8.0). A newer SDK talking to
  an older controller is not supported; upgrade both together.
- The agent's environment (`RAX_AGENT_PYTHON`) has no robot SDK at all.
- Leave other environments on the UPC alone (vendor or teleoperation stacks), and list
  every one that has the SDK in `RAX_PROTECTED_PATHS`.
- If ROS 2 is sourced in `~/.bashrc`, it puts its packages on `PYTHONPATH`; run the
  harness's Python with `env -u PYTHONPATH`.

## 3. Other robot stacks on the UPC

Anything else on the UPC that can command the robot (a robot server, a gamepad teleop, an
SDK checkout with its examples) is a second writer. Never run one next to a session, and
add its checkout and environment to `RAX_PROTECTED_PATHS` so the agent's sandbox cannot
read or run it. Watch for stacks that change the robot as a side effect: one server stack
we found powered the robot on as soon as its client was constructed, homed the grippers
at start-up and persisted motion parameters to the controller.

## 4. Lessons that shaped the backend

- **Connecting must change nothing** (Rule 1). The SDK's `connect()` is read-only; a
  client that powers on or enables the control manager on construction is not. The
  backend defers power, servo and the control manager to the first armed motion.
- **A dropped SDK handler cancels its command.** `RobotCommandHandler` and
  `CommandStreamHandler` cancel in their destructors, so `SdkRobot` keeps every live
  handler, including the hold it sends after a stop.
- **Liveness is not validity.** A session once ran on a frozen all-zero lidar pose. The
  backend checks each moving joint's own `time_since_last_update` every tick, and a base
  whose odometry does not change while driving is stopped.
- **Check force every tick.** A force watchdog that checks only between steps misses a
  contact during a step. Here the wrist force is checked every control tick of an arm or
  torso motion.
- **One writer.** A teleop mode that commands :50051 directly during a session defeats
  the backend's checks.
- **A silent gripper fails start-up.** The controller can drop off the network
  mid-session, and a gripper that does not answer fails with "Dynamixel ID 0 is not
  active" (the right gripper). Check the gripper's cable and flange power before the
  `gripper` motion-check step.

## 5. Sandbox notes

The agent's Bash sandbox allows internet hosts through its proxy, and the controller's
gRPC port is unauthenticated, so the controller is in the launcher's `DENIED_HOSTS`
(`sandbox.network.deniedDomains` and a `WebFetch(domain:…)` deny rule): the factory
address 192.168.30.1, plus the host of a session's `address` backend option. The proxy
matches literal hosts (no CIDR ranges); a hostname that resolves to the controller is not
covered by that entry. Verify in the sandbox audit (checklist step 10 in
[RBY1.md](../../RBY1.md)) that a script cannot reach the controller's :50051 by IP, by an
alias, or through the SOCKS proxy; a firewall rule for the agent's traffic is the
stronger fix.
