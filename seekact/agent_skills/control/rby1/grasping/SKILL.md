---
name: grasping
description: Use the RB-Y1's two-finger grippers - approach, open, close with a force limit, check the hold from the measured opening, wrist force and an image, and release. Use for every pick, hold or placement.
---

# Grasping

Each arm ends in a two-finger parallel gripper (opening 0 to 0.1 m) driven by a
Dynamixel in current-limited position mode: when closing, the fingers stop where they
meet the object and squeeze with at most `current_a` amps. That current is the grip
force; the default is in the `{{ robot.gripper }}` description.

The first `{{ robot.gripper }}` call of a session homes both grippers: each opens and
closes fully once (about 10 s). Do it before the first grasp, with nothing between the
fingers.

## The fingers may not match the model

Look at both grippers in an image before the first grasp of a session (bring a hand into
the head camera's view and `{{ robot.observe }}`). **Round fingertips are custom fingers
added by hand**, not the vendor's: their length, shape and contact points differ from the
URDF fingers (`ee_finger_*`), so the 0.073 m pad offset below, the fingertip positions
from FK and the finger collision geometry are wrong for them. Their measured lengths
(closed ball 0.195 m, open tips ~0.20 m along −z) are in
[robot-interface](../robot-interface/SKILL.md#custom-fingers-measured). With round fingers:

- plan with those measured lengths; if the fingers in the image look different, measure
  them again from images of the open and closed gripper next to the hand pose in the
  reply, and record it in `memory/TIPS.md`;
- round tips touch at a point, not a flat pad: grasp objects at their flat sides, below
  their centre of mass where possible, and expect them to rotate in the grip; check the
  hold from an image and the wrist force before lifting further;
- keep a few extra millimetres of clearance from surfaces when approaching: the real tips
  may reach further than the model's.

## A grasp

1. **Localize** the object ([object-localization](../object-localization/SKILL.md)) and
   choose the grasp axis. In the end-effector frame the fingers point along −z (the URDF's pads
   0.073 m beyond `ee_*`; measure it for round custom fingers, see above) and close along x, so the object's width along the end effector's
   x axis must be below 0.1 m with some margin, and `ee_*` itself stops about 0.07 m short
   of the grasp point along the approach.
2. **Open** (`action: open`) before the approach.
3. **Pre-pose** 0.10 m before the grasp point along the approach direction, then the
   final approach with `{{ robot.move }}` in steps of a few centimetres; observe before the
   last step. A wrist force rise of 60 N stops a motion: that is contact.
4. **Close** (`action: close`, `current_a` higher for heavier or slippery objects). The
   reply's `measured_width_m` and `blocked_by_object` tell whether the fingers stopped on
   something: an opening near 0 means they closed on air.
5. **Lift** a few centimetres with `{{ robot.move }}`, then check the hold: the opening is
   unchanged, `wrist_force_n` in `{{ robot.status }}` shows the added weight, and an image
   shows the object between the fingers.
6. **Release** with `action: open` (or `width` to a known opening) only once the object
   is supported.

## Rules

- Never drive or change the torso with an object held out; bring it close to the body
  first.
- `rby1_hold` holds the grippers where they are. At the end of the session the grippers
  go limp: put objects down before you finish.
