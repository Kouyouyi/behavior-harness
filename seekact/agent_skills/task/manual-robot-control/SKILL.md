---
name: manual-robot-control
description: Solve the task on the real {{ robot.name }} by choosing robot actions directly from observations. Use for solving the task by hand, recovery, and preparing the scene for script tests.
---

# Manual robot control

This is a real robot with people nearby. Every motion is physical and cannot be
undone; move slowly, in small inspectable steps, and stop when unsure. First learn how
to complete the task yourself through the robot tools, then automate what worked.

## First look

1. Read `TASK.md` and identify the required object relationships, orientation,
   placement and robot end state. Write a short completion checklist in your notes.
2. Call `status` and read `MCP_SESSION.md`. Inspect the robot with
   `{{ robot.status }}` and the scene with `{{ robot.observe }}`. If the robot reports
   that motion is not armed, do not try to work around it: the operator arms motion.
3. View the current camera images. Identify the relevant objects and a short sequence
   of manipulation stages.
4. Ground the next target in current observations with
   [object-localization](../object-localization/SKILL.md), choose a small action, and
   pick the motion tool for it as [robot-interface](../robot-interface/SKILL.md)
   describes.
5. Inspect the reply and new images, revise the next action, and save useful
   parameters and evidence. Work through the checklist until every condition holds.

## Observe and choose actions

Use these skills as needed rather than loading every reference up front:

- [Robot interface](../robot-interface/SKILL.md) for this robot's motion tools,
  frames, limits and conventions. Read it before composing actions.
{% if camera_geometry %}
- [Object localization](../object-localization/SKILL.md) for metric positions of
  visible object points from the current images and depth.
{% else %}
- [Object localization](../object-localization/SKILL.md) to locate visible object
  points from the current images.
{% endif %}
- [Experience memory](../experience-memory/SKILL.md) to record and reuse tested
  motion strategies under matching observable conditions.

{{ block("images.forwarding") }}
{% if robot.look_around %}

## Active perception

The camera sees only what it faces. When the target, a placement spot, an obstacle or
a person may be beside or behind the robot, move the viewpoint yourself instead of
guessing. To see sideways and behind, {{ robot.look_around }}.

**Clear your own path; do not ask the operator.** Before an in-place turn, a reverse or
any motion that sweeps space you cannot see, plan a look-around and check that space
in the images and depth yourself. Ask the operator only when the views cannot settle
it: a person stays in the way, or the space remains hidden.

Keep the arms close to the body, call `{{ robot.observe }}` at each view, and note
the joint angles and odometry of each view so you can relate it to the robot's
frame. Turn back to the working view before manipulating, and re-localize the target
there.
{% endif %}

## Motion and feedback

Distinguish commanded targets from measured states: the reply's measured pose tells
you what physically happened. Real tracking is imperfect; check the measured error
before the next step rather than assuming the target was reached.

Observe the result of each meaningful action and adapt. When a tool call fails, read
its full error (`reason`, `robot_state`) and observe before the next motion; do not
repeat the motion blindly: reason about what happened, and continue only from a known
state. If the robot refuses a target
(workspace or joint limit), treat it as a wall, not a fault. If localization fails or
is uncertain, take a new view; never replace missing geometry with guessed
coordinates.

## Confirm success

Partial visible progress or a clean tool return is not completion. Check every
condition in `TASK.md` against fresh images from more than one camera. Record the
successful strategy and evidence in `memory/`, then automate it as AGENTS.md
describes.
