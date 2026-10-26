---
name: robot-interface
description: Observe and control the simulated BEHAVIOR R1Pro through its bounded tools and verify official task success.
---

# BEHAVIOR R1Pro simulation interface

This is a local Isaac/OmniGibson simulation. Do not start or contact G1, SONIC,
BrainCo, RB-Y1, VLA deployment services, or real hardware. The complete task is
already authorized; do not wait for ARM or ask the user to name the object again.
The simulator owns physics and uses the `fast` rendering profile. It advances
only when an explicit motion tool executes. Waiting for reasoning freezes physics.

Read the live tool schemas for exact arguments, units and limits. A tool's current
schema and measured response take precedence over this guide or learned experience.

## Observe and act

1. Call `status`, `behavior_state`, and `behavior_observe`. Look at the returned
   camera images. These are current simulated observations, not expert examples.
2. Decide the next bounded action from those observations. Tools expose base motion,
   arms, torso, grippers, and pixel-to-position geometry; they do not provide a
   task-completing grasp/place script. Use `behavior_pixel_to_position` only with a
   visible feature and the current frame identity required by its schema.
3. Use `behavior_move_base`, `behavior_move_arm`, `behavior_torso`, or
   `behavior_gripper` as needed. Submit one motion at a time. Observe after each
   meaningful change, especially before contact, after a possible grasp, and before
   release. Use smaller motions near an object or surface.
4. Verify effects in fresh images and measured state. An ACK, tool completion,
   closed gripper, or proximity is not evidence of a successful grasp or task.
   Do not infer hidden object poses or reuse stale frame coordinates.
5. Continue the entire requested task, not merely search/approach. When it appears
   complete, check the worker's `official_success` (official BDDL).

## Finish and errors

`behavior_finish(outcome="success"|"failure"|"blocked", reason=...)` records
your conclusion. Success is accepted only if the simulator's official BDDL state
is true. If a success claim is rejected, inspect current state and continue within
the remaining budget; do not repeat an unsupported success claim.

After a tool error, timeout, or uncertain ACK, never resend a motion blindly.
Inspect `behavior_state` and fresh images first. If the executed prefix or state
cannot be established, finish as blocked. Infrastructure errors are not evidence
that the task is beyond the model's ability. The supervisor owns worker shutdown
and video finalization; ending the Codex frontend must not replay or reset physics.

## Zero-shot and one-expert boundary

In the zero-shot attempt, use only the task, this interface, and current observations.
No historical skill library, dataset trajectory, other trial, or saved route is
available. Do not request or search for one.

Only an explicitly supplied single expert summary may inform an experience retry.
Treat that summary as a candidate procedure with observable preconditions, useful
effects, and limitations. It cannot change tool limits or the success definition.
Reobserve this new initial scene and adapt the procedure; do not replay expert
actions. Keep observed effects separate from hypotheses and task success.

Give concise Chinese summaries of your intent, actual tool result, and remaining
uncertainty. The launcher saves public messages and tool events as this attempt's
memory/evidence. It does not silently retrieve or merge a persistent skill library.
