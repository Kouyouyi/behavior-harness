---
name: object-localization
description: Metric 3-D positions of visible object points in the RB-Y1 base frame from the head ZED 2i camera's RGB and depth. Use before choosing any approach, grasp, placement or drive target, and again after anything moved.
---

# Object localization (head camera, RGB + depth)

## The camera

ZED 2i stereo camera on the 2-joint head (pan, tilt), left lens. Saved frames are
rectified (no distortion) at the session's camera resolution, with a depth map from the
ZED SDK's neural stereo. `meta.json` gives `K`, the image size and `T_base_cam`. Depth is
least reliable on object edges, dark, shiny or transparent surfaces, very close to the
camera and at the image border; it is best for points near the image centre.

## Pose and accuracy (measured on this robot)

`T_base_cam` is computed for every frame from the joints measured with it: forward
kinematics through the torso and both head joints (pan, tilt) to `link_head_2`, composed
with the fixed hand-eye calibration of the camera mount. Moving the head or torso is
accounted for; a pose from another frame is not valid for this one.

Checked on 112 saved frames (2026-10-08) over head pan −0.47…+0.46, tilt 0…1.3 and torso
from upright to `lowest`, with the floor (z = 0) as the reference:

- the fitted floor 1.5 m ahead came out at −7 mm (median), and did not depend on the head
  or torso angles (at most 1 cm per radian): the pose composition is right;
- a single frame can still be off: 90 % of frames put the floor within 4.6 cm at 1.5 m,
  and the floor plane looked tilted by 1.6° (median, 90 % under 4°). Errors grow with
  distance and towards the image border;
- object edges and end faces are worse: box end faces came out 3–5 cm off; a box's top
  surface and centre were good from 1 m or more;
- close to the robot (and when squatting) the robot's own body and arms appear in the
  depth: points with x < 0.36 m and |y| < 0.32 m in the base frame are the robot.

So: localize from several interior pixels, cross-check from a second view (step 6), and
for contact-critical positions (grasp faces, placing heights) confirm by touch (wrist
force) or a close-up view rather than one frame's depth.

## Procedure

1. **Centre the target**: `rby1_look_at` a rough base-frame position of it (from an
   earlier frame or the task), so it is near the image centre.
2. **Observe**: `{{ robot.observe }}` → `frame_id`. Files in `runtime/frames/<frame_id>/`:
   `head_camera.jpg`, `head_camera_depth.png` (uint16 millimetres, 0 = invalid) and
   `meta.json` (`K`, `T_base_cam` camera-to-base 4×4, every joint at capture time).
3. **Pick pixels in full-resolution coordinates**. The returned preview is downscaled by
   `cameras.head_camera.preview_scale`: full = preview / scale. Choose 3–5 pixels inside
   the object's visible top surface, away from its edges.
4. **Back-project**: `{{ robot.pixel }} {frame_id, camera: "head_camera", pixels:
   [[u, v], ...], radius?}`. Each point is `{pixel, position (base frame), depth_m,
   valid_fraction}` from the median valid depth in a (2·radius+1)² window (default
   radius 3), or `{pixel, error}` when there is no valid depth.
5. **Combine**: object centre = mean x, y of the top-surface points; top height = their
   z. Discard points with `valid_fraction` < 0.5 or more than 2 cm off the others.
6. **Cross-check before a precise action**: `rby1_look_at` the estimate from a slightly
   different direction (or after a small head change), observe, localize again. Two
   estimates agreeing within 2 cm are usable; otherwise pick interior pixels again.

## When positions go stale

The camera pose comes from forward kinematics at capture time, so a position stays valid
while the base and the object do not move. After a base move, re-observe and
re-localize; before acting after a pause, check the scene with `{{ robot.stream }}`
([live-stream](../live-stream/SKILL.md)). Do not move the head while you observe: the
joints in `meta.json` are read just after the image.

## In code

Load the depth PNG with `cv2.imread(path, cv2.IMREAD_UNCHANGED)` (millimetres) and back-
project with `K` and `T_base_cam` from `meta.json`: `p_cam = depth_m · K⁻¹ [u, v, 1]ᵀ`,
`p_base = T_base_cam · [p_cam, 1]`.
