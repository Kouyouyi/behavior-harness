---
name: object-localization
description: Metric 3-D positions of visible object points in the RB-Y1 base frame from head-camera RGB only (no depth in this session) - by intersecting a pixel ray with a known plane, or by triangulating from two camera positions. Use before choosing any approach, grasp, placement or drive target.
---

# Object localization (head camera, RGB only)

This session has no depth, so `{{ robot.pixel }}` cannot back-project. Every saved frame
still has exact camera geometry: `runtime/frames/<frame_id>/meta.json` holds `K` (3×3)
and `T_base_cam` (camera-to-base 4×4, from the joint state at capture time) for
`head_camera` (rectified, no distortion). A full-resolution pixel `(u, v)`
defines a ray in the base frame:

```python
import json, numpy as np
meta = json.load(open(f'runtime/frames/{frame_id}/meta.json'))['cameras']['head_camera']
K, T = np.array(meta['K']), np.array(meta['T_base_cam'])
d = T[:3, :3] @ np.linalg.solve(K, [u, v, 1.0])
origin, direction = T[:3, 3], d / np.linalg.norm(d)
```

Preview images are downscaled: full-resolution = preview pixel / `preview_scale` (from
the `{{ robot.observe }}` reply).

## Method 1: a known plane (one view)

For a point on a horizontal surface of known height `h` (the floor is z = 0; a table
height from `TASK.md` or from the operator via `request_operator`):
`s = (h − origin_z) / direction_z`, `point = origin + s · direction`, valid when
`direction_z < 0` (the ray goes down). Use the contact point of the object with the
surface (its lowest visible edge centre), not its top.

## Method 2: two camera positions (triangulation)

1. Observe with the object near the image centre (`rby1_look_at`); note its pixel.
2. Move the camera at least 0.15 m with the base still, then observe again and find the
   same object point. Changing the torso height moves the camera far enough: `ready`
   (a = 30°) → a = 60° (`[0, 1.047, −2.094, 1.047, 0, 0]`) lowers it 0.26 m in one call;
   leaning (`torso_3` 20° above a) moves it 0.18 m forward. Turning the head moves it only
   a few centimetres: too little. Base moves slip: do not use them for triangulation.
3. Take the midpoint of the shortest segment between the two rays:

```python
def triangulate(o1, d1, o2, d2):
    w = o1 - o2; a, b, c = d1 @ d1, d1 @ d2, d2 @ d2; e, f = d1 @ w, d2 @ w
    den = a * c - b * b                      # ~0 when the rays are parallel
    s, t = (b * f - c * e) / den, (a * f - b * e) / den
    p1, p2 = o1 + s * d1, o2 + t * d2
    return (p1 + p2) / 2, np.linalg.norm(p1 - p2)   # point, gap between the rays
```

Accept the point when the angle between the rays is ≥ 5° and the gap is < 2 cm;
otherwise move the camera further or pick a sharper feature (a corner, a mark).

## When positions go stale

Positions stay valid while neither the base nor the object moves. After a base move or
any change in the scene, observe and localize again; check recent changes with
`{{ robot.stream }}` ([live-stream](../live-stream/SKILL.md)).
