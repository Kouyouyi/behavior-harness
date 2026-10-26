---
name: object-localization
description: Interpret G1 stereo depth and its uncalibrated robot transform.
---

# Stereo localization

Use `g1_observe` then `g1_pixel_to_position`. The image is rectified; do not use pixels
from the raw left preview or a different frame. Depth comes from factory fisheye
calibration plus stereo matching, with invalid pixels masked. Real distance accuracy
and camera-to-body extrinsics have not passed independent measurement.

Camera coordinates cannot be used as a robot reach target. A plush surface, shiny
can, occlusion edge, or weak texture can have missing or biased depth. Inspect valid
samples and multiple observations; do not fill unknown depths with plausible values.
