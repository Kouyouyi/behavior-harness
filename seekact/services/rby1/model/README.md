# RB-Y1 model A URDF

`rby1a.urdf` is `models/rby1a/urdf/model.urdf` of Rainbow Robotics' rby1-sdk at commit
`f3bb123` (v0.8.3-3, the SDK commit this robot's controller reports), Apache License 2.0.
Collision geometry is capsules only; the visual meshes it names are not needed for
kinematics and are not copied.

The backend uses the robot's own model (`Robot.get_robot_model()`) on the real robot;
this copy is for offline use: unit tests, the dashboard telemetry's end-effector FK and
`urdf_path` when the operator wants to pin a model.
