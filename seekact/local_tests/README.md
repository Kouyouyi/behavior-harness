# Regression tests

Run from the repository root in the robot service's Python environment:

```bash
env -u PYTHONPATH "$RAX_PYTHON" -m pytest -q
```

`env -u PYTHONPATH` keeps packages a ROS installation puts on `PYTHONPATH` (often from
`~/.bashrc`) out of the run; their pytest plugins fail to import in this environment.

The suite uses temporary artifact roots, the `mock` robot backend, a simulated RB-Y1
behind the `SdkRobot` surface with the vendor URDF, and fake model providers. It never
moves hardware, opens cameras, calls a model or needs credentials. It needs `rby1-sdk`
(for the vendor kinematics) but no robot. Tests marked `hardware` are excluded by
default.
