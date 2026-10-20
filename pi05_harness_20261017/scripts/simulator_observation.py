"""Allowlisted adapter for one flattened OmniGibson evaluator observation.

Source contract: BEHAVIOR-1K v3.9.3-post2 eval/r1pro.yaml and eval_utils.py.
No simulator imports; runtime integration still requires an RTX environment.
"""
from pathlib import Path
import cv2
import numpy as np
CAMERAS=("robot_r1::robot_r1:zed_link:Camera:0::rgb", "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb", "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb")
def array(value):
 if hasattr(value,"detach"):value=value.detach().cpu().numpy()
 return np.asarray(value)
def onboard_observation(flat,step,output_dir):
 if not isinstance(step,int) or step<0:raise ValueError("Nonnegative integer step required")
 state=array(flat["robot_r1::proprio"])
 if state.shape!=(61,) or not np.isfinite(state).all():raise ValueError("Expected one finite 61D R1Pro observation; split batches first")
 images=[]
 for key in CAMERAS:
  rgb=array(flat[key])
  if rgb.ndim!=3 or rgb.shape[2] not in (3,4) or rgb.dtype!=np.uint8:raise ValueError(f"Expected uint8 HWC RGB/RGBA: {key}")
  images.append(rgb[:,:,:3])
 directory=Path(output_dir)/f"observation_{step:08d}";directory.mkdir(parents=True,exist_ok=False)
 paths=[]
 for role,rgb in zip(("head","left_wrist","right_wrist"),images):
  path=directory/f"{role}.png"
  if not cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise RuntimeError(f"Failed to write {path}")
  paths.append(str(path.resolve()))
 # Construct a fresh allowlisted mapping. Never forward task/scorer/segmentation fields.
 return {"state":state.astype(np.float32,copy=True),"image_paths":paths,"step":step}
