"""Simulator-side paired scene mutations. Never pass these objects or reports to the policy.

Prepared against BEHAVIOR-1K v3.9.3-post2; not yet executed in OmniGibson.
The runner must supply real physics settling, geometry and rendered-visibility validators.
"""
import copy
import random
import numpy as np

def array(value):
 if hasattr(value,"detach"):value=value.detach().cpu().numpy()
 return np.asarray(value,dtype=float)

def robot_pose(robot):
 position,quaternion=robot.get_position_orientation()
 return array(position),array(quaternion),array(robot.get_joint_positions())

def check_robot_unchanged(before,after,tolerance=1e-3):
 p,q,j=before;p2,q2,j2=after
 position_error=float(np.max(np.abs(p-p2)));joint_error=float(np.max(np.abs(j-j2)))
 # q and -q encode the same orientation.
 quaternion_error=float(min(np.max(np.abs(q-q2)),np.max(np.abs(q+q2))))
 if max(position_error,joint_error,quaternion_error)>tolerance:raise RuntimeError("Robot initial configuration changed during scene setup")
 return {"position_max_abs_error_m":position_error,"joint_max_abs_error":joint_error,"quaternion_max_abs_error":quaternion_error}

def prepare_condition(sim,robot,target,condition,*,destination=None,television=None,seed,settle,validate_geometry,target_visible):
 """Mutate a fully loaded and settled single-scene instance, then validate or roll back.

 Callbacks operate on privileged simulator state outside the planner/VLA:
 settle(): fixed physics-settling procedure shared across conditions.
 validate_geometry(): dict with no_interpenetration plus support_under_tv (table)
 or target_fully_inside (container), based on actual geometry.
 target_visible(): bool from rendered onboard views (not an Inside/Open proxy).
 """
 if condition not in ("original","radio_on_table_under_tv","radio_inside_closed_container"):raise ValueError(condition)
 if condition!="original" and (destination is None or destination is target):raise ValueError("Distinct destination object required")
 if condition=="radio_on_table_under_tv" and television is None:raise ValueError("Explicit television object required")
 if not all(callable(f) for f in (settle,validate_geometry,target_visible)):raise ValueError("Actual setup validation callbacks required")
 import torch
 from omnigibson.object_states import OnTop,Inside,Open
 original=copy.deepcopy(sim.dump_state(serialized=False));robot_before=robot_pose(robot)
 rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 try:
  if condition=="radio_on_table_under_tv":
   if not target.states[OnTop].set_value(destination,True):raise RuntimeError("Could not sample a valid supported target placement")
  elif condition=="radio_inside_closed_container":
   if not destination.states[Open].set_value(True,fully=True):raise RuntimeError("Container cannot open")
   if not target.states[Inside].set_value(destination,True):raise RuntimeError("Could not sample target inside container")
   if not destination.states[Open].set_value(False,fully=True):raise RuntimeError("Container cannot close")
  # Samplers may advance the whole simulation. Restore the paired initial world,
  # then apply only the requested target and container changes.
  target_state=copy.deepcopy(target.dump_state(serialized=False))
  container_state=copy.deepcopy(destination.dump_state(serialized=False)) if condition=="radio_inside_closed_container" else None
  sim.load_state(original,serialized=False)
  if condition!="original":target.load_state(target_state,serialized=False)
  if container_state is not None:destination.load_state(container_state,serialized=False)
  settle()
  geometry=validate_geometry()
  if geometry.get("no_interpenetration") is not True:raise RuntimeError("Post-settling interpenetration validation failed or missing")
  if condition=="radio_on_table_under_tv":
   if not bool(target.states[OnTop].get_value(destination)) or geometry.get("support_under_tv") is not True:raise RuntimeError("Target support/under-TV relation not verified")
  if condition=="radio_inside_closed_container":
   if not bool(target.states[Inside].get_value(destination)) or bool(destination.states[Open].get_value()):raise RuntimeError("Closed-container relation not maintained after settling")
   if geometry.get("target_fully_inside") is not True:raise RuntimeError("Full target containment not verified")
  visible=target_visible()
  if not isinstance(visible,(bool,np.bool_)):raise ValueError("Visibility validator must return a boolean")
  if condition=="radio_inside_closed_container" and visible:raise RuntimeError("Target is still visible; occlusion condition not achieved")
  robot_check=check_robot_unchanged(robot_before,robot_pose(robot))
  position,quaternion=target.get_position_orientation()
  return {"condition":condition,"seed":seed,"target":target.name,"destination":destination.name if destination is not None else None,"television":television.name if television is not None else None,"target_position":array(position).tolist(),"target_quaternion":array(quaternion).tolist(),"initial_target_visible":bool(visible),"geometry":geometry,"robot_initial_state_check":robot_check,"scope":"Privileged setup report; do not feed to planner or VLA"}
 except Exception:
  sim.load_state(original,serialized=False)
  raise
 finally:
  random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
  if rng[3] is not None:torch.cuda.set_rng_state_all(rng[3])
