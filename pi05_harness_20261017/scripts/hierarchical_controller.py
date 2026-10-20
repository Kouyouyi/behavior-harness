"""Onboard-input-only controller with explicit observation requests and execution logs."""
from collections import deque
from pathlib import Path
import json
import numpy as np

class HierarchicalController:
 def __init__(self,planner,policy,run_dir,execute_steps=8):
  if not isinstance(execute_steps,int) or not 1<=execute_steps<=32:raise ValueError("execute_steps must be between 1 and 32")
  self.planner=planner;self.policy=policy;self.run_dir=Path(run_dir);self.run_dir.mkdir(parents=True,exist_ok=True);self.execute_steps=execute_steps;self.episode=-1
 def reset(self,goal):
  self.episode+=1;self.episode_dir=self.run_dir/f"episode_{self.episode:06d}";self.episode_dir.mkdir(exist_ok=False)
  self.goal=goal;self.history=[];self.queue=deque();self.planner_calls=0;self.finished=False;self.last_status="need_observation";self.action_index=0;self.active_command=None
  self._log({"event":"reset","goal":goal})
 def _log(self,event):
  with (self.episode_dir/"controller_events.jsonl").open("a") as f:f.write(json.dumps(event,ensure_ascii=False)+"\n")
 def act(self,observation):
  # The runner supplies fresh onboard inputs; no environment or scorer is accepted.
  if self.finished:return None
  if not self.queue:
   result=self.planner.plan(observation["image_paths"],self.goal,self.history,self.episode_dir/f"planner_{self.planner_calls:06d}");self.planner_calls+=1
   command=result["command"];self.history.append({"command":command,"observation_step":observation["step"]});self.last_status=command["status"]
   self._log({"event":"planner_decision","planner_call":self.planner_calls,"observation_step":observation["step"],"command":command,"instruction":result["instruction"]})
   if self.last_status!="execute":
    self.finished=self.last_status=="reached";return None
   actions=np.asarray(self.policy.predict(observation,result["instruction"]))
   if actions.ndim!=2 or actions.shape[1]!=23 or len(actions)<self.execute_steps or not np.isfinite(actions).all():raise ValueError("VLA chunk must contain enough finite 23D actions")
   self.active_command=command;self.queue.extend((i,a.copy()) for i,a in enumerate(actions[:self.execute_steps]))
  offset,action=self.queue.popleft()
  self._log({"event":"action_issued","action_index":self.action_index,"planner_call":self.planner_calls,"chunk_offset":offset,"observation_step":observation["step"],"motion_tags":self.active_command["motion_tags"],"action":action.tolist()})
  self.action_index+=1
  return action
