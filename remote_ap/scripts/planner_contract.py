"""Qualitative planner -> VLA contract; no angle or distance parameters."""
import re,json,jsonschema
from pathlib import Path
WORDS={'turn_left':'向左转','turn_right':'向右转','forward':'向前移动','backward':'向后移动','strafe_left':'向左平移','strafe_right':'向右平移','raise_view':'抬高观察位置','lower_view':'降低观察位置'}
OPEN={'open_door':'打开','open_drawer':'拉开','open_lid':'揭开'}
MOTIONS=list(WORDS)+list(OPEN)
SCHEMA={'type':'object','additionalProperties':False,'properties':{'status':{'type':'string','enum':['execute','reached','need_observation']},'motion_tags':{'type':'array','items':{'type':'string','enum':MOTIONS}},'target_description':{'type':'string'},'goal_kind':{'type':'string','enum':['inspect_target','inspect_container_interior']},'visual_evidence':{'type':'string'},'target_visible':{'type':'boolean'}},'required':['status','motion_tags','target_description','goal_kind','visual_evidence','target_visible']}
VALIDATOR=jsonschema.Draft202012Validator(SCHEMA)
def validate(command):
 VALIDATOR.validate(command);tags=command['motion_tags'];target=command['target_description']
 if len(tags)!=len(set(tags)):raise ValueError('Duplicate motion tags')
 capability_path=Path('/cs/kyy/behavior_ap_experiment/reports/sampling_distribution.json')
 if tags and capability_path.exists():
  known={frozenset(k.split('+')) for k in json.loads(capability_path.read_text())['combination_counts']}
  if frozenset(tags) not in known:raise ValueError('Motion combination has no training examples; use a supported short command')
 if any(a in tags and b in tags for a,b in [('turn_left','turn_right'),('forward','backward'),('strafe_left','strafe_right'),('raise_view','lower_view')]):raise ValueError('Contradictory directions')
 if re.search(r'(?:[0-9]+(?:\.[0-9]+)?|[一二三四五六七八九十百]+)\s*(?:度|厘米|毫米|米)' ,target):raise ValueError('Numeric parameters are excluded from the qualitative command')
 if command['status']=='execute' and (not tags or not target.strip()):raise ValueError('Executing requires motion and target')
 if command['status']!='execute' and tags:raise ValueError('Nonexecuting status must not carry motion')
 if command['status']=='reached' and not command['target_visible']:raise ValueError('Cannot declare reached when target is not visible')
 opening=[x for x in tags if x in OPEN]
 if opening and (len(tags)!=1 or command['goal_kind']!='inspect_container_interior'):raise ValueError('Opening must be a single container-inspection command')
 if not opening and tags and command['goal_kind']!='inspect_target':raise ValueError('Navigation goal kind mismatch')
 return command

def render(command):
 validate(command)
 if command['status']!='execute':return None
 target=command['target_description'];tags=sorted(command['motion_tags'],key=MOTIONS.index)
 if tags[0] in OPEN:return OPEN[tags[0]]+target+'，观察'+target+'内部。'
 return '并'.join(WORDS[t] for t in tags)+'，观察并接近'+target+'。'
if __name__=='__main__':
 root=Path('/cs/kyy/behavior_ap_experiment');(root/'configs/planner_command.schema.json').write_text(json.dumps(SCHEMA,ensure_ascii=False,indent=2))
 c={'status':'execute','motion_tags':['turn_right','forward'],'target_description':'收音机','goal_kind':'inspect_target','visual_evidence':'example only, not a live planner observation','target_visible':False}
 assert render(c)=='向右转并向前移动，观察并接近收音机。'
 for tags in [['torso_left'],['turn_left','turn_right'],['open_drawer','forward']]:
  try:render({**c,'motion_tags':tags})
  except (ValueError,jsonschema.ValidationError):pass
  else:raise AssertionError(tags)
 print('planner contract checks passed')
