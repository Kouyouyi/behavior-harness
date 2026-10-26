"""Agent-visible controls: robot feedback and RGB-D, never object ground truth."""
import math


def tool(name, description, properties=None, required=()):
    return dict(name=name, description=description, inputSchema=dict(
        type='object', properties=properties or {}, required=list(required), additionalProperties=False))


SIDE = dict(type='string', enum=['left', 'right'])
STEPS = dict(type='integer', minimum=2, maximum=120, default=60,
             description='30 Hz control steps; wall-clock runtime is measured separately.')
V3 = dict(type='array', items=dict(type='number'), minItems=3, maxItems=3)
Q4 = dict(type='array', items=dict(type='number'), minItems=4, maxItems=4)


VLA_TOOL_NAMES = ('behavior_active_perception_vla', 'behavior_champion_vla', 'behavior_vla_continue')
NATIVE_MOTION_TOOLS = frozenset({'behavior_move_base','behavior_move_arm','behavior_gripper','behavior_torso'})
VLA_HANDOFF = 'behavior_handoff_to_champion'
VLA_ASSESS = 'behavior_assess_interaction'


def tools(include_vla=False, vla_only=False):
    if vla_only and not include_vla:
        raise ValueError('VLA-only control requires enabled learned tools')
    definitions = [
        tool('behavior_state', 'Read measured robot joints, gripper opening, end-effector poses and official task success. No object positions or privileged scene state.'),
        tool('behavior_observe', 'Capture current native head/left-wrist/right-wrist RGB images, depth calibration and robot state. Physics remains frozen while thinking. Use returned frame_id for pixel projection.'),
        tool('behavior_pixel_to_position', 'Project a selected pixel of a saved RGB-D frame into camera optical or current robot base coordinates. Not semantic object detection; invalid depth and frames from an earlier physical step are rejected.',
             dict(frame_id=dict(type='string'), camera=dict(type='string', enum=['head','left_wrist','right_wrist']),
                  x=dict(type='integer',minimum=0), y=dict(type='integer',minimum=0),
                  radius=dict(type='integer',minimum=0,maximum=5,default=1),
                  coordinate_frame=dict(type='string',enum=['camera','base'],default='base')), ('frame_id','camera','x','y')),
        tool('behavior_move_base', 'Bounded body-relative base velocity, with a final zero-velocity control step. At most 0.30m translation and pi/2 rotation per request: duration=(steps-1)/30 seconds, so choose speed and steps accordingly. Hold arms and trunk at their measured positions at tool entry, and preserve the latest gripper commands; no teleport. vx forward, vy left, yaw_rate counterclockwise. Observe obstacle clearance; command completion is not navigation success.',
             dict(vx=dict(type='number',minimum=-0.3,maximum=0.3), vy=dict(type='number',minimum=-0.3,maximum=0.3),
                  yaw_rate=dict(type='number',minimum=-0.5,maximum=0.5), steps=STEPS), ('vx','vy','yaw_rate')),
        tool('behavior_move_arm', 'Move one end-effector toward an explicit Cartesian pose through closed-loop Jacobian IK and physical joint control. position in metres, orientation quaternion xyzw. Missing orientation holds the measured orientation. At most 0.3m per request; no object teleport, collision avoidance or grasp-success guarantee. Inspect images and measured residual.',
             dict(side=SIDE, position=V3, orientation=Q4, frame=dict(type='string',enum=['base','world'],default='base'), steps=STEPS), ('side','position')),
        tool('behavior_gripper', 'Set one parallel gripper opening fraction: 1=open, 0=close. Uses the existing R1Pro assisted-grasp controller. Closing does not prove grasp; verify lifting and object following in subsequent images.',
             dict(side=SIDE, opening=dict(type='number',minimum=0,maximum=1), steps=STEPS), ('side','opening')),
        tool('behavior_torso', 'Move four trunk joints toward absolute radian targets through physical joint control. Check measured limits in behavior_state. Hold arms at their measured positions at tool entry, preserve the latest gripper commands, and stop base movement. Observe for collisions.',
             dict(joints=Q4, steps=STEPS), ('joints',)),
        tool('behavior_finish', 'Record the Agent outcome and reason. Success is accepted only when official BehaviorTask.success is true. failure/blocked preserves unsuccessful evidence; no reset or replay occurs.',
             dict(outcome=dict(type='string',enum=['success','failure','blocked']), reason=dict(type='string',minLength=1,maxLength=4000)), ('outcome','reason')),
    ]
    if include_vla:
        from .vla import MOTIONS
        from pathlib import Path
        import json
        ap_combinations=list(json.loads(Path(__file__).with_name('ap_capabilities.json').read_text())['combination_counts'])
        definitions += [
            tool('behavior_active_perception_vla', 'Start a bounded learned AP policy job. It understands only the trained qualitative navigation/view/container-opening tags, not arbitrary grasp/place/switch instructions. Each fresh RGB+61D observation predicts32 original23D actions; execute prefix8. No action runs until behavior_vla_continue. After each completed job inspect returned fresh images. Maximum15 chunks=120 physical steps.',
                 dict(motion_tags=dict(type='array',items=dict(type='string',enum=list(MOTIONS)),minItems=1,maxItems=3,uniqueItems=True),
                      target_description=dict(type='string',minLength=1,maxLength=80),
                      chunks=dict(type='integer',minimum=1,maximum=15,default=4)), ('motion_tags','target_description')),
            tool('behavior_champion_vla', 'Start the published champion fixed-task policy. skill must be in state.vla.champion_skills. It does NOT consume your language. Selecting a different skill is an explicit task-identity transfer, not language following. Each chunk executes the unchanged published wrapper20x23 actions including its corrections. At most6 chunks=120 steps; then fresh images return to you. Continue the returned job, never start duplicates.',
                 dict(skill=dict(type='string',minLength=1,maxLength=100),chunks=dict(type='integer',minimum=1,maximum=6,default=6)), ('skill',)),
            tool('behavior_vla_continue', 'Continue an already authorized VLA job. Wait up to20 seconds for inference while physics is frozen, then execute at most one saved prefix exactly once on the Isaac owner thread. Repeat in a code loop until status=completed (or vla_job.status=completed), then inspect the fresh returned images. Ordinary motion is blocked while pending. This tool can advance physics; it is not a read-only status query.',
                 dict(job_id=dict(type='string',minLength=1,maxLength=80),wait_seconds=dict(type='integer',minimum=0,maximum=20,default=20)), ('job_id',)),
        ]
    if vla_only:
        definitions = [t for t in definitions if t['name'] not in NATIVE_MOTION_TOOLS]
        definitions.append(tool(VLA_HANDOFF,
            'After AP search, record your visual evidence that the target is found and authorize champion execution. No physical action. Requires a fresh observed frame after a completed AP job; this is your assessment, not official success. Calling AP again returns to search and requires a fresh handoff.',
            dict(frame_id=dict(type='string',minLength=1,maxLength=100),
                 target_description=dict(type='string',minLength=1,maxLength=80),
                 reason=dict(type='string',minLength=1,maxLength=2000)),
            ('frame_id','target_description','reason')))
    if vla_only:
        definitions.append(tool(VLA_ASSESS,
            'GPT must inspect the current images after every champion segment and assess its current interaction. completed or need_ap immediately returns ownership to AP; another executed AP job and fresh handoff are required before champion can run again. in_progress permits continuing only the same interaction. This is visual assessment, never official BDDL success.',
            dict(frame_id=dict(type='string',minLength=1,maxLength=100),
                 outcome=dict(type='string',enum=['in_progress','completed','need_ap','unsafe']),
                 evidence=dict(type='string',minLength=1,maxLength=2000)),('frame_id','outcome','evidence')))
    if include_vla:
        for definition in definitions:
            if definition['name'] in VLA_TOOL_NAMES[:2]:
                if definition['name']=='behavior_active_perception_vla':
                    definition['description'] += ' Supported motion_tags combinations only: '+', '.join(ap_combinations)+'. raise_view and lower_view must each be used alone; do not combine them with turning. Unsupported arguments reject before submitting any policy job/action; correct them instead of terminating.'
                definition['inputSchema']['properties'].update(
                    frame_id=dict(type='string',minLength=1,maxLength=100),
                    step_budget=dict(type='integer',minimum=1),
                    visual_progress=dict(type='string',enum=['initial','progress','no_progress','lost_target','unsafe']),
                    reason=dict(type='string',minLength=1,maxLength=2000),
                    interaction_goal=dict(type='string',minLength=1,maxLength=500),
                    next_goal=dict(type='string',minLength=1,maxLength=500))
                definition['description'] += (' In closed_loop mode every start requires frame_id, step_budget (positive physical steps, no per-call maximum; episode remaining budget still applies), visual_progress, reason comparing current images to the last segment, and next_goal. Do not supply chunks in this mode. The exact budget executes only fresh raw action prefixes; unused suffixes are discarded. Each job returns to GPT with new images; no authorization to keep running automatically.')
    return definitions


MUTATING = frozenset({*NATIVE_MOTION_TOOLS, *VLA_TOOL_NAMES, VLA_HANDOFF, VLA_ASSESS})


def validate(name, arguments):
    definitions = tools(include_vla=True) + tools(include_vla=True, vla_only=True)
    definition = next((t for t in definitions if t['name']==name), None)
    if definition is None or not isinstance(arguments,dict):
        raise ValueError('Unknown tool or non-object arguments')
    schema=definition['inputSchema']
    if set(arguments)-schema['properties'].keys() or set(schema['required'])-arguments.keys():
        raise ValueError('Unexpected or missing tool arguments')
    for key,value in arguments.items():
        spec=schema['properties'][key]; kind=spec['type']
        if kind=='number' and (type(value) not in (int,float) or not math.isfinite(value)):
            raise ValueError(f'{key} must be finite numeric')
        if kind=='integer' and type(value) is not int: raise ValueError(f'{key} must be an integer')
        if kind=='string' and not isinstance(value,str): raise ValueError(f'{key} must be text')
        if 'enum' in spec and value not in spec['enum']: raise ValueError(f'Invalid {key}')
        if kind in ('integer','number'):
            if value<spec.get('minimum',-math.inf) or value>spec.get('maximum',math.inf):
                raise ValueError(f'{key} outside tool limits')
        if kind=='string' and not spec.get('minLength',0)<=len(value)<=spec.get('maxLength',256):
            raise ValueError(f'Invalid {key} length')
        if kind=='array':
            if not isinstance(value,list) or not spec['minItems']<=len(value)<=spec['maxItems']:
                raise ValueError(f'Invalid {key} dimensions')
            if spec['items']['type']=='string':
                if any(not isinstance(v,str) or v not in spec['items']['enum'] for v in value):
                    raise ValueError(f'Invalid {key} entries')
                if spec.get('uniqueItems') and len(value)!=len(set(value)):
                    raise ValueError(f'Duplicate {key}')
            elif any(type(v) not in (int,float) or not math.isfinite(v) for v in value):
                raise ValueError(f'Invalid {key} entries')
