"""Repeatable RAX controller: observe, preview, submit, verify feedback, observe.

Only the standard agent Context is used. No SDK, motor, camera or network access.
"""
import time


def main(ctx, side='right', delta_rad=.03, action='elbow_pulse',delta_m=None,rotation_delta=None):
    before = ctx.value('g1_observe')
    status = ctx.value('g1_status')
    if not status['motion_available'] or before.get('robot_state') is None:
        raise RuntimeError('Fresh measured standing reference is required')
    preview = (ctx.value('g1_plan_walk') if action=='walk_step' else
               ctx.value('g1_plan_grasp_step',delta_m=delta_m,rotation_delta=rotation_delta) if action=='grasp_step' else
               ctx.value('g1_plan_arm_motion', side=side, delta_rad=delta_rad) if action=='elbow_pulse' else
               ctx.value('g1_plan_reach',action=action))
    submitted = ctx.value('g1_execute_reference', plan_id=preview['plan_id'])
    deadline = time.monotonic()+(22 if action=='elbow_pulse' else 60)
    while True:
        result = ctx.value('g1_reference_result', job_id=submitted['job_id'])
        if 'completed' in result:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Reference outcome uncertain; do not replay automatically')
        time.sleep(.1)
    after = ctx.value('g1_observe')
    return dict(before_frame=before['frame_id'], after_frame=after['frame_id'],
                preview=preview, result=result,
                measured_roundtrip_passed=bool(result['completed'] and (action=='reach_return' or
                    (action=='elbow_pulse' and result.get('measured_response_detected') and result.get('measured_body_return_within_0_05_rad')))),
                clear_stage_passed=action=='clear_arms' and bool(result['completed']),
                endpoint_step_passed=action=='grasp_step' and bool(result['completed']),grasp_verified=False)
