"""Offline current-measurement route tests, real TaskSteps and fake MCP owner."""
import json
import math
import cv2
import numpy as np
import pytest
from harness.codex_cli.replay_route import ReplayRoute, BUNDLE
from harness.codex_cli.relative_perception import measure, heading, wrap_degrees
from local_tests.unit.test_g1_task_steps import Context, make, observe, report, mutations


class RouteContext(Context):
    def __init__(self):
        super().__init__()
        self.recipe = json.loads((BUNDLE/'recipe.json').read_text())
        self.index = 0
        self.yaw = 179.
        self.bad_yaw = False
        self.underturn = False
        self.correcting = False
        self.ignore_correction = False
        self.overrides = {}
        self.interrupt_after = None

    def call(self, tool, **args):
        reply = super().call(tool, **args)
        obs = json.loads(reply['content'][0]['text'])
        angle = math.radians(self.yaw)/2
        obs['robot_state'] = dict(measured=True, lowstate_age_s=.01,
                                 base_quat_wxyz=[math.cos(angle),0,0,math.sin(angle)])
        reply['content'][0]['text'] = json.dumps(obs)
        return reply

    def value(self, tool, **args):
        if tool == 'g1_pixel_to_position':
            self.calls.append((tool, args))
            return dict(frame_id=args['frame_id'], position_m=[.1, .2, .8],
                        coordinate_frame='rectified_left_x_right_y_down_z_forward',
                        valid_samples=20, metric_accuracy_validated=False)
        result = super().value(tool, **args)
        if tool == 'g1_move' and args['action'] == 'run':
            if self.correcting:
                delta = 0 if self.ignore_correction else -7.
                if not self.ignore_correction:
                    self.correcting = False
            else:
                delta = self.recipe['steps'][self.index]['measured_yaw_delta_deg']
                delta = self.overrides.get(self.index, delta)
                if self.index == 0 and self.underturn:
                    delta = 5.  # More than 10 degrees from the taught turn target.
                    self.correcting = True
                self.index += 1
            if self.bad_yaw:
                delta = 100.
            self.yaw = wrap_degrees(self.yaw+delta)
            self.jobs[result['job_id']]['measured_yaw_delta_deg'] = delta
            if self.index == self.interrupt_after:
                self.pending = [dict(id='stop', text='暂停')]
        return result


def setup(tmp_path):
    ctx, engine = make(tmp_path, RouteContext())
    route = ReplayRoute(engine, tmp_path)
    route.read = lambda *a: (np.zeros((360,640,3), np.uint8),
                             np.full((360,640), 1200, np.uint16))
    args = dict(action='run', request_id='route1', frame_id=observe(engine),
                evidence='Current UI confirms route including bounded turns and destination arm clearance',
                start_ready=True, route_clear=True, hands_empty=True, arm_clear=True)
    return ctx, engine, route, args


def test_one_call_runs_route_and_prepare_then_current_grasp_review(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    result = report(route.call(args))
    assert result['reason'] == 'route_complete_review_current_grasp'
    assert [(t,a['action']) for t,a in mutations(ctx)] == [('g1_move','run')]*8 + [('g1_vla','prepare')]
    assert result['step_result']['ok']
    assert result['replay']['cursor'] == 8
    assert not result['replay']['task_success_verified']
    assert result['replay']['phase'] == 'await_current_grasp_review'
    assert result['observation']['frame_id'] == result['step_result']['measured']['frame_id']
    # Resume grasp via the existing step interface, without replaying locomotion.
    engine.call(dict(action='grasp', request_id='contact', frame_id=result['observation']['frame_id'],
                     evidence='Current image shows reachable toy and clear approach',
                     target_reachable=True, approach_clear=True))
    assert mutations(ctx)[-1][0] == 'g1_vla'
    assert len(mutations(ctx)) == 10
    assert report(route.call(args))['reason'] == 'route_already_recorded_no_replay'
    _, resumed = make(tmp_path, ctx)
    assert report(ReplayRoute(resumed,tmp_path).call(args))['reason'] == 'route_already_recorded_no_replay'
    assert len(mutations(ctx)) == 10
    with pytest.raises(ValueError, match='different arguments'):
        route.call(args | dict(evidence='changed'))


def test_missing_current_depth_never_moves(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    route.read = lambda *a: (None,None)
    result = report(route.call(args))
    assert result['step_result']['why'] == 'current_rgbd_unavailable'
    assert not mutations(ctx)


def test_unexpected_measured_heading_stops(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    ctx.bad_yaw = True
    assert report(route.call(args))['replay']['reason'] == 'unexpected_heading_response'
    assert len(mutations(ctx)) == 1


def test_underturn_gets_bounded_measured_correction(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    ctx.underturn = True
    result = report(route.call(args))
    assert result['reason'] == 'route_complete_review_current_grasp'
    correction = mutations(ctx)[1][1]
    assert correction['direction'] == 'rotate_right'
    assert correction['duration_s'] <= .5
    assert len(mutations(ctx)) == 10
    assert any(c['why']=='heading_within_travel_tolerance' and c['measured']['attempt']==1
               and c['ok'] for c in result['replay']['checks'])


def test_unresponsive_heading_returns_recovery_without_repeating_weak_turn(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    ctx.underturn = ctx.ignore_correction = True
    result = report(route.call(args))
    assert result['replay']['reason'] == 'heading_correction_stalled'
    assert len(mutations(ctx)) == 2
    assert result['recovery']['next_route_step'] == 1
    assert len(result['recovery']['remaining_commands']) == 7
    assert 'same UI turn' in result['recovery']['instruction']


def test_recorded_six_degree_residual_continues_without_micro_turns(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    # Measured errors from agent-20261020-180739, before its futile corrections.
    first_error, second_error = 3.37755973320742, 5.9769776386207525
    ctx.overrides = {
        0: ctx.recipe['steps'][0]['measured_yaw_delta_deg'] + first_error,
        1: ctx.recipe['steps'][1]['measured_yaw_delta_deg'] + second_error-first_error}
    result = report(route.call(args))
    assert result['reason'] == 'route_complete_review_current_grasp'
    assert len(mutations(ctx)) == 9  # Eight route segments and prepare, no corrections.
    assert not any(':adjust:' in key for key in engine.state['requests'])
    accepted = [c for c in result['replay']['checks']
                if c['why'] == 'heading_within_travel_tolerance']
    assert accepted[1]['measured']['heading_error_deg'] == pytest.approx(-second_error)
    assert accepted[1]['ok'] and not accepted[1]['measured']['precision_target_met']


def test_accepted_residual_is_not_reset_to_hide_accumulating_drift(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    ctx.overrides = {
        0: ctx.recipe['steps'][0]['measured_yaw_delta_deg'] + 8.,
        2: ctx.recipe['steps'][2]['measured_yaw_delta_deg'] + 4.}
    result = report(route.call(args))
    assert result['replay']['reason'] == 'travel_heading_review_required'
    assert len(mutations(ctx)) == 3
    assert result['recovery']['next_route_step'] == 3


def test_new_ui_message_prevents_next_segment(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    ctx.interrupt_after = 1
    assert report(route.call(args))['reason'] == 'replay_checkpoint_required'
    assert len([a for t,a in mutations(ctx) if a['action']=='run']) == 1
    assert all(a.get('action') not in ('shutdown','reset') for t,a in ctx.calls)


@pytest.mark.parametrize('field', ['start_ready','route_clear','hands_empty','arm_clear'])
def test_current_preconditions_required(tmp_path,field):
    ctx, engine, route, args = setup(tmp_path)
    with pytest.raises(ValueError,match='Review current'):
        route.call(args | {field:False})
    assert not mutations(ctx)


def test_budget_does_not_start_or_shutdown(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    ctx.now += 420
    assert report(route.call(args))['reason'] == 'insufficient_remaining_budget_for_route'
    assert not mutations(ctx)


def test_route_deadline_restores_guard(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    original = engine.guard
    original_read = route.read
    def slow_read(*a):
        ctx.now += 181
        return original_read(*a)
    route.read = slow_read
    result = report(route.call(args))
    assert result['replay']['reason'] == 'route_time_budget_exhausted_no_native_stop'
    assert engine.guard is original
    assert not mutations(ctx)


def test_localize_uses_existing_tool_and_reports_only_actual_frame(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    result = report(route.call(dict(action='localize',frame_id=args['frame_id'], selections=[dict(x=300,y=200)])))
    assert result['ok']
    measurement = result['measured']['selections'][0]['measured']
    assert measurement['position_m'] == [.1,.2,.8]
    assert 'position_pelvis_m' not in measurement
    assert not result['measured']['camera_to_robot_extrinsics_validated']
    assert ctx.calls[-1] == ('g1_pixel_to_position',dict(frame_id=args['frame_id'],
        coordinate_frame='camera',radius=2,x=300,y=200))
    assert not mutations(ctx)
    observe(engine)
    with pytest.raises(ValueError,match='latest frame'):
        route.call(dict(action='localize',frame_id=args['frame_id'],selections=[dict(x=300,y=200)]))


def test_localize_invalid_depth_is_structured_failure(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    original = ctx.value
    def missing(tool,**kw):
        if tool == 'g1_pixel_to_position':
            raise ValueError('No valid depth around pixel')
        return original(tool,**kw)
    ctx.value = missing
    result = report(route.call(dict(action='localize',frame_id=args['frame_id'],selections=[dict(x=300,y=200)])))
    assert not result['ok']
    assert 'No valid depth' in result['measured']['selections'][0]['why']
    assert not mutations(ctx)


def test_current_geometry_has_no_historical_image_dependency(tmp_path):
    ctx, engine, route, args = setup(tmp_path)
    # Any old frame identifier is irrelevant to controller execution.
    for step in route.recipe['steps']:
        step['before'] = step['after'] = 'not-a-real-frame'
    route.recipe['prepared_frame'] = 'not-a-real-frame'
    assert report(route.call(args))['reason'] == 'route_complete_review_current_grasp'


def test_historical_real_frame_can_be_measured_without_reference():
    recipe = json.loads((BUNDLE/'recipe.json').read_text())
    folder = BUNDLE/recipe['prepared_frame']
    rgb = cv2.imread(str(folder/'rgb.png'))
    depth = cv2.imread(str(folder/'depth.png'),cv2.IMREAD_UNCHANGED)
    observation = json.loads((folder/'meta.json').read_text())
    result = measure(rgb,depth,observation)
    assert result['ok']
    assert not result['measured']['position_feedback_available']
    observation['robot_state']['lowstate_age_s'] = 2.
    assert not measure(rgb,depth,observation)['ok']


def test_heading_wrap_and_invalid_orientation():
    assert wrap_degrees(-358) == 2
    with pytest.raises(ValueError):
        heading(dict(robot_state=dict(measured=True,lowstate_age_s=0,base_quat_wxyz=[0,0,0,0])))
