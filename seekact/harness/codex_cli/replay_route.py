"""Reusable route primitives with current measurements and resumable UI checkpoints.

Official-style observation/decision/action steps return {ok, measured, why}.
Translation remains taught primitives: this real camera has no calibrated body
extrinsics or odometry. A UI checkpoint verifies the CURRENT grasp conditions.
"""
import json
import math
from pathlib import Path
import re
import cv2
from .relative_perception import measure, heading, wrap_degrees, turn_correction

NAME = 'g1_replay_route'
BUNDLE = Path(__file__).resolve().parents[2]/'data/tasks/mickey_grasp/routes/mickey_right_side_v1'
# Route travel, not a precision contact tolerance. 5 degrees is a diagnostic
# target; the ~6-degree 2026-10-20 trial must not trigger futile micro-turns.
# 10 degrees is a commissioning bound, not a hardware-validated safety margin.
HEADING_TARGET_DEG = 5.
TRAVEL_HEADING_TOLERANCE_DEG = 10.
MIN_CORRECTION_PROGRESS_DEG = 1.
MAX_CORRECTIONS = 2
SCHEMA = dict(name=NAME, description=(
    'Current-observation route controller. preview returns live image and measurements; '
    'localize back-projects up to 32 CURRENT frame pixels through g1_pixel_to_position '
    'to camera optical metres (not pelvis/world coordinates). run chains eight taught '
    'right-side route primitives, remeasures orientation and corrects turn heading with '
    'travel errors up to 10 degrees are recorded and accepted, not micro-corrected. '
    'Larger errors receive at most two bounded corrections; a correction with less '
    'than 1 degree improvement returns a recovery checkpoint instead of repeating it. '
    'Translation is not position-servoed. '
    'No historical image/depth matching. Start area/heading, empty hands, whole route '
    'and destination arm clearance require current evidence. After prepare, returns a '
    'CURRENT grasp-review checkpoint: inspect image, localize target if helpful, then '
    'use g1_task_step grasp/check/pause/confirm_hold; never rerun the route to resume. '
    'Steps return ok/measured/why. 180-second route budget; new messages pause owned '
    'upper-level action, never native shutdown.'),
    inputSchema=dict(type='object', additionalProperties=False, required=['action'],
        properties=dict(action=dict(type='string', enum=['preview', 'localize', 'run']),
            request_id=dict(type='string'), frame_id=dict(type='string'), evidence=dict(type='string'),
            start_ready=dict(type='boolean', description='Current starting area/heading suit this right-side route; no image matching.'),
            start_matches=dict(type='boolean', description='Deprecated alias for start_ready; never means pixel matching.'),
            route_clear=dict(type='boolean', description='Current route clearance allows coarse travel heading (up to 10 degrees residual) and bounded correction turns.'),
            hands_empty=dict(type='boolean'), arm_clear=dict(type='boolean'),
            selections=dict(type='array', minItems=1, maxItems=32, items=dict(type='object',
                additionalProperties=False, required=['x','y'], properties=dict(
                    x=dict(type='integer',minimum=0,maximum=639),
                    y=dict(type='integer',minimum=0,maximum=359)))))))


class RouteCheckpoint(RuntimeError):
    pass


class ReplayRoute:
    def __init__(self, engine, run, *, bundle=BUNDLE):
        self.engine = engine
        self.workspace = Path(run)/'session/agent-workspace/runtime/frames'
        self.recipe = json.loads((Path(bundle)/'recipe.json').read_text())
        if len(self.recipe['steps']) != 8 or self.recipe['max_seconds'] != 180:
            raise ValueError('Unexpected route recipe')

    def read(self, ident):
        if not isinstance(ident, str) or not re.fullmatch('[0-9a-f]{32}', ident):
            raise ValueError('Invalid observation frame')
        return (cv2.imread(str(self.workspace/ident/'rgb.png')),
                cv2.imread(str(self.workspace/ident/'depth.png'), cv2.IMREAD_UNCHANGED))

    def latest(self):
        frames = self.engine.state['frames']
        if not frames:
            raise RouteCheckpoint('No live image')
        ident = next(reversed(frames))
        self.engine.frame(ident)
        return ident

    def check_current(self, frame_id, record):
        self.engine.guard()
        observation = self.engine.frame(frame_id)
        result = measure(*self.read(frame_id), observation)
        self.engine.guard()
        record['checks'].append(result)
        self.engine.save()
        if not result['ok']:
            raise RouteCheckpoint(result['why'])
        return result

    def localize(self, args):
        frame_id, selections = args.get('frame_id'), args.get('selections')
        self.engine.guard()
        self.engine.update()
        if (self.engine.state['active'] or frame_id != self.latest()
                or frame_id in self.engine.state['used_frames']):
            raise ValueError('Observe after motion and select pixels from the latest frame')
        observation = self.engine.frame(frame_id)
        if not isinstance(selections, list) or not 1 <= len(selections) <= 32:
            raise ValueError('Select 1–32 pixels in the current frame')
        for point in selections:
            if (not isinstance(point, dict) or set(point) != {'x', 'y'} or
                    any(type(point[k]) is not int or not 0 <= point[k] < limit
                        for k, limit in [('x',640),('y',360)])):
                raise ValueError('Selections require current original-image integer x/y')
        results = []
        for point in selections:
            self.engine.guard()
            self.engine.frame(frame_id)
            try:
                value = self.engine.value('g1_pixel_to_position', frame_id=frame_id,
                                          coordinate_frame='camera', radius=2, **point)
                xyz = value['position_m']
                if (len(xyz) != 3 or not all(math.isfinite(v) for v in xyz)
                        or xyz[2] <= 0 or value.get('frame_id') != frame_id):
                    raise ValueError('Invalid current pixel geometry')
                value['bearing_optical_deg'] = math.degrees(math.atan2(xyz[0], xyz[2]))
                results.append(dict(ok=True, measured=value, why='current_camera_surface_point'))
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                results.append(dict(ok=False, measured=point, why=str(exc)))
        return dict(content=[dict(type='text', text=json.dumps(dict(
            ok=all(r['ok'] for r in results), measured=dict(frame_id=frame_id,
                source_frame_id=observation.get('source_frame_id'), selections=results,
                camera_to_robot_extrinsics_validated=False,
                robot_camera_hardware_synchronized=False),
            why='camera_points_only_not_body_positions_or_reachability')))])

    def preview(self):
        reply = self.engine.output('review_current_start_and_whole_route', route=dict(
            id=self.recipe['id'], steps=[s['command'] for s in self.recipe['steps']],
            validation=self.recipe['verification'], max_seconds=180,
            scene_check='current_measurements_no_historical_comparison',
            travel_heading_tolerance_deg=TRAVEL_HEADING_TOLERANCE_DEG,
            translation_feedback='unavailable', final_step='current_grasp_review'))
        report = json.loads(reply['content'][0]['text'])
        if report.get('observation'):
            report['step_result'] = measure(*self.read(report['observation']['frame_id']), report['observation'])
            reply['content'][0]['text'] = json.dumps(report)
        return reply

    def move(self, command, request_id, current, evidence, record):
        self.engine.guard()
        self.check_current(current, record)
        reply = self.engine.call(dict(action='move', request_id=request_id,
            frame_id=current, path_clear=True, evidence=evidence, **command))
        result = self.engine.state.get('last_result') or {}
        report = json.loads(reply['content'][0]['text'])
        ok = (report['reason'] == 'action_complete_review_image' and report.get('observation')
              and result.get('completed') and result.get('settled') and not self.engine.state['active'])
        record['checks'].append(dict(ok=bool(ok), measured=result, why='movement_completion'))
        self.engine.save()
        if not ok:
            raise RouteCheckpoint('movement_not_complete_and_settled')
        current = self.latest()
        self.check_current(current, record)
        return current

    def call(self, args):
        if set(args)-SCHEMA['inputSchema']['properties'].keys():
            raise ValueError('Unexpected route arguments')
        if args.get('action') == 'preview':
            return self.preview()
        if args.get('action') == 'localize':
            return self.localize(args)
        if args.get('action') != 'run':
            raise ValueError('Choose preview, localize or run')
        ident, evidence = args.get('request_id'), args.get('evidence')
        if not isinstance(ident, str) or not 1 <= len(ident) <= 64:
            raise ValueError('Unique route request_id required')
        if not isinstance(evidence, str) or not 1 <= len(evidence.strip()) <= 1200:
            raise ValueError('Current UI/visual evidence required')
        if ('start_ready' in args and 'start_matches' in args
                and args['start_ready'] != args['start_matches']):
            raise ValueError('Conflicting start conditions')
        if (args.get('start_ready', args.get('start_matches')) is not True or
                any(args.get(k) is not True for k in ('route_clear','hands_empty','arm_clear'))):
            raise ValueError('Review current start, whole right-side path, empty hands and arm clearance')
        engine = self.engine
        records = engine.state.setdefault('replays', {})
        if ident in records:
            if records[ident]['arguments'] != args:
                raise ValueError('Route ID reused with different arguments')
            return engine.output('route_already_recorded_no_replay', replay=records[ident])
        if engine.remaining() < 185:
            return engine.output('insufficient_remaining_budget_for_route')
        if engine.state['active'] or any(r['phase']=='submitting' for r in engine.state['requests'].values()):
            raise ValueError('Existing or uncertain action; inspect before replay')
        record = dict(arguments=args, phase='checking', cursor=0, checks=[],
                      started_monotonic=engine.clock(), task_success_verified=False)
        records[ident] = record
        engine.save()
        original_guard = engine.guard
        def guard():
            original_guard()
            if engine.clock()-record['started_monotonic'] >= 180:
                raise RouteCheckpoint('route_time_budget_exhausted_no_native_stop')
        engine.guard = guard
        try:
            current = args.get('frame_id')
            self.check_current(current, record)
            goal = heading(engine.frame(current))
            for index, step in enumerate(self.recipe['steps']):
                guard()
                if engine.value('g1_operator_messages', action='read').get('pending'):
                    raise RouteCheckpoint('new_operator_message')
                record.update(phase='moving', cursor=index, movement_completed=False,
                              next_route_step=index)
                # Never reset the target to an accepted off-target heading:
                # small residuals remain visible and cannot silently accumulate.
                if (not step['command']['direction'].startswith('rotate_') and
                        abs(wrap_degrees(goal-heading(engine.frame(current)))) >
                        TRAVEL_HEADING_TOLERANCE_DEG):
                    raise RouteCheckpoint('travel_heading_review_required')
                goal = wrap_degrees(goal+step['measured_yaw_delta_deg'])
                before = heading(engine.frame(current))
                current = self.move(step['command'], f'{ident}:move:{index}', current, evidence, record)
                record.update(movement_completed=True, next_route_step=index+1)
                engine.save()
                measured = wrap_degrees(heading(engine.frame(current))-before)
                if abs(wrap_degrees(measured-step['measured_yaw_delta_deg'])) > 20:
                    raise RouteCheckpoint('unexpected_heading_response')
                if step['command']['direction'].startswith('rotate_'):
                    previous_error = None
                    for attempt in range(MAX_CORRECTIONS+1):
                        error = wrap_degrees(goal-heading(engine.frame(current)))
                        ok = abs(error) <= TRAVEL_HEADING_TOLERANCE_DEG
                        progress = None if previous_error is None else abs(previous_error)-abs(error)
                        record['checks'].append(dict(ok=ok, measured=dict(goal_heading_deg=goal,
                            heading_error_deg=error, attempt=attempt,
                            correction_progress_deg=progress,
                            precision_target_met=abs(error) <= HEADING_TARGET_DEG,
                            travel_tolerance_deg=TRAVEL_HEADING_TOLERANCE_DEG),
                            why='heading_within_travel_tolerance' if ok else 'heading_goal'))
                        engine.save()
                        if ok:
                            break
                        if progress is not None and progress < MIN_CORRECTION_PROGRESS_DEG:
                            raise RouteCheckpoint('heading_correction_stalled')
                        if attempt == MAX_CORRECTIONS:
                            raise RouteCheckpoint('heading_review_required')
                        previous_error = error
                        current = self.move(turn_correction(error, step), f'{ident}:adjust:{index}:{attempt}',
                                            current, evidence, record)
                record['cursor'] = index+1
                engine.save()
            guard()
            record['phase'] = 'preparing'
            reply = engine.call(dict(action='prepare', request_id=f'{ident}:prepare',
                frame_id=current, hands_empty=True, arm_clear=True, evidence=evidence))
            report = json.loads(reply['content'][0]['text'])
            if report['reason'] != 'action_complete_review_image' or not report.get('observation'):
                raise RouteCheckpoint('prepare_not_complete')
            current = self.latest()
            measurement = self.check_current(current, record)
            record.update(phase='await_current_grasp_review',
                          elapsed_seconds=engine.clock()-record['started_monotonic'])
            engine.save()
            # Return the just-measured frame, not a different camera frame whose
            # coordinates could accidentally be paired with these measurements.
            report.update(reason='route_complete_review_current_grasp', replay=record,
                step_result=dict(ok=True, measured=measurement['measured'], why='route_and_prepare_complete'),
                next_action='Inspect CURRENT target/hand geometry; localize current pixels if useful, '
                            'then g1_task_step grasp if reachable and clear. Otherwise adjust via '
                            'g1_task_step; movement requires prepare again. Never rerun the route.')
            reply['content'][0]['text'] = json.dumps(report)
            return reply
        except Exception as exc:
            record.update(phase='checkpoint_required', reason=str(exc),
                          elapsed_seconds=engine.clock()-record['started_monotonic'])
            try:
                engine.pause_owned()
            except Exception as pause_error:
                record['pause_error'] = str(pause_error)
            engine.save()
            recovery = dict(
                next_route_step=record.get('next_route_step', record['cursor']),
                remaining_commands=[s['command'] for s in self.recipe['steps'][
                    record.get('next_route_step', record['cursor']):]],
                instruction='Review the NEW image and measured heading in this same UI turn. '
                    'A checkpoint is not task completion. If current route/target geometry '
                    'supports continuation, use g1_task_step from the current pose; otherwise '
                    'choose a different bounded maneuver with current evidence. Do not repeat '
                    'an ineffective correction or rerun the route from its start. Remaining '
                    'commands are planning context, not permission for blind execution.')
            return engine.output('replay_checkpoint_required', replay=record,
                                 recovery=recovery,
                                 step_result=dict(ok=False, measured=dict(cursor=record['cursor']), why=str(exc)))
        finally:
            engine.guard = original_guard
