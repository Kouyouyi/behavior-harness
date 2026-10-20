"""Synchronous chunk execution: no physics ticks while inference is pending."""
import json
from pathlib import Path
import time
import numpy as np

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

def run_loop(sim, planner, policy, goal, output, max_rounds=100, max_observation_requests=3, planner_max_chunks=1, opening_commit_chunks=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    history = []
    requests = 0
    reason = 'round_limit'
    reached_claim = False
    navigation_success = False
    cached = None
    remaining = 0
    evidence_step = None
    planner_calls = 0
    health = policy.health()
    if health.get('ready') is not True or any(health.get(k) != v for k, v in
            [('state_dim', 61), ('action_dim', 23), ('prediction_steps', 32)]):
        raise ValueError('Remote VLA health contract mismatch')
    save_json(output / 'remote_health.json', health)
    try:
        for round_id in range(max_rounds):
            if sim.done:
                reason = sim.termination_reason
                break
            directory = output / f'round_{round_id:05d}'
            directory.mkdir()
            observation = sim.capture(directory)
            step = sim.steps
            reused = remaining > 0
            if reused:
                result = cached
                (directory/'planner').mkdir()
                save_json(directory/'planner/result.json', {**result,'seconds':0,
                    'reused_instruction':True,'evidence_observation_step':evidence_step})
                remaining -= 1
            else:
                result = planner(observation['image_paths'], goal, history, directory / 'planner')
                planner_calls += 1
                evidence_step = step
                tags = result['command']['motion_tags']
                limit = opening_commit_chunks if any(t.startswith('open_') for t in tags) else (
                    min(planner_max_chunks,4) if tags in (['turn_left'],['turn_right']) else min(planner_max_chunks,2))
                remaining = limit - 1 if result['command']['status'] == 'execute' else 0
                cached = result
            if sim.done:
                reason = sim.termination_reason
                break
            command = result['command']
            # Validate again at the control boundary, even for alternative planners.
            from planner_contract import render
            instruction = render(command)
            if result['instruction'] != instruction:
                raise ValueError('Planner instruction differs from the training renderer')
            history.append({'step': step, 'command': command, 'executed_steps': 0,
                            'reused_instruction':reused, 'evidence_observation_step':evidence_step})
            if command['status'] == 'reached':
                reached_claim = True
                if hasattr(sim,'save_handoff'):
                    sim.save_handoff(output.parent/'task_handoff')
                # Terminal-only audit: never resume physics after enabling segmentation.
                evidence = sim.verify_approach(directory / 'terminal_audit')
                navigation_success = evidence['find_and_approach_success'] is True
                reason = 'find_and_approach_success' if navigation_success else 'arrival_verification_failed'
                break
            if command['status'] == 'need_observation':
                requests += 1
                if requests >= max_observation_requests:
                    reason = 'observation_request_limit'
                    break
                sim.refresh()
                continue
            requests = 0
            from .preprocessing import prepare_text
            _, _, prompt, encoded = prepare_text(observation['state'], instruction)
            (directory / 'vla_prompt.txt').write_text(prompt)
            save_json(directory / 'text_validation.json', {'tokens_before_padding': len(encoded.ids),
                                                         'truncated': False})
            now = policy.health()
            if now != health:
                raise RuntimeError('Remote model identity/health changed during the episode')
            started = time.monotonic()
            actions = np.asarray(policy.predict(observation, instruction), dtype=np.float32)
            if actions.shape != (32, 23) or not np.isfinite(actions).all():
                raise ValueError('Expected finite raw 32x23 action chunk')
            np.save(directory / 'predicted_actions.npy', actions)
            save_json(directory / 'prediction.json', {'instruction': instruction,
                      'observation_step': step, 'seconds': time.monotonic() - started,
                      'representation': 'original_R1Pro_controller_inputs', 'execute_prefix': 8,
                      'remaining_predictions_discarded': 24})
            if sim.done:
                reason = sim.termination_reason
                break
            if sim.steps != step:
                raise RuntimeError('Simulation advanced during inference; refusing stale actions')
            with (directory / 'execution.jsonl').open('w', buffering=1) as log:
                for offset, action in enumerate(actions[:8]):
                    if sim.done:
                        break
                    log.write(json.dumps({'event': 'submitted', 'offset': offset,
                              'step': sim.steps, 'time_ns': time.time_ns(), 'action': action.tolist()}) + '\n')
                    # A failed/partial step is never automatically replayed.
                    if not sim.step(action):
                        break
                    history[-1]['executed_steps'] += 1
                    log.write(json.dumps({'event': 'ack', 'offset': offset,
                              'step': sim.steps, 'time_ns': time.time_ns()}) + '\n')
            sim.capture(directory / 'after')
            save_json(output / 'history.json', history)
            if sim.done:
                reason = sim.termination_reason
                break
    except BaseException as exc:
        reason = 'infrastructure_error'
        save_json(output / 'error.json', {'type': type(exc).__name__, 'message': str(exc),
                  'automatic_replay': False})
        raise
    finally:
        save_json(output / 'summary.json', {'reason': reason, 'steps': sim.steps,
                  'official_task_success': sim.success, 'planner_rounds': len(history),
                  'perception_reached': reached_claim,
                  'find_and_approach_success': navigation_success,
                  'objective': 'find_and_approach',
                  'planner_calls':planner_calls, 'planner_max_chunks':planner_max_chunks,
                  'opening_commit_chunks':opening_commit_chunks,
                  'physics_hz': 30, 'simulation_wait_mode': 'synchronous_no_step',
                  'history': history})
    return reason
