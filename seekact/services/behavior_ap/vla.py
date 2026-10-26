"""Learned-policy jobs. HTTP runs in a background thread; Isaac never does.

Each continue request admits at most one saved model prefix on the owner thread.
Model waits can retry indefinitely without holding an MCP request open forever.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

AP_CHECKPOINT = os.environ.get('SEEKACT_AP_CHECKPOINT','/cs/kyy/behavior_ap_experiment/runs/pi05_goal_horizon_v2_b8/checkpoint_024000')
CHAMPION_REVISION = '89545bc1b7aa7f2e687bc0032d091f132d715d4e'
WORDS = {'turn_left': '向左转', 'turn_right': '向右转', 'forward': '向前移动',
         'backward': '向后移动', 'strafe_left': '向左平移', 'strafe_right': '向右平移',
         'raise_view': '抬高观察位置', 'lower_view': '降低观察位置'}
OPEN = {'open_door': '打开', 'open_drawer': '拉开', 'open_lid': '揭开'}
MOTIONS = tuple(WORDS) + tuple(OPEN)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def render_ap(tags, target):
    combinations = json.loads(Path(__file__).with_name('ap_capabilities.json').read_text())['combination_counts']
    if (not isinstance(tags, list) or len(tags) != len(set(tags))
            or frozenset(tags) not in {frozenset(k.split('+')) for k in combinations}):
        raise ValueError('Unsupported AP tag combination. No policy request or physical action was submitted; correct the arguments using supported combinations: '+', '.join(combinations))
    if not isinstance(target, str) or not 1 <= len(target.strip()) <= 80:
        raise ValueError('Use a short visible target/region noun phrase')
    if re.search(r'(?:[0-9]+(?:\.[0-9]+)?|[一二三四五六七八九十百]+)\s*(?:度|厘米|毫米|米)', target):
        raise ValueError('AP language does not accept metric angle/distance parameters')
    tags = sorted(tags, key=MOTIONS.index)
    if tags[0] in OPEN:
        return OPEN[tags[0]] + target + '，观察' + target + '内部。'
    return '并'.join(WORDS[t] for t in tags) + '，观察并接近' + target + '。'


class RemotePolicy:
    """One serialized policy client per worker, with pinned model identities."""
    def __init__(self, kind, options, output, episode=None):
        self.kind, self.options, self.output = kind, options, Path(output)
        self.url = options['ap_url' if kind == 'ap' else 'champion_url'].rstrip('/')
        self.episode, self.seq = episode or uuid.uuid4().hex, 0
        self.identity = None
        self.opener = build_opener(ProxyHandler({}))

    def fetch(self, path, payload, stop, directory):
        attempt = 0
        while not stop.is_set():
            start = time.perf_counter_ns()
            try:
                req = Request(self.url + path, data=payload, headers={'Content-Type': 'application/json'})
                with self.opener.open(req, timeout=25) as response:
                    value = json.load(response)
                self.log(directory, path, attempt, start, 'response')
                return value
            except HTTPError as exc:
                body = exc.read().decode(errors='replace')[:3000]
                self.log(directory, path, attempt, start, 'http_error', code=exc.code, detail=body)
                if exc.code not in (408, 429, 500, 502, 503, 504) or 'requires_restart' in body:
                    raise RuntimeError(f'Policy protocol/state error {exc.code}: {body}') from exc
            except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
                self.log(directory, path, attempt, start, 'waiting_for_model', error=type(exc).__name__)
            attempt += 1
            # Same immutable POST bytes, including champion episode/sequence.
            stop.wait(min(10, attempt))
        raise RuntimeError('Policy request cancelled before any further physical execution')

    def log(self, directory, path, attempt, start, status, **extra):
        with (directory / 'http_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(path=path, attempt=attempt, status=status,
                started_perf_counter_ns=start, ended_perf_counter_ns=time.perf_counter_ns(), **extra)) + '\n')

    def predict(self, observation, args, epoch, directory, stop):
        health = self.fetch('/health', None, stop, directory)
        if self.kind == 'ap':
            expected = {'ready': True, 'checkpoint': self.options.get('ap_checkpoint', AP_CHECKPOINT), 'state_dim': 61,
                        'action_dim': 23, 'prediction_steps': 32}
            if any(health.get(k) != v for k, v in expected.items()):
                raise ValueError('AP checkpoint/state/action contract mismatch')
        elif (health.get('role') != 'seekact_champion_tools' or not health.get('ready')
              or health.get('revision') != CHAMPION_REVISION or health.get('checkpoint') != 'checkpoint_2'
              or health.get('supported_skills', {}).get(args['skill']) != self.options['champion_skills'][args['skill']]):
            raise ValueError('Champion task/checkpoint contract mismatch')
        if self.identity is None:
            self.identity = health
        elif health != self.identity:
            raise ValueError('Policy identity changed during this physical episode')
        save(directory / 'remote_health.json', health)
        body = {'state': observation['state'], 'images_base64': [
            base64.b64encode(Path(p).read_bytes()).decode() for p in observation['image_paths']]}
        if self.kind == 'ap':
            body['instruction'] = args['instruction']
        else:
            body.update(episode=self.episode, seq=self.seq, context_epoch=epoch, skill=args['skill'])
        payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        # Preserve exact bytes before network submission; never recreate on retry.
        (directory / 'request.json').write_bytes(payload)
        save(directory / 'request_identity.json', {'sha256': hashlib.sha256(payload).hexdigest(),
            'observation_step': observation['step'], 'episode': self.episode,
            'sequence': self.seq if self.kind == 'champion' else None})
        result = self.fetch('/predict', payload, stop, directory)
        expected = ('original_R1Pro_controller_inputs' if self.kind == 'ap'
                    else 'published_champion_wrapper_R1Pro_controller_inputs')
        if result.get('representation') != expected:
            raise ValueError('Policy action representation changed')
        if self.kind == 'champion':
            if (result.get('boot_id') != health['boot_id'] or result.get('seq') != self.seq
                    or result.get('episode') != self.episode or result.get('skill') != args['skill']
                    or result.get('context_epoch') != epoch):
                raise ValueError('Champion response/episode/sequence identity mismatch')
            self.seq += 1
        save(directory / 'response.json', result)
        return result


class VlaTools:
    def __init__(self, worker, options):
        self.worker, self.options = worker, options
        self.root = worker.output / 'vla'
        self.root.mkdir(exist_ok=False)
        self.clients = {kind: RemotePolicy(kind, options, self.root) for kind in ('ap', 'champion')}
        self.current = None
        self.count = 0
        self.epoch = 0
        self.infrastructure_errors = []
        self.history = []
        self.phase = 'search' if options.get('vla_only') else 'flexible'
        self.handoff = None
        self.handoffs = []
        self.decisions = []
        self.interaction_assessments=[]
        self.awaiting_interaction_assessment=False
        self.ap_required_after_step=None
        self.active_interaction_goal=None
        self.interaction_unsafe=False
        from .qwen_monitor import QwenMonitor, CompletionVotes
        self.qwen = QwenMonitor(options['qwen_monitor']) if options.get('qwen_monitor') else None
        self.qwen_votes = CompletionVotes()
        self.qwen_checks = []

    def pending(self):
        return self.current is not None and not self.current['done']

    def state(self):
        return {'enabled': True, 'ap_checkpoint': AP_CHECKPOINT,
            'champion_revision': CHAMPION_REVISION, 'champion_skills': self.options['champion_skills'],
            'champion_language_input': False, 'champion_cross_task_transfer': self.options.get('champion_cross_task_transfer', False),
            'vla_only': bool(self.options.get('vla_only')), 'native_motion_enabled': not self.options.get('vla_only',False),
            'phase': self.phase, 'handoff': self.handoff,
            'closed_loop': bool(self.options.get('closed_loop')),
            'qwen_monitor_enabled':self.qwen is not None,
            'qwen_completion_streak':self.qwen_votes.streak,
            'interaction_only':bool(self.options.get('interaction_only')),
            'awaiting_interaction_assessment':self.awaiting_interaction_assessment,
            'active_interaction_goal':self.active_interaction_goal,
            'ap_required_after_step':self.ap_required_after_step,
            'recent_decisions': self.decisions[-5:],
            'ap_supported_tag_combinations': list(json.loads(Path(__file__).with_name('ap_capabilities.json').read_text())['combination_counts']),
            'job': self.summary() if self.current else None,
            'infrastructure_errors': list(self.infrastructure_errors)}

    def summary(self):
        j = self.current
        return {k: j[k] for k in ('job_id', 'kind', 'args', 'chunks_requested', 'chunks_completed',
                                  'executed_steps', 'start_step', 'done', 'status', 'step_budget')}

    def start(self, kind, args):
        if self.pending() or self.infrastructure_errors:
            raise RuntimeError('A policy job is pending or invalid; continue it or finish blocked')
        args = dict(args)
        if self.options.get('interaction_only') and self.interaction_unsafe:
            raise ValueError('GPT assessed an unsafe interaction; no more policy motion is admitted')
        if self.options.get('interaction_only') and self.awaiting_interaction_assessment:
            raise ValueError('Inspect fresh images and call behavior_assess_interaction before any further policy job')
        if kind=='champion' and self.options.get('interaction_only'):
            goal=args.get('interaction_goal','').strip()
            if not goal:raise ValueError('Champion needs one concrete local interaction_goal; AP handles travel')
            if self.active_interaction_goal and goal!=self.active_interaction_goal:
                raise ValueError('Finish/assess current interaction and return to AP before selecting a different interaction')
        if kind == 'champion' and self.options.get('vla_only') and (self.phase != 'champion' or self.handoff is None):
            raise ValueError('Complete AP search and a fresh visual handoff before champion execution')
        if kind == 'ap':
            args['instruction'] = render_ap(args['motion_tags'], args['target_description'])
        elif args['skill'] not in self.options['champion_skills']:
            raise ValueError('This champion skill is not authorized for the current task')
        count = args.get('chunks', 4 if kind == 'ap' else 6)
        if type(count) is not int or not 1 <= count <= (15 if kind == 'ap' else 6):
            raise ValueError('VLA tool is bounded to at most120 physical steps')
        quantum = 8 if kind == 'ap' else 20
        budget = count * quantum
        decision = None
        if self.options.get('closed_loop'):
            required = {'frame_id','step_budget','visual_progress','reason','next_goal'}
            if not required <= args.keys() or 'chunks' in args:
                raise ValueError('Closed-loop start requires fresh frame, explicit step_budget, visual_progress, reason and next_goal; omit chunks')
            frame = self.worker.frames.get(args['frame_id'])
            if frame is None or frame['step'] != self.worker.env.steps:
                raise ValueError('Closed-loop decision requires a fresh current visual frame')
            budget = args['step_budget']
            if type(budget) is not int or budget < 1:
                raise ValueError('step_budget must be a positive physical step count')
            limit=self.worker.config.get('max_steps',10000)
            if limit is not None:budget = min(budget, max(0, limit-self.worker.env.steps))
            if budget == 0:
                raise ValueError('Episode physical budget exhausted')
            if args['visual_progress'] not in ('initial','progress','no_progress','lost_target','unsafe') or not args['reason'].strip() or not args['next_goal'].strip():
                raise ValueError('Describe actual visual progress and the next subgoal')
            if args['visual_progress']=='unsafe':
                raise ValueError('Unsafe assessment cannot authorize motion; observe or finish blocked with safety evidence')
            recent = self.decisions[-3:]
            if kind=='champion' and len(recent)==3 and all(d['kind']=='champion' and d['visual_progress'] in ('no_progress','lost_target') for d in recent) and args['visual_progress'] in ('no_progress','lost_target'):
                raise ValueError('Repeated champion stagnation: reassess and use AP for a safe view/approach recovery before further champion execution')
            count = (budget + quantum - 1) // quantum
            decision = dict(kind=kind, step=self.worker.env.steps,
                **{key:args[key] for key in required})
        if kind == 'ap' and self.options.get('vla_only'):
            self.phase='search';self.handoff=None
            self.active_interaction_goal=None
        if kind == 'ap':
            self.qwen_votes.reset()
            self.epoch += 1  # Only admitted AP ownership invalidates champion history.
        if kind=='champion' and self.options.get('interaction_only'):
            self.active_interaction_goal=args['interaction_goal']
        self.count += 1
        self.current = dict(job_id=f'job_{self.count:05d}', kind=kind, args=args,
            chunks_requested=count, chunks_completed=0, executed_steps=0, step_budget=budget,
            start_step=self.worker.env.steps, done=False, status='waiting_for_model', stop=threading.Event())
        if decision is not None:
            decision['job_id']=self.current['job_id']
            self.decisions.append(decision)
            save(self.root / ('decision_%05d.json' % self.count), decision)
        self._launch()
        return self.summary(), []

    def authorize_champion(self, frame_id, target_description, reason):
        if not self.options.get('vla_only'):
            raise ValueError('Explicit handoff is used only by VLA-only scheduling')
        if self.pending() or self.infrastructure_errors:
            raise RuntimeError('Finish the pending AP job before visual handoff')
        frame=self.worker.frames.get(frame_id)
        if frame is None or frame['step'] != self.worker.env.steps:
            raise ValueError('Handoff requires a fresh current visual observation')
        if not any(j['kind']=='ap' and j['executed_steps']>0 for j in self.history):
            raise ValueError('AP must execute search actions before champion handoff')
        if self.options.get('interaction_only'):
            if self.awaiting_interaction_assessment:raise ValueError('Assess the last champion interaction first')
            if self.ap_required_after_step is not None and not any(j['kind']=='ap' and j['start_step']>=self.ap_required_after_step and j['executed_steps']>0 for j in self.history):
                raise ValueError('Completed interaction requires executed AP navigation/view/approach before another champion handoff')
        self.handoff=dict(frame_id=frame_id,target_description=target_description,reason=reason,
            step=self.worker.env.steps,source='planner_visual_assessment',official_success_claim=False)
        self.handoffs.append(self.handoff);self.phase='champion'
        save(self.root/f'handoff_{len(self.handoffs):05d}.json',self.handoff)
        return {'accepted':True,'phase':self.phase,'handoff':self.handoff,'physical_steps_added':0}

    def assess_interaction(self, frame_id, outcome, evidence):
        if not self.options.get('interaction_only') or not self.awaiting_interaction_assessment:
            raise ValueError('No completed champion segment awaiting GPT assessment')
        if self.pending() or self.infrastructure_errors:raise RuntimeError('Cannot assess pending or uncertain execution')
        frame=self.worker.frames.get(frame_id)
        if frame is None or frame['step']!=self.worker.env.steps:
            raise ValueError('Interaction assessment requires a fresh current visual frame')
        if outcome not in ('in_progress','completed','need_ap','unsafe') or not evidence.strip():
            raise ValueError('Provide an interaction outcome and concrete visual evidence')
        if self.qwen and outcome=='completed':
            raise ValueError('Qwen mode requires two consecutive fresh-frame confirmations; GPT can request need_ap or unsafe instead')
        item=dict(frame_id=frame_id,step=self.worker.env.steps,outcome=outcome,evidence=evidence,
            interaction_goal=self.active_interaction_goal,source='gpt_visual_assessment',official_success_claim=False)
        self.interaction_assessments.append(item);self.awaiting_interaction_assessment=False
        if outcome=='unsafe':self.interaction_unsafe=True
        if outcome!='in_progress':
            self.qwen_votes.reset()
            self.phase='search';self.handoff=None;self.ap_required_after_step=self.worker.env.steps
            self.active_interaction_goal=None
        save(self.root/('interaction_assessment_%05d.json'%len(self.interaction_assessments)),item)
        return dict(accepted=True,assessment=item,phase=self.phase,physical_steps_added=0)

    def _complete(self, j, job_id):
        j.update(done=True,status='completed')
        if j['kind']=='champion' and self.options.get('interaction_only'):
            self.awaiting_interaction_assessment=not j.get('qwen_completed',False)
        if j.get('qwen_completed'):
            item=dict(step=self.worker.env.steps, outcome='completed',
                interaction_goal=self.active_interaction_goal, source='qwen_two_consecutive_current_frames',
                official_success_claim=False, checks=self.qwen_checks[-2:])
            self.interaction_assessments.append(item)
            save(self.root/('interaction_assessment_%05d.json'%len(self.interaction_assessments)),item)
            self.phase='search';self.handoff=None;self.ap_required_after_step=self.worker.env.steps
            self.active_interaction_goal=None;self.qwen_votes.reset()
        save(self.root/job_id/'result.json',self.summary())
        self.history.append(self.summary())
        value,images=self.worker.observe()
        return {'vla_job':self.summary(),'observation':value},images

    def _expire(self):
        expire=getattr(self.worker,'expire_if_due',None)
        return bool(expire and expire())

    def _launch(self):
        j = self.current
        d = self.root / j['job_id'] / f"chunk_{j['chunks_completed']:04d}"
        d.mkdir(parents=True, exist_ok=False)
        with self.worker.ledger.span('vla_observation', policy=j['kind']):
            obs = self.worker.env.capture(d / 'observation')
        j.update(directory=d, observation_step=self.worker.env.steps, event=threading.Event(), response=None, error=None)
        save(d / 'job.json', self.summary())
        # Copy all inputs. Background code has no worker/env/control reference.
        client, args, epoch, stop, event = self.clients[j['kind']], dict(j['args']), self.epoch, j['stop'], j['event']
        def infer():
            try:
                j['response'] = client.predict(obs, args, epoch, d, stop)
            except BaseException as exc:
                j['error'] = f'{type(exc).__name__}: {exc}'
            finally:
                event.set()
        j['thread'] = threading.Thread(target=infer, daemon=True, name='vla-http-only')
        j['thread'].start()

    def _launch_qwen(self):
        j=self.current
        d=self.root/j['job_id']/f"qwen_{j['chunks_completed']:04d}"
        d.mkdir(exist_ok=False)
        with self.worker.ledger.span('qwen_observation',policy='champion'):
            obs=self.worker.env.capture(d/'observation')
        j.update(monitor_pending=True,monitor_step=self.worker.env.steps,
                 event=threading.Event(),error=None,monitor_result=None,status='waiting_for_qwen')
        goal=j['args']['interaction_goal'];stop=j['stop'];event=j['event']
        def check():
            try:j['monitor_result']=self.qwen.check(obs,goal,d,stop)
            except BaseException as exc:j['error']=f'{type(exc).__name__}: {exc}'
            finally:event.set()
        j['thread']=threading.Thread(target=check,daemon=True,name='qwen-http-only')
        j['thread'].start()

    def continue_job(self, job_id, wait_seconds=20):
        j = self.current
        if j is None or j['job_id'] != job_id:
            raise ValueError('Unknown/stale VLA job; never start a replacement to replay it')
        if j['done']:
            return self.summary(), []
        if self._expire():
            j['stop'].set()
            return self._complete(j,job_id)
        if not j['event'].is_set():
            with self.worker.ledger.span('vla_wait', policy=j['kind']):
                remaining=getattr(self.worker,'remaining_wall_seconds',lambda:None)()
                j['event'].wait(wait_seconds if remaining is None else min(wait_seconds,max(0,remaining)))
        if self._expire():
            j['stop'].set()
            return self._complete(j,job_id)
        if not j['event'].is_set():
            return self.summary(), []
        try:
            if j['error']:
                raise RuntimeError(j['error'])
            if j.get('monitor_pending'):
                if self.worker.env.steps!=j['monitor_step']:
                    raise RuntimeError('Physics advanced during Qwen assessment')
                result=j['monitor_result']
                completed=self.qwen_votes.add(result['verdict'],result['observation_step'])
                result=dict(result,completion_streak=self.qwen_votes.streak,job_id=job_id,
                    evidence_directory=str(self.root/job_id/f"qwen_{j['chunks_completed']:04d}"))
                self.qwen_checks.append(result)
                save(self.root/'qwen_checks.json',self.qwen_checks)
                j['monitor_pending']=False
                if completed:
                    j['qwen_completed']=True
                    return self._complete(j,job_id)
                if j['chunks_completed']>=j['chunks_requested'] or self.worker.env.done or self.worker.env.success:
                    return self._complete(j,job_id)
                self._launch()
                return self.summary(), []
            if self.worker.env.steps != j['observation_step']:
                raise RuntimeError('Physics advanced during policy inference; stale chunk rejected')
            np = self.worker.np
            a = np.asarray(j['response']['actions'], dtype=np.float32)
            expected = (32 if j['kind'] == 'ap' else 20, 23)
            if a.shape != expected or not np.isfinite(a).all():
                raise ValueError('Nonfinite or malformed model action chunk')
            d = j['directory']
            np.save(d / 'predicted_actions.npy', a)
            prefix = min(8 if j['kind'] == 'ap' else 20, j['step_budget'] - j['executed_steps'])
            before = self.worker.env.steps
            with self.worker.ledger.span('action_execution', tool=j['kind'] + '_vla', job_id=job_id):
                for offset, action in enumerate(a[:prefix]):
                    if self._expire() or self.worker.env.done or self.worker.env.success:
                        break
                    self.worker.action_owner = {'controller': j['kind'] + '_vla', 'job_id': job_id,
                        'chunk': j['chunks_completed'], 'offset': offset}
                    self.worker.step(action)
                    self.worker.control.adopt_external_action(action)
                    j['executed_steps'] += 1
            save(d / 'execution_prefix.json', {'step_before': before, 'step_after': self.worker.env.steps,
                'executed_prefix': self.worker.env.steps - before, 'planned_prefix': prefix,
                'discarded_suffix': len(a) - (self.worker.env.steps - before), 'all_actions_acknowledged': True})
            if j['kind']=='champion' and self.worker.env.steps-before < len(a):
                # Published wrapper history must not assume the discarded tail was executed.
                self.epoch += 1
                save(d / 'history_reset.json', {'context_epoch':self.epoch,'reason':'unexecuted_champion_suffix'})
            j['chunks_completed'] += 1
            if j['kind']=='champion' and self.qwen and not self.worker.env.done and not self.worker.env.success:
                self._launch_qwen()
                return self.summary(), []
            if j['chunks_completed'] >= j['chunks_requested'] or self.worker.env.done or self.worker.env.success:
                return self._complete(j,job_id)
            self._launch()
            return self.summary(), []
        except BaseException as exc:
            error = {'job_id': job_id, 'kind': j['kind'], 'step': self.worker.env.steps,
                     'error': f'{type(exc).__name__}: {exc}'}
            self.infrastructure_errors.append(error)
            j.update(done=True, status='infrastructure_error')
            save(self.root / job_id / 'error.json', error)
            raise
        finally:
            self.worker.action_owner = {'controller': 'native_tools'}

    def close(self):
        if self.current:
            self.current['stop'].set()
            if not self.current['done'] and self._expire():
                self.current.update(done=True,status='completed')
                self.history.append(self.summary())
                save(self.root/self.current['job_id']/'result.json',self.summary())
        save(self.root / 'summary.json', {'jobs_started': self.count, 'completed_jobs': self.history,
            'last_job': self.summary() if self.current else None,
            'infrastructure_errors': self.infrastructure_errors, 'background_physics_steps': 0,
            'vla_only':bool(self.options.get('vla_only')),'handoffs':self.handoffs,'phase':self.phase,
            'closed_loop':bool(self.options.get('closed_loop')),'decisions':self.decisions,
            'interaction_only':bool(self.options.get('interaction_only')),
            'qwen_checks':self.qwen_checks,
            'interaction_assessments':self.interaction_assessments,
            'awaiting_interaction_assessment':self.awaiting_interaction_assessment})
