"""Public conversation projection; demo animation never represents model reasoning."""
from .adapters import finite
from .scenario import DECISIONS, RESULTS, RESULT_AFTER


def _text(value, limit=4000):
    return value[:limit] if isinstance(value, str) else ''


def _chat(view, messages, user):
    if user and user.get('channel') == 'chat':
        reply = next((m for m in reversed(messages) if m.get('reply_to') == user['id']), {})
        history = [m for m in messages if m.get('channel') == 'chat'
                   and m['id'] != user['id'] and m.get('reply_to') != user['id']][-8:]
        view.update(phase='replied', tool=None, plan=[], elapsed_s=None, history=history,
                    summary='演示会话 · 消息已接收，真实 Agent 尚未接入。',
                    output=dict(id=reply.get('id', 'waiting'), text=reply.get('text', ''), streaming=False))
    return view


def project(mode, task, messages, events, now, stages):
    user = next((m for m in reversed(messages) if m.get('role') == 'user'), None)
    reply = next((m for m in reversed(messages) if m.get('role') == 'assistant'), {})
    view = dict(source=mode, phase='idle', task=None, user=user,
                summary='', tool=None, elapsed_s=None,
                output=dict(id=reply.get('id', 'idle'), text=reply.get('text', ''), streaming=False))
    if mode == 'demo':
        if not task:
            return _chat(view, messages, user)
        stage = stages[task['stage_index']]
        plan = next((e for e in reversed(events) if e.get('kind') == 'plan'
                     and e.get('task_id') == task['id']), {})
        plan_steps = plan.get('details', {}).get('steps', [])
        view['plan'] = [dict(id=s['id'], label=s['label'], status='completed'
                            if i <= task.get('stage_result_index', -1) else 'running'
                            if i == task['stage_index'] and task['status'] == 'running' else 'pending')
                        for i, s in enumerate(plan_steps)]
        view['plan_note'] = task.get('interpretation', '')
        view['task'] = dict(id=task['id'], instruction=task['instruction'], status=task['status'],
                            elapsed_s=task['elapsed_s'], current_step=stage[1])
        view['tool'] = next((e for e in reversed(events)
                             if e.get('kind') == 'tool' and e.get('task_id') == task['id']), None)
        if view['tool']:
            view['tool'] = dict(view['tool'])
            if task.get('stage_result_index', -1) >= task['stage_index']:
                view['tool'].update(status='completed', result={'simulated': True, **RESULTS[task['stage_index']]})
        active = task['status'] in ('running', 'paused')
        point = task['paused_at'] if task['status'] == 'paused' else now
        elapsed = max(0, point - task['stage_started_at']) if active else 7
        view['elapsed_s'] = round(elapsed, 1)
        view['phase'] = ('thinking' if elapsed < 1.5 else 'tool' if elapsed < RESULT_AFTER else 'responding') if active else task['status']
        view['summary'] = DECISIONS[task['stage_index']]
        if elapsed < 1.5:
            view['tool'] = None
        if active:
            text = stage[4]
            count = min(len(text), max(0, int((elapsed - RESULT_AFTER) * 23)))
            view['output'] = dict(id=f"{task['id']}:{stage[0]}", text=text[:count],
                                  streaming=view['phase'] == 'responding' and count < len(text))
        if task['status'] == 'paused':
            view['phase'] = 'paused'
            view['output']['streaming'] = False
        return _chat(view, messages, user)

    # A task envelope accepts any user instruction; no fixed steps or inferred percentage.
    latest = next((e for e in reversed(events) if e.get('kind') == 'task'), None)
    if latest:
        details = latest.get('details')
        details = details if isinstance(details, dict) else {}
        task_id = _text(latest.get('task_id'), 200)
        elapsed = details.get('elapsed_s')
        status = latest.get('status')
        view['task'] = dict(id=task_id, instruction=_text(details.get('instruction'), 1000),
                            status=status if status in ('running', 'paused', 'completed', 'cancelled', 'failed', 'waiting') else 'unknown',
                            elapsed_s=elapsed if finite(elapsed) and elapsed >= 0 else None,
                            current_step=_text(details.get('current_step'), 200))
        events = [e for e in events if task_id and e.get('task_id') == task_id]
        view['user'] = dict(id=task_id, text=view['task']['instruction'], time=latest.get('time'))
        view['output'] = dict(id=task_id + ':waiting', text='等待 Agent 的公开输出。', streaming=False)
    view['phase'] = 'observing'
    plan = next((e for e in reversed(events) if e.get('kind') == 'plan'), {})
    details = plan.get('details')
    steps = details.get('steps') if isinstance(details, dict) else None
    if isinstance(steps, list):
        view['plan'] = [dict(id=_text(s.get('id')), label=_text(s.get('label'), 100),
                            status=s.get('status') if s.get('status') in ('running', 'completed', 'failed') else 'pending')
                        for s in steps[:12] if isinstance(s, dict) and isinstance(s.get('label'), str)]
        view['plan_note'] = _text(plan.get('text'), 300)
    view['tool'] = next((e for e in reversed(events) if e.get('kind') == 'tool'), None)
    summary = next((e for e in reversed(events) if e.get('kind') in ('observation', 'verification', 'error')), {})
    view['summary'] = _text(summary.get('text'))
    reply = next((e for e in reversed(events) if e.get('kind') in ('summary', 'message') and e.get('role') != 'user'), None)
    user = next((e for e in reversed(events) if e.get('kind') == 'message' and e.get('role') == 'user'), None)
    if user:
        view['user'] = dict(id=user.get('id'), text=_text(user.get('text')), time=user.get('time'))
    if reply:
        stamp = reply.get('time')
        streaming = reply.get('status') == 'streaming' and finite(stamp) and 0 <= now - stamp <= 5
        view['output'] = dict(id=_text(reply.get('id')) or 'agent-output', text=_text(reply.get('text')), streaming=streaming)
    return view
