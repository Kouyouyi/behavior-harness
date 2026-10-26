"""Complete public errors: the full message, with credentials, URLs and host directories removed."""
import re
import traceback

MAX_REASON = 8000


def public_failure(exc, *, operation=None, stage='request', traceback_tail=False):
    """The whole error the agent sees. Never replace it with a summary or a generic label.

    traceback_tail: append the innermost frames (file, line, function) of an unexpected
    exception, so the agent sees where it came from, not only its message.
    """
    if stage == 'provider_request':
        # Provider exceptions can embed request headers and arbitrary response bodies.
        status = getattr(exc, 'code', None)
        reason = ('Provider request timed out; its billing reservation is retained.'
                  if isinstance(exc, TimeoutError) else
                  f'Provider request failed ({type(exc).__name__}' +
                  (f', HTTP {status}' if isinstance(status, int) else '') +
                  '); its billing reservation is retained. Private service logs contain diagnostics.')
    else:
        reason = str(exc) or type(exc).__name__
        if traceback_tail and exc.__traceback__ is not None:
            frames = traceback.extract_tb(exc.__traceback__)[-4:]
            reason += '\nTraceback (innermost last):\n' + '\n'.join(
                f'  {f.filename}:{f.lineno} in {f.name}' for f in frames)
        reason = re.sub(r'https?://\S+', '[redacted URL]', reason)
        # Keep file names (tracebacks name the failing module) but not the host's directories.
        reason = re.sub(r'(?<![\w.])/(?:[^\s,;\]\)\"\'/]+/)+', '', reason)
        reason = re.sub(r'(?i)(bearer\s+|(?:api[_-]?key|authorization|token)\s*[=:]\s*)[^\s,;]+', r'\1[redacted]', reason)
        reason = re.sub(r'\bsk-[A-Za-z0-9_-]+', '[redacted key]', reason)
        if len(reason) > MAX_REASON:  # Keep the start and the innermost traceback frames.
            reason = reason[:MAX_REASON // 4] + '\n...\n' + reason[-(MAX_REASON * 3 // 4):]
    failure = {'type': type(exc).__name__, 'operation': operation, 'stage': stage, 'reason': reason}
    state = getattr(exc, 'state', None)
    if state:
        failure['simulator_state'] = state
    robot_state = getattr(exc, 'robot_state', None)
    if isinstance(robot_state, dict):
        failure['robot_state'] = robot_state  # the measured state the robot was left in
    return failure
