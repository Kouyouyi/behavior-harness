"""Visible native terminal: relay operator keys, report warnings, NEVER synthesize O.

The operator alone decides when to stop native control. Time, temperature and
telemetry warnings do not authorize motor shutdown. Native protections are unchanged.
"""
import argparse
import errno
import json
import os
from pathlib import Path
import pty
import select
import sys
import termios
import time
import tty


TEMPERATURE_ALERT_C = 130


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def health_reason(health, now):
    if not isinstance(health, dict):
        return 'invalid_motor_telemetry'
    stamp = health.get('monotonic')
    if type(stamp) not in (int, float) or not 0 <= now - stamp < 1:
        return 'missing_or_stale_motor_telemetry'
    motors = health.get('motors')
    if not isinstance(motors, list) or len(motors) != 29:
        return 'incomplete_motor_telemetry'
    try:
        for motor in motors:
            temperatures = motor['temperature_case_winding_c']
            if (len(temperatures) != 2 or
                    any(type(t) not in (int, float) or not -40 <= t <= 200 for t in temperatures)):
                return 'invalid_motor_temperature'
            if motor['motorstate'] != 0:
                return 'motor_fault'
            if max(temperatures) >= TEMPERATURE_ALERT_C:
                return f'temperature_alert_{TEMPERATURE_ALERT_C}C'
    except (KeyError, TypeError):
        return 'invalid_motor_telemetry'
    return None


def warning_reason(run, now, began):
    if (run / 'STOP_O').exists():
        return 'legacy_STOP_O_ignored_operator_keyboard_required'
    if read(run / 'ankle_status.json').get('phase') == 'fault_idle':
        return 'owner_fault'
    return health_reason(read(run / 'telemetry.json'), now) or (
        'trial_wall_time_warning' if now-began > 150 else None)


def report_warning(run, reason, now, prefix='native'):
    """Display only. No keyboard injection, signal, publisher or controller action."""
    value = dict(reason=reason, monotonic=now, automatic_stop=False,
                 temperature_alert_c=TEMPERATURE_ALERT_C,
                 message='仅告警，未发送 O。是否关闭底层控制由操作员决定。')
    (run / (prefix + '-warning.json')).write_text(json.dumps(value, ensure_ascii=False)+'\n')
    print('\r\n[Operator notice] '+json.dumps(value,ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true', help='Only run a fake keyboard child; type O to exit; no robot process')
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_dir():
        parser.error('--run must be an existing trial directory')
    if not args.self_test:
        if not (run / 'controller.sh').is_file():
            parser.error('controller.sh is missing; prepare this trial first')
        reason = health_reason(read(run / 'telemetry.json'), time.monotonic())
        if reason:
            parser.error('Native startup refused: ' + reason)
    pid, master = pty.fork()
    if pid == 0:
        if args.self_test:
            os.execvp('python3', ['python3', '-c',
                'import os,tty;tty.setraw(0);print("TEST READY",flush=True);x=os.read(0,1);print(repr(x),flush=True);raise SystemExit(0 if x==b"o" else 1)'])
        os.execvp('bash', ['bash', str(run / 'controller.sh')])
    old = termios.tcgetattr(0) if os.isatty(0) else None
    if old:
        tty.setraw(0)
    began = time.monotonic()
    reason = None
    previous_warning = None
    prefix = 'relay-test' if args.self_test else 'native'
    try:
        with (run / (prefix + '-terminal.log')).open('ab', buffering=0) as output:
            while True:
                ready, _, _ = select.select([master] + ([0] if old else []), [], [], .05)
                for fd in ready:
                    try:
                        data = os.read(fd, 65536)
                    except OSError as exc:
                        if exc.errno != errno.EIO:
                            raise
                        data = b''
                    if not data:
                        raise EOFError
                    if fd == master:
                        output.write(data)
                        os.write(1, data)
                    else:
                        # This is the ONLY path that writes to the native keyboard.
                        # Bytes originate at the operator terminal, never a timer/agent/file.
                        if b'o' in data or b'O' in data:
                            reason = 'operator_keyboard_O'
                        os.write(master, data)
                now = time.monotonic()
                warning = warning_reason(run, now, began)
                if warning and warning != previous_warning:
                    report_warning(run, warning, now, prefix)
                previous_warning = warning
    except EOFError:
        pass
    finally:
        if old:
            termios.tcsetattr(0, termios.TCSADRAIN, old)
        os.close(master)
    _, status = os.waitpid(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    (run / (prefix + '-exit.json')).write_text(json.dumps(dict(exit_code=code, reason=reason)) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
