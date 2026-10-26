"""Temperature boundaries and warning conditions; no robot I/O."""
import pytest

from scripts.g1_vla_trial_terminal import health_reason


def health(temperature=40, fault=0):
    return dict(monotonic=10., motors=[
        dict(temperature_case_winding_c=[35, temperature], motorstate=fault)
        for _ in range(29)])


@pytest.mark.parametrize('temperature', [80, 100, 129])
def test_below_requested_temperature_does_not_alert(temperature):
    assert health_reason(health(temperature), 10.2) is None


@pytest.mark.parametrize('temperature', [130, 150])
def test_at_or_above_requested_temperature_alerts(temperature):
    assert health_reason(health(temperature), 10.2) == 'temperature_alert_130C'


def test_casing_temperature_also_alerts():
    sample = health()
    sample['motors'][0]['temperature_case_winding_c'] = [130, 40]
    assert health_reason(sample, 10.2) == 'temperature_alert_130C'


def test_fault_alerts_below_temperature_limit():
    assert health_reason(health(100, 0x200), 10.2) == 'motor_fault'


@pytest.mark.parametrize('now', [9, 11])
def test_invalid_telemetry_age_alerts(now):
    assert health_reason(health(), now) == 'missing_or_stale_motor_telemetry'


def test_incomplete_telemetry_alerts():
    sample = health()
    sample['motors'].pop()
    assert health_reason(sample, 10.2) == 'incomplete_motor_telemetry'


def test_invalid_temperature_alerts():
    assert health_reason(health(float('nan')), 10.2) == 'invalid_motor_temperature'


def test_time_fault_and_stop_file_only_generate_warnings(tmp_path, capsys):
    import json
    from scripts.g1_vla_trial_terminal import warning_reason, report_warning
    (tmp_path/'telemetry.json').write_text(json.dumps(health()))
    reason=warning_reason(tmp_path,10.2,-150)
    assert reason=='trial_wall_time_warning'
    report_warning(tmp_path,reason,10.2)
    assert not json.loads((tmp_path/'native-warning.json').read_text())['automatic_stop']
    assert not (tmp_path/'native-stop.json').exists()
    (tmp_path/'ankle_status.json').write_text('{"phase":"fault_idle"}')
    assert warning_reason(tmp_path,10.2,0)=='owner_fault'
    (tmp_path/'STOP_O').touch()
    assert warning_reason(tmp_path,10.2,0)=='legacy_STOP_O_ignored_operator_keyboard_required'


def test_fake_native_stays_running_until_operator_types_o(tmp_path):
    import os
    import pty
    import select
    import subprocess
    import sys
    import time
    import json
    master,slave=pty.openpty()
    process=subprocess.Popen([sys.executable,'scripts/g1_vla_trial_terminal.py','--run',str(tmp_path),'--self-test'],
                             stdin=slave,stdout=slave,stderr=slave)
    os.close(slave)
    try:
        output=b'';deadline=time.monotonic()+5
        while b'TEST READY' not in output and time.monotonic()<deadline:
            if select.select([master],[],[],.1)[0]:output+=os.read(master,65536)
        assert b'TEST READY' in output
        (tmp_path/'STOP_O').touch()
        # The old relay auto-sent O after 0.5 s. Both that path and STOP_O must be gone.
        time.sleep(.8)
        assert process.poll() is None
        warning=json.loads((tmp_path/'relay-test-warning.json').read_text())
        assert warning['automatic_stop'] is False
        os.write(master,b'o')
        assert process.wait(timeout=5)==0
        assert json.loads((tmp_path/'relay-test-exit.json').read_text())['reason']=='operator_keyboard_O'
    finally:
        if process.poll() is None:
            os.write(master,b'o');process.wait(timeout=5)
        os.close(master)
