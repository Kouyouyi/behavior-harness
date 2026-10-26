"""Supported server delays must fit the queue without relaxing deadlines."""
import pytest
from services.g1_sonic.vla_runtime import request_interval


@pytest.mark.parametrize('delay,expected', [(8,30),(9,30),(10,29)])
def test_interval_keeps_prefix_inside_horizon(delay,expected):
    actual=request_interval(dict(rtc_delay_steps=delay,rtc_max_delay_steps=10,action_horizon=40))
    assert actual == expected
    assert delay <= actual and actual+delay < 40


@pytest.mark.parametrize('config', [
    dict(rtc_delay_steps=11,rtc_max_delay_steps=10,action_horizon=40),
    dict(rtc_delay_steps=10,rtc_max_delay_steps=8,action_horizon=40),
    dict(rtc_delay_steps=True,rtc_max_delay_steps=10,action_horizon=40),
    dict(rtc_delay_steps=8,rtc_max_delay_steps=10,action_horizon=20),
    {},
])
def test_rejects_untrained_or_unknown_timing(config):
    with pytest.raises(ValueError):request_interval(config)
