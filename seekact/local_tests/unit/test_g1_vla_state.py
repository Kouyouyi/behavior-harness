"""Native schema compatibility and source-age rejection; no robot commands."""
from types import SimpleNamespace

import numpy as np
import pytest

from services.g1_sonic import vla_runtime


def reader(monkeypatch, state):
    monkeypatch.setattr(vla_runtime.time, 'monotonic_ns', lambda: 10_000_000_000)
    return vla_runtime.FreshBodyState(SimpleNamespace(get_msg=lambda clear: state))


def test_field_timestamp_converts_without_mutating_source(monkeypatch):
    state = dict(trace_monotonic_s=np.array([9.95]), body_q=np.ones(29))
    result = reader(monkeypatch, state).get_msg()
    assert result['state_monotonic_ns'] == 9_950_000_000
    result['body_q'][0] = 0
    assert state['body_q'][0] == 1 and 'state_monotonic_ns' not in state


def test_cached_sample_ages_out(monkeypatch):
    r = reader(monkeypatch, dict(trace_monotonic_s=[9.95]))
    assert r.get_msg()['state_monotonic_ns'] == 9_950_000_000
    monkeypatch.setattr(vla_runtime.time, 'monotonic_ns', lambda: 10_200_000_000)
    with pytest.raises(RuntimeError, match='Stale body'):
        r.get_msg()


@pytest.mark.parametrize('state', [
    {}, {'trace_monotonic_s': [float('nan')]}, {'trace_monotonic_s': []},
    {'trace_monotonic_s': [9.95, 9.96]}, {'trace_monotonic_s': ['9.95']},
    {'state_monotonic_ns': None, 'trace_monotonic_s': [9.95]},
    {'state_monotonic_ns': True},
])
def test_missing_or_malformed_timestamp_is_not_refreshed(monkeypatch, state):
    with pytest.raises(RuntimeError):
        reader(monkeypatch, state).get_msg()


@pytest.mark.parametrize('stamp', [9_700_000_000, 10_100_000_000])
def test_stale_or_future_primary_stamp_cannot_fall_back(monkeypatch, stamp):
    with pytest.raises(RuntimeError, match='Stale body'):
        reader(monkeypatch, dict(state_monotonic_ns=stamp, trace_monotonic_s=[9.95])).get_msg()


def test_new_schema_and_no_message(monkeypatch):
    assert reader(monkeypatch, dict(state_monotonic_ns=9_950_000_000)).get_msg()['state_monotonic_ns'] == 9_950_000_000
    assert reader(monkeypatch, None).get_msg() is None
