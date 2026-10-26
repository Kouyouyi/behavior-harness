"""Runtime shipping without a network: the upload is replaced by a local copy."""
import json
import time
import zipfile

import pytest

from services import runtime_ship


def session(root, name, status='exited'):
    path = root/'sessions'/name
    (path/'agent-workspace'/'runtime'/'frames'/'f1').mkdir(parents=True)
    (path/'agent-workspace'/'runtime'/'frames'/'f1'/'head_camera.jpg').write_bytes(b'\xff\xd8' + b'x' * 1000)
    (path/'mcp.jsonl').write_text('{"event": "mcp_request"}\n')
    (path/'model-relay.json').write_text('{"key": "secret"}')
    (path/'claude-home'/'projects').mkdir(parents=True)
    (path/'claude-home'/'.claude.json').write_text('{"oauth": "secret"}')
    (path/'claude-home'/'projects'/'t.jsonl').write_text('{}\n')
    (path/'agent-lifecycle.json').write_text(json.dumps({'pid': 1, 'status': status, 'finished_at': time.time()}))
    return path


@pytest.fixture
def remote(tmp_path, monkeypatch):
    uploads = tmp_path/'remote'
    uploads.mkdir()

    def fake_upload(zip_path, digest, config):
        target = uploads/zip_path.name
        target.write_bytes(zip_path.read_bytes())
        return str(target)
    monkeypatch.setattr(runtime_ship, 'upload', fake_upload)
    return uploads


def test_finished_session_is_zipped_shipped_and_deleted_without_secrets(artifact_root, remote):
    root = session(artifact_root, 's1')
    assert runtime_ship.ship(root, runtime_ship.settings())
    assert not root.exists()
    [package] = list(remote.glob('*.zip'))
    names = zipfile.ZipFile(package).namelist()
    assert 's1/mcp.jsonl' in names and 's1/claude-home/projects/t.jsonl' in names
    assert 's1/agent-workspace/runtime/frames/f1/head_camera.jpg' in names
    assert not any('model-relay' in n or '.claude.json' in n for n in names)
    assert not list((artifact_root/'outbox').glob('*.zip'))
    assert 's1' in (artifact_root/'shipped.jsonl').read_text()


def test_stopped_session_ships_a_snapshot_but_stays_resumable(artifact_root, remote):
    root = session(artifact_root, 's2', status='stopped')
    assert runtime_ship.ship(root, runtime_ship.settings())
    assert root.exists() and (root/'.shipped.json').exists()
    assert not runtime_ship.ship(root, runtime_ship.settings())  # unchanged since: not again


def test_running_or_prepared_sessions_are_left_alone(artifact_root, remote):
    running = session(artifact_root, 's3', status='running')
    assert not runtime_ship.ship(running, runtime_ship.settings())
    assert running.exists() and not list(remote.iterdir())


def test_failed_upload_keeps_the_session(artifact_root, monkeypatch):
    root = session(artifact_root, 's4')

    def broken(*_):
        raise RuntimeError('network down')
    monkeypatch.setattr(runtime_ship, 'upload', broken)
    with pytest.raises(RuntimeError):
        runtime_ship.ship(root, runtime_ship.settings())
    assert root.exists() and not (root/'.shipped.json').exists()


def test_sweep_evicts_shipped_sessions_when_disk_is_low(artifact_root, remote, monkeypatch):
    root = session(artifact_root, 's5', status='stopped')
    monkeypatch.setattr(runtime_ship, 'free_gb', lambda: 1.0)
    runtime_ship.sweep({**runtime_ship.settings(), 'min_free_gb': 100})
    assert not root.exists() and list(remote.glob('s5__stopped__*.zip'))
