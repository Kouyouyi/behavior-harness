"""Read-only, bounded artifact adapters. No SDK imports or robot connections."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

JOINTS = sorted(json.loads((Path(__file__).parents[1] / 'g1_sonic/joints.json').read_text())['joints'],
                key=lambda joint: joint['hardware_index'])
BOOT_ID = Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def read_json(path):
    try:
        value = json.loads(Path(path).read_text()) if path else {}
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def tail_json(path, limit=100):
    if not path:
        return []
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 262144))
            raw = stream.read().splitlines()
        if size > 262144:
            raw = raw[1:]
        rows = []
        for line in raw[-limit:]:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
            except ValueError:
                continue
        return rows
    except OSError:
        return []


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def joint_group(name):
    if name.startswith('waist'):
        return 'torso'
    side = 'left' if name.startswith('left') else 'right'
    return side + ('_leg' if any(part in name for part in ('hip', 'knee', 'ankle')) else '_arm')


def telemetry(path, now=None):
    now = time.time() if now is None else now
    rows = tail_json(path, 2) if path and str(path).endswith('.jsonl') else [read_json(path)]
    raw = rows[-1] if rows else {}
    sample = raw.get('sample', raw)
    if not isinstance(sample, dict):
        sample = {}
    stamp = sample.get('sampled_at')
    age = now - stamp if finite(stamp) else None
    fresh = (age is not None and 0 <= age <= 2.5 and sample.get('host_boot_id') == BOOT_ID
             and sample.get('source') == 'live_lowstate')
    source_motors = sample.get('motors')
    source_motors = source_motors if isinstance(source_motors, list) else []
    motors = {m['hardware_index']: m for m in source_motors
              if isinstance(m, dict) and type(m.get('hardware_index')) is int}
    joints = []
    for joint in JOINTS:
        motor = motors.get(joint['hardware_index'], {})
        # Legacy field name is wrong: the SDK tuple order is case, winding.
        temps = motor.get('temperature_case_winding_c', motor.get('temperature_winding_driver_c', []))
        valid = isinstance(temps, list) and len(temps) == 2 and all(finite(t) for t in temps)
        error = motor.get('motorstate')
        error = error if type(error) is int else None
        joints.append(dict(id=joint['hardware_index'], name=joint['name'], group=joint_group(joint['name']),
                           case_c=temps[0] if valid else None, winding_c=temps[1] if valid else None,
                           motorstate=error, fresh=fresh and valid and isinstance(error, int)))
    complete = all(j['fresh'] for j in joints)
    return dict(source='live_lowstate' if motors else 'unavailable', fresh=complete,
                sampled_at=stamp if finite(stamp) else None, age_s=round(age, 3) if age is not None else None,
                joints=joints, fault_count=sum(bool(j['motorstate']) for j in joints),
                note=None if complete else '等待有效的同机实时遥测；旧值仅供查看')


class Camera:
    def __init__(self, directory=None):
        self.directory = Path(directory).resolve() if directory else None

    def snapshot(self):
        meta = read_json(self.directory / 'latest.json') if self.directory else {}
        frame_id = meta.get('frame_id', '')
        valid = isinstance(frame_id, str) and bool(re.fullmatch(r'[a-f0-9]{32}', frame_id))
        stamp = meta.get('published_unix')
        received = meta.get('received_monotonic')
        age = time.monotonic() - received if finite(received) else None
        fresh = (valid and age is not None and 0 <= age <= 1 and meta.get('host_boot_id') == BOOT_ID
                 and meta.get('source') == 'live_camera')
        return dict(source='live_camera' if valid else 'unavailable', fresh=bool(fresh),
                    frame_id=frame_id if valid else None, sampled_at=stamp,
                    age_s=round(age, 3) if age is not None else None,
                    url='/api/v1/camera/frame?frame_id=' + frame_id if valid else None,
                    timestamp_basis=meta.get('timestamp_basis'))

    def frame(self, frame_id):
        if not self.directory or not re.fullmatch(r'[a-f0-9]{32}', frame_id):
            raise KeyError('画面不存在')
        path = (self.directory / frame_id / 'rgb.png').resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError('invalid camera path')
        try:
            return path.read_bytes()
        except OSError:
            raise KeyError('画面已过期，请等待下一帧') from None


def agent_events(path):
    """Accept app-server notifications or the documented console event envelope.

    Deliberately allowlist public summaries and tool records, never raw reasoning.
    This is an observation adapter, not a Codex process launcher.
    """
    events = []
    for index, row in enumerate(tail_json(path, 120)):
        kind = row.get('kind')
        if kind in ('summary', 'tool', 'observation', 'verification', 'error', 'task', 'message', 'plan'):
            events.append({key: row[key] for key in ('id', 'time', 'kind', 'title', 'text', 'tool', 'status', 'details', 'role', 'task_id')
                           if key in row})
            continue
        method = row.get('method')
        params = row.get('params', {})
        item = params.get('item', {}) if isinstance(params, dict) else {}
        if method not in ('item/started', 'item/completed') or not isinstance(item, dict):
            continue
        if item.get('type') == 'mcpToolCall':
            events.append(dict(id=item.get('id', str(index)), time=row.get('timestamp'), kind='tool',
                               title=item.get('tool', 'MCP'), tool=item.get('tool'),
                               status=item.get('status', 'running'), text='MCP 工具调用',
                               details={key: item.get(key) for key in ('server', 'arguments', 'result', 'error')}))
        elif item.get('type') == 'agentMessage':
            events.append(dict(id=item.get('id', str(index)), time=row.get('timestamp'), kind='summary',
                               title='Agent 回复', text=str(item.get('text', ''))[:4000]))
    return events[-60:]
