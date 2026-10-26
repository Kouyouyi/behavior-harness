#!/usr/bin/env python3
"""Build a Chinese offline report from sealed experiment evidence.

No model, simulator, expert text or video contents are opened. This is a report
of recorded outcomes, not a new evaluation. Calibration folders are excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
from urllib.parse import quote

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2]/'outputs/seekact'
PHASES = ('zero_shot', 'one_expert_retry')
PHASE_LABELS = {'zero_shot': '零样本', 'one_expert_retry': '单专家经验重试'}
TASK_LABELS = {'putting_shoes_on_rack': '把鞋放上鞋架', 'putting_away_toys': '收拾玩具',
               'turning_on_radio': '打开收音机'}
STATUS_LABELS = {'not_run': '未运行', 'unsealed': '尚未封存', 'task_success': '任务成功',
    'not_required': '零样本成功，无需专家重试',
    'task_failure': '任务未完成（有效评测）', 'agent_infrastructure_error': 'Agent 基础设施故障',
    'infrastructure_error': '基础设施故障', 'evidence_error': '证据不完整，待检查',
    'complete': '已完成', 'pending': '待完成', 'unknown': '未知 / 待检查', 'not_configured': '未设置'}
REASONS = {'supervisor_error': '调度异常', 'worker_nonzero_exit_or_still_running': '仿真进程未正常退出',
    'worker_or_video_finalization_error': '仿真或视频收尾异常', 'missing_closed_episode': '缺少封存结果',
    'action_ack_integrity_not_verified': '动作 ACK 未通过完整性检查', 'simulator_errors': '仿真返回错误',
    'invalid_environment_or_success_evidence': '环境或成功判定证据无效',
    'video_integrity_not_verified': '视频未通过完整验收', 'no_verified_playable_video': '缺少已验收视频',
    'robot_cameras_full_decode_not_verified': '机器人视角未通过完整解码',
    'third_person_full_decode_not_verified': '第三人称未通过完整解码',
    'agent_did_not_complete_normally': 'Agent 未正常结束', 'agent_not_started': 'Agent 尚未启动',
    'zero_step_video_status_missing': '缺少零动作视频状态', 'invalid_video_evidence': '视频证据格式异常'}
TIMING_NOTE = ('所有时间均为真实墙钟，不是仿真时间。调度总耗时包含初始化、Agent 会话和收尾；'
    '等待 runner.lock 的排队发生在本轮调度计时开始前；有匹配凭据时单列“排队起点到本轮 ledger 启动”，'
    '其中包含 launcher 开销，不等于纯锁等待，也不加入本轮运行墙钟。未记录或校验不通过时未知，不能记作零。'
    'worker 的 GPU 等待另有记录，已包含在初始化等待及调度总墙钟中。'
    'Agent、worker 与调度的测量区间重叠，不能跨层相加。Agent 进程耗时包含网络、推理、工具等待及进程开销，'
    '仅在完整封存后计算 Agent 区间与工具跨度并集的重叠及剩余时间；剩余时间不是纯 LLM 推理耗时。'
    '视频覆盖每个已执行物理步、按 30fps 回放；模型等待时冻结仿真，等待墙钟计入总耗时，'
    '但不会重复编码成静止帧，因此视频时长不是实际实验用时，不用于性能计时。')


def load(path, warnings, *, required=False):
    path = Path(path)
    if not path.is_file():
        if required:
            warnings.append(f'缺少文件：{path.name}')
        return None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('JSON 顶层不是对象')
        return value
    except (OSError, ValueError) as exc:
        warnings.append(f'{path.name} 无法读取：{type(exc).__name__}: {exc}')
        return None


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def category(ledger, name):
    values = ledger.get('by_name', {}) if isinstance(ledger, dict) else {}
    row = values.get(name, {}) if isinstance(values, dict) else {}
    return number(row.get('exclusive_seconds')) if isinstance(row, dict) else None


def interval_union_ns(intervals):
    """Union duration, so overlapping/nested spans are never double-counted."""
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _utc_seconds(value):
    if not isinstance(value, str):
        raise ValueError('missing UTC timestamp')
    instant = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError('UTC timestamp must carry timezone')
    return instant.timestamp()


def _monotonic_ns(value):
    if type(value) is not int or value < 0:
        raise ValueError('invalid monotonic nanosecond timestamp')
    return value


def analyze_agent_tool_time(agent, worker, events, *, sealed, normal_close):
    """Intersect complete local-worker tool spans with the Agent process window.

    UTC is used only for cross-clock sanity checks, never for a duration. Older
    evidence has no host/boot identifier, so a consistent UTC/perf mapping plus
    containment is required; this is not a claim of externally synchronized clocks.
    Missing or inconsistent evidence leaves all derived times unknown.
    """
    answer = {
        'status': 'unknown', 'valid': False, 'reasons': [],
        'agent_wall_ns': None, 'agent_wall_seconds': None,
        'tool_dispatch_intersection_union_ns': None, 'tool_dispatch_intersection_union_seconds': None,
        'agent_residual_ns': None, 'agent_residual_seconds': None,
        'observation_intersection_union_ns': None, 'observation_intersection_union_seconds': None,
        'action_execution_intersection_union_ns': None, 'action_execution_intersection_union_seconds': None,
        'accounting_exact_in_nanoseconds': None,
        'definition': 'Agent monotonic interval intersected with the union of worker tool_dispatch spans',
        'residual_scope': '推理、网络、CLI、Frontend 传输/调度等未落在 worker 工具跨度中的时间；不是纯 LLM 耗时',
        'subspan_scope': '观察和物理动作属于工具子跨度，不能再与工具并集相加',
        'duration_clock': 'time.perf_counter_ns; never simulation steps or video duration',
    }
    if not sealed:
        answer.update(status='pending', reasons=['实验尚未封存，暂不计算最终时间分解'])
        return answer
    if not normal_close:
        answer['reasons'] = ['本轮未正常封存，不能声称工具区间覆盖完整']
        return answer
    try:
        if not isinstance(agent, dict) or not isinstance(worker, dict) or not isinstance(events, list) or not events:
            raise ValueError('缺少 Agent、worker 汇总或完整计时事件')
        if agent.get('status') != 'agent_completed' or agent.get('exit_code') != 0:
            raise ValueError('Agent 未正常结束')
        if worker.get('finished') is not True or worker.get('duration_clock') != 'time.perf_counter_ns':
            raise ValueError('worker 计时未封存或时钟来源不匹配')
        coverage = worker.get('coverage', {})
        if not isinstance(coverage, dict) or coverage.get('open_span_count') != 0:
            raise ValueError('worker 尚有未关闭计时跨度')
        a0 = _monotonic_ns(agent.get('started_perf_counter_ns'))
        a1 = _monotonic_ns(agent.get('ended_perf_counter_ns'))
        w0, w1 = _monotonic_ns(worker.get('start_ns')), _monotonic_ns(worker.get('end_ns'))
        if not w0 <= a0 <= a1 <= w1 or a0 == a1:
            raise ValueError('Agent 区间不在 worker 生命周期内，可能跨主机/跨启动时钟')
        if worker.get('total_ns') != w1 - w0:
            raise ValueError('worker 单调时钟总量不一致')
        reported_agent = number(agent.get('agent_wall_seconds'))
        if reported_agent is None or abs(reported_agent - (a1 - a0) / 1e9) > 1e-6:
            raise ValueError('Agent 墙钟与单调区间不一致')
        tolerance = 2.0
        reference_offset = _utc_seconds(worker.get('start_utc')) - w0 / 1e9
        stamps = [(w1, worker.get('end_utc')), (a0, agent.get('started_at_utc')),
                  (a1, agent.get('ended_at_utc'))]
        for stamp, utc in stamps:
            if abs((_utc_seconds(utc) - stamp / 1e9) - reference_offset) > tolerance:
                raise ValueError('UTC 与 perf_counter 映射不一致，跨时钟结果记为未知')
        if worker.get('event_count') != len(events):
            raise ValueError('计时事件数量不完整')
        if events[0].get('event') != 'ledger_started' or events[-1].get('event') != 'ledger_finished':
            raise ValueError('计时事件缺少完整起止标记')
        if events[0].get('perf_counter_ns') != w0 or events[-1].get('perf_counter_ns') != w1:
            raise ValueError('计时事件与汇总生命周期不一致')
        if events[-1].get('total_ns') != w1 - w0:
            raise ValueError('计时事件终点总量不一致')
        stack, spans = [], {}
        previous = w0
        for index, row in enumerate(events, 1):
            if not isinstance(row, dict) or row.get('event_index') != index:
                raise ValueError('计时事件编号不连续')
            stamp = _monotonic_ns(row.get('perf_counter_ns'))
            if not previous <= stamp <= w1 or row.get('elapsed_ns') != stamp - w0:
                raise ValueError('计时事件顺序或 elapsed_ns 不一致')
            previous = stamp
            if abs((_utc_seconds(row.get('timestamp_utc')) - stamp / 1e9) - reference_offset) > tolerance:
                raise ValueError('事件 UTC 与单调时钟映射不一致')
            kind = row.get('event')
            if kind == 'span_started':
                ident = row.get('span_id')
                parent = stack[-1] if stack else None
                if not isinstance(ident, str) or ident in spans or row.get('parent_id') != parent:
                    raise ValueError('计时跨度身份或嵌套关系不一致')
                if row.get('start_ns') != stamp or row.get('status') != 'running':
                    raise ValueError('计时跨度起点无效')
                if not isinstance(row.get('name'), str) or not row['name']:
                    raise ValueError('计时跨度缺少名称')
                spans[ident] = dict(row)
                stack.append(ident)
            elif kind == 'span_finished':
                ident = row.get('span_id')
                if not stack or stack[-1] != ident:
                    raise ValueError('计时跨度未成对关闭')
                old = spans[ident]
                if any(old.get(key) != row.get(key) for key in ('name', 'parent_id', 'start_ns')):
                    raise ValueError('跨度结束记录与开始记录不同')
                if (row.get('end_ns') != stamp or row.get('duration_ns') != stamp - old['start_ns']
                        or row.get('status') not in ('ok', 'error')):
                    raise ValueError('计时跨度终点无效')
                spans[ident] = dict(row)
                stack.pop()
        if stack:
            raise ValueError('计时事件存在未关闭跨度')
        by_name = {}
        for span in spans.values():
            by_name.setdefault(span['name'], []).append(span)
        actual_names = sorted(by_name)
        if coverage.get('observed_categories') != actual_names:
            raise ValueError('实际跨度类别与 coverage 不一致')
        declared = worker.get('by_name')
        if not isinstance(declared, dict) or sorted(declared) != actual_names:
            raise ValueError('实际跨度类别与汇总不一致')
        for name, rows in by_name.items():
            summary = declared[name]
            if (not isinstance(summary, dict) or summary.get('count') != len(rows)
                    or summary.get('completed_count') != len(rows) or summary.get('running_count') != 0
                    or summary.get('inclusive_ns') != interval_union_ns((s['start_ns'], s['end_ns']) for s in rows)):
                raise ValueError('跨度数量或区间并集与汇总不一致')
        # An observation/action must be nested inside a dispatch; otherwise scope is not the contract.
        for span in spans.values():
            if span['name'] not in ('observation', 'action_execution'):
                continue
            parent = span.get('parent_id')
            while parent is not None and spans[parent]['name'] != 'tool_dispatch':
                parent = spans[parent].get('parent_id')
            if parent is None:
                raise ValueError('观察/动作跨度不在工具分发内，无法按该口径分解')
        def clipped(name):
            return interval_union_ns((max(a0, row['start_ns']), min(a1, row['end_ns']))
                                     for row in by_name.get(name, []))
        tool_ns = clipped('tool_dispatch')
        agent_ns = a1 - a0
        answer.update(status='valid', valid=True, agent_wall_ns=agent_ns, agent_wall_seconds=agent_ns / 1e9,
            tool_dispatch_intersection_union_ns=tool_ns, tool_dispatch_intersection_union_seconds=tool_ns / 1e9,
            agent_residual_ns=agent_ns - tool_ns, agent_residual_seconds=(agent_ns - tool_ns) / 1e9,
            worker_event_count=len(events),
            worker_tool_dispatch_spans_total=len(by_name.get('tool_dispatch', [])),
            tool_dispatch_spans_intersecting_agent=sum(
                max(a0, row['start_ns']) < min(a1, row['end_ns'])
                for row in by_name.get('tool_dispatch', [])),
            tool_span_count_is_not_model_request_count=True,
            clock_validation={'method': 'local interval containment and UTC/perf mapping consistency',
                              'utc_offset_tolerance_seconds': tolerance, 'host_boot_id_recorded': False},
            accounting_exact_in_nanoseconds=tool_ns + (agent_ns - tool_ns) == agent_ns)
        for name in ('observation', 'action_execution'):
            value = clipped(name)
            answer[name + '_intersection_union_ns'] = value
            answer[name + '_intersection_union_seconds'] = value / 1e9
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        answer['reasons'] = [str(exc)]
    return answer


def load_timing_events(path):
    """Return unknown on incomplete/invalid logs, without treating partial spans as zero."""
    try:
        raw = Path(path).read_bytes()
        if not raw.endswith(b'\n'):
            return None
        events = [json.loads(line) for line in raw.splitlines()]
        return events if all(isinstance(row, dict) for row in events) else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def safe_relative(path, output):
    """Only link files in this report's evidence tree; never arbitrary assets."""
    path = Path(path)
    if not path.is_absolute():
        path = output / path
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(output.resolve())
    except ValueError:
        return None
    return relative.as_posix() if resolved.is_file() else None


def source_link(path, output):
    value = safe_relative(path, output)
    return value if value else None


def _sha256_text(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def expert_stages(output, task):
    """Read only the two completion metadata files, never expert text/media data."""
    fixed = task.get('fixed_expert', {})
    episode = fixed.get('episode_index') if isinstance(fixed, dict) else None
    pending = {'status': 'pending', 'wall_seconds': None, 'reasons': []}
    result = {'fixed_episode_index': episode, 'source_count': None,
        'export': dict(pending), 'analysis': dict(pending), 'source_file_sha256': {},
        'expert_video_links': {}, 'experience_link': None, 'experience_sha256_reported': None,
        'metadata_links': {}, 'expert_content_read': False,
        'timing_scope': '专家导出和经验分析是独立墙钟；可能与其他任务执行重叠，不与零样本或重试墙钟相加为实验总耗时。',
        'analysis_scope': '从首次读取专家内容前到分析完成的实际经过墙钟，包含读取、工具、推理及其他任务编排等待；不是纯模型推理或纯 CPU 分析时间。'}
    if type(episode) is not int or episode < 0:
        result['export']['status'] = result['analysis']['status'] = 'not_configured'
        return result
    directory = output / task['task_name'] / f'expert_episode_{episode}'
    if not directory.resolve().is_relative_to(output.resolve()):
        result['export'].update(status='unknown', reasons=['专家目录不在本实验输出树内'])
        return result
    export_path, analysis_path = directory / 'export_complete.json', directory / 'experience_analysis.json'
    export_warnings, analysis_warnings = [], []
    exported = load(export_path, export_warnings)
    analyzed = load(analysis_path, analysis_warnings)
    for label, path in (('export_complete', export_path), ('experience_analysis', analysis_path)):
        relative = source_link(path, output)
        if relative:
            result['metadata_links'][label] = relative
    export_digest = None
    if export_path.is_file():
        # Hashing this completion JSON does not read a demonstration or summary.
        export_digest = hashlib.sha256(export_path.read_bytes()).hexdigest()
        result['export_complete_sha256'] = export_digest
    if exported is not None:
        result['source_count'] = exported.get('source_count')
        if exported.get('status') != 'export_verified':
            export_warnings.append('导出完成标记尚未达到 export_verified')
        if type(exported.get('source_count')) is not int or exported['source_count'] != 1:
            export_warnings.append('导出来源数量不是严格的一条')
        if exported.get('task_name') != task['task_name']:
            export_warnings.append('导出任务名称不一致')
        selected = exported.get('fixed_expert', {})
        if not isinstance(selected, dict) or any(selected.get(key) != fixed.get(key)
                for key in ('episode_index', 'raw_episode_id', 'frames')):
            export_warnings.append('导出与协议预先固定的专家条目不一致')
        wall = exported.get('wall_timing', {})
        try:
            if not isinstance(wall, dict) or wall.get('finished') is not True or wall.get('duration_clock') != 'time.perf_counter_ns':
                raise ValueError('专家导出墙钟尚未正常封存')
            elapsed = _monotonic_ns(wall.get('end_ns')) - _monotonic_ns(wall.get('start_ns'))
            total = number(wall.get('total_seconds'))
            if elapsed < 0 or elapsed != wall.get('total_ns') or total is None or abs(total - elapsed / 1e9) > 1e-6:
                raise ValueError('专家导出真实墙钟证据不一致')
            result['export'].update(wall_seconds=total, started_at_utc=exported.get('started_at_utc'),
                ended_at_utc=exported.get('ended_at_utc'),
                phases_exclusive_seconds={name: category(wall, name) for name in (
                    'expert_export', 'expert_transfer', 'expert_verification')})
        except (ValueError, TypeError) as exc:
            export_warnings.append(str(exc))
        manifest = exported.get('remote_manifest', {})
        files = manifest.get('files', {}) if isinstance(manifest, dict) else {}
        if isinstance(files, dict):
            for name in ('episode.parquet', 'annotation.json', 'videos/head.mp4',
                         'videos/left_wrist.mp4', 'videos/right_wrist.mp4'):
                meta = files.get(name, {})
                digest = meta.get('sha256') if isinstance(meta, dict) else None
                if _sha256_text(digest):
                    result['source_file_sha256'][name] = digest
        if len(result['source_file_sha256']) != 5:
            export_warnings.append('单专家导出文件的 SHA256 元数据不完整')
        videos = exported.get('videos', {})
        video_meta = videos.get('videos', {}) if isinstance(videos, dict) else {}
        if isinstance(video_meta, dict):
            for role in ('head', 'left_wrist', 'right_wrist'):
                meta = video_meta.get(role, {})
                if isinstance(meta, dict) and isinstance(meta.get('path'), str):
                    relative = safe_relative(meta['path'], output)
                    if relative:
                        result['expert_video_links'][role] = {'path': relative,
                            'full_decode_verified_in_export_metadata': videos.get('passed') is True and meta.get('full_decode_passed') is True}
        result['export']['status'] = 'unknown' if export_warnings else 'complete'
    elif export_warnings:
        result['export']['status'] = 'unknown'
    result['export']['reasons'] = export_warnings
    if analyzed is not None:
        if analyzed.get('status') != 'complete':
            analysis_warnings.append('经验分析标记尚未达到 complete')
        if type(analyzed.get('source_count')) is not int or analyzed['source_count'] != 1:
            analysis_warnings.append('经验分析来源数量不是严格的一条')
        if type(analyzed.get('episode_index')) is not int or analyzed['episode_index'] != episode:
            analysis_warnings.append('经验分析的 episode 与固定专家不一致')
        if result['export']['status'] != 'complete' or analyzed.get('export_complete_sha256') != export_digest:
            analysis_warnings.append('经验分析未关联到本次完整验收的导出元数据')
        try:
            elapsed = _monotonic_ns(analyzed.get('ended_perf_counter_ns')) - _monotonic_ns(analyzed.get('started_perf_counter_ns'))
            total = number(analyzed.get('analysis_wall_seconds'))
            if elapsed < 0 or total is None or abs(total - elapsed / 1e9) > 1e-6:
                raise ValueError('经验分析真实墙钟证据不一致')
            result['analysis'].update(wall_seconds=total,
                started_perf_counter_ns=analyzed['started_perf_counter_ns'],
                ended_perf_counter_ns=analyzed['ended_perf_counter_ns'])
        except (ValueError, TypeError) as exc:
            analysis_warnings.append(str(exc))
        path = analyzed.get('experience_path')
        relative = safe_relative(path, output) if isinstance(path, str) and path else None
        digest = analyzed.get('experience_sha256')
        if not relative or not _sha256_text(digest):
            analysis_warnings.append('缺少可访问的经验文本链接或所记录的 SHA256')
        result['experience_link'] = relative
        result['experience_sha256_reported'] = digest if _sha256_text(digest) else None
        result['analysis']['status'] = 'unknown' if analysis_warnings else 'complete'
        result['analysis']['scope_recorded'] = analyzed.get('scope')
    elif analysis_warnings:
        result['analysis']['status'] = 'unknown'
    result['analysis']['reasons'] = analysis_warnings
    return result


def queue_to_ledger_time(directory, output, task, phase):
    """Match a queued command to one fresh trial; durations use monotonic ns only."""
    unknown = {'status': 'unknown', 'wall_seconds': None, 'duration_ns': None,
        'included_in_trial_wall': False,
        'scope': '排队起点到本轮 ledger 启动；含 launcher 开销，不是纯 flock 等待；独立于本轮运行墙钟。'}
    attempt = directory.name[len(phase) + 1:]

    def option(command, name):
        values = [command[i + 1] for i, value in enumerate(command[:-1]) if value == name]
        values += [value[len(name) + 1:] for value in command if value.startswith(name + '=')]
        if len(values) > 1 or (command and command[-1] == name):
            raise ValueError('排队命令含重复或缺值参数')
        return values[0] if values else None

    matches = []
    for path in sorted(output.glob('*_queue.json')):
        receipt = load(path, [])
        if receipt is None:
            continue
        targets = [receipt[key] for key in ('fresh_target', 'expected_trial') if key in receipt]
        target_match = any(isinstance(value, str) and Path(value).is_absolute()
                           and Path(value).resolve() == directory.resolve() for value in targets)
        command = receipt.get('command')
        valid_command = isinstance(command, list) and all(isinstance(value, str) for value in command)
        try:
            command_match = (valid_command and option(command, '--task') in
                (task['task_name'], str(task.get('task_index'))) and option(command, '--phase') == phase
                and option(command, '--attempt') == attempt)
        except ValueError:
            command_match = False
        if target_match or command_match:
            matches.append((path, receipt, targets, command, command_match))
    if len(matches) != 1:
        return dict(unknown, reason='无唯一匹配的排队凭据（缺失或存在歧义）')
    path, receipt, targets, command, command_match = matches[0]
    source = source_link(path, output)
    try:
        if (not targets or not all(isinstance(value, str) and Path(value).is_absolute()
                and Path(value).resolve() == directory.resolve() for value in targets)
                or not command_match or len(command) < 2
                or Path(command[1]).name != 'run_behavior_experiment.py'
                or command.count('--wait-for-runner') != 1
                or any(value == '--output' or value.startswith('--output=') for value in command)
                or Path(option(command, '--protocol') or DEFAULT_OUTPUT / 'protocol.json').resolve().parent != output.resolve()
                or receipt.get('phase', phase) != phase):
            raise ValueError('排队目标、任务/阶段/attempt 或输出目录与本轮不匹配')
        events_path = directory / 'supervisor_timing/timing_events.jsonl'
        with events_path.open() as stream:
            started = json.loads(stream.readline())
        if not isinstance(started, dict) or started.get('event') != 'ledger_started':
            raise ValueError('缺少本轮首条 ledger_started 证据')
        queued_ns = _monotonic_ns(receipt.get('queued_perf_counter_ns'))
        started_ns = _monotonic_ns(started.get('perf_counter_ns'))
        if started_ns < queued_ns:
            raise ValueError('单调时钟逆序')
        duration_ns = started_ns - queued_ns
        utc_delta = _utc_seconds(started.get('timestamp_utc')) - _utc_seconds(receipt.get('queued_at_utc'))
        # Receipts have no boot ID: this is an epoch sanity check, not a second
        # duration source or proof of host identity. Never fill missing perf data.
        if abs(utc_delta - duration_ns / 1e9) > 1.0:
            raise ValueError('UTC 标签与 perf 区间不一致，无法确认同一时钟 epoch')
        return dict(unknown, status='measured', wall_seconds=duration_ns / 1e9,
            duration_ns=duration_ns, queued_perf_counter_ns=queued_ns,
            ledger_started_perf_counter_ns=started_ns, receipt=source,
            ledger_start_source=source_link(events_path, output),
            clock_epoch_sanity_passed=True, clock_epoch_tolerance_seconds=1.0,
            clock_identity_limit='原凭据没有 host/boot_id；只验证同一本地输出树的匹配命令及 UTC/perf epoch 一致性。')
    except (OSError, ValueError, TypeError) as exc:
        return dict(unknown, receipt=source, reason=str(exc))


def attempt_row(directory, output, task, phase):
    warnings = []
    record = load(directory / 'trial_record.json', warnings)
    started = load(directory / 'trial_started.json', warnings) or {}
    sealed = record is not None
    data = record or started
    if sealed and (data.get('task') != task['task_name'] or data.get('phase') != phase):
        warnings.append('封存记录的任务或阶段与目录不一致')
    status = data.get('status') if sealed else 'unsealed'
    if status not in STATUS_LABELS:
        warnings.append('封存记录含未知状态')
        status = 'evidence_error'
    supervisor = load(directory / 'supervisor_timing/wall_timing.json', warnings)
    if supervisor is None and isinstance(data.get('supervisor_timing'), dict):
        supervisor = data['supervisor_timing']
    worker = load(directory / 'worker_timing/wall_timing.json', warnings)
    agent = load(directory / 'agent/agent_timing.json', warnings)
    if agent is None and isinstance(data.get('agent_timing'), dict):
        agent = data['agent_timing']
    closed = load(directory / 'episode_closed.json', warnings)
    if closed is None and isinstance(data.get('episode_closed'), dict):
        closed = data['episode_closed']
    closed = closed or {}
    steps = data.get('steps', closed.get('steps'))
    steps = steps if type(steps) is int and steps >= 0 else None
    success = data.get('official_success', closed.get('official_success'))
    success = success if type(success) is bool else None
    evidence = data.get('videos', closed.get('videos', {}))
    evidence = evidence if isinstance(evidence, dict) else {}
    metadata = evidence.get('videos', {})
    metadata = metadata if isinstance(metadata, dict) else {}
    videos = {}
    for role in ('robot_cameras', 'third_person'):
        meta = metadata.get(role, {})
        meta = meta if isinstance(meta, dict) else {}
        raw_path = meta.get('path')
        if not isinstance(raw_path, str) or not raw_path:
            raw_path = str(directory / 'sim/videos' / f'{role}.mp4')
        relative = safe_relative(raw_path, output)
        verified = bool(relative and evidence.get('passed') is True and meta.get('full_decode_passed') is True
                        and type(meta.get('actual_frames')) is int and meta.get('actual_frames') == steps)
        videos[role] = {'path': relative, 'full_decode_verified': verified,
                        'actual_frames': meta.get('actual_frames'),
                        'availability': 'available' if relative else 'not_available'}
        if sealed and steps and data.get('evaluation_valid') is True and not verified:
            warnings.append(f'{role} 缺少可访问的完整解码证据')
    if sealed and data.get('evaluation_valid') is True:
        if status not in ('task_success', 'task_failure') or success is None:
            warnings.append('有效评测标志与结果不一致')
        if status == 'task_success' and success is not True:
            warnings.append('任务成功缺少官方成功判定')
        if status == 'task_failure' and success is not False:
            warnings.append('任务未完成记录与官方判定不一致')
    elif status in ('task_success', 'task_failure'):
        warnings.append('任务结果未标记为有效评测')
    if sealed and warnings:
        status = 'evidence_error'
    times = {
        'supervisor_total_seconds': number((supervisor or {}).get('total_seconds')),
        'worker_startup_seconds': category(supervisor, 'worker_startup'),
        'agent_session_seconds': category(supervisor, 'agent_session'),
        'worker_finalize_seconds': category(supervisor, 'worker_finalize'),
        'agent_process_wall_seconds': number((agent or {}).get('agent_wall_seconds')),
        'worker_observation_exclusive_seconds': category(worker, 'observation'),
        'worker_action_execution_exclusive_seconds': category(worker, 'action_execution'),
        'worker_tool_dispatch_exclusive_seconds': category(worker, 'tool_dispatch'),
        'worker_initialization_exclusive_seconds': category(worker, 'initialization'),
        'worker_gpu_wait_exclusive_seconds': category(worker, 'gpu_wait'),
        'worker_video_finalize_exclusive_seconds': category(worker, 'video_finalize'),
    }
    normal_close = (sealed and data.get('evaluation_valid') is True
                    and status in ('task_success', 'task_failure')
                    and data.get('agent_exit_code') == 0 and data.get('worker_exit_code') == 0
                    and closed.get('status') == 'closed' and not data.get('supervisor_error'))
    events_path = directory / 'worker_timing/timing_events.jsonl'
    analysis = analyze_agent_tool_time(agent, worker,
        load_timing_events(events_path) if normal_close else None,
        sealed=sealed, normal_close=normal_close)
    experience = data.get('experience')
    experience_info = None
    if isinstance(experience, dict):
        # Describe existing provenance without reading the expert summary itself.
        experience_info = {key: experience.get(key) for key in ('sha256', 'bytes', 'sources_permitted')}
    # Read only launcher metadata. Never infer the old Codex settings from the
    # current launcher defaults or the unrelated Configuration.agent_effort.
    profile_warnings = []
    prepared = load(directory / 'agent/behavior-agent-prepared.json', profile_warnings) or {}
    profile = {key: (agent or {}).get(key, prepared.get(key)) for key in (
        'model_selection', 'model_reasoning_effort', 'model_service_tier_requested')}
    profile['server_service_tier_verified'] = (agent or {}).get(
        'server_service_tier_verified', prepared.get('server_service_tier_verified')) is True
    processes = load(directory / 'processes.json', profile_warnings) or {}
    profile['session_started'] = (data.get('agent_started') is True
        or type(processes.get('agent_pid')) is int and processes['agent_pid'] > 0
        or type((agent or {}).get('exit_code')) is int)
    profile['warnings'] = profile_warnings
    live_snapshot = None
    for path in sorted(directory.glob('live_time_breakdown_*.json'), reverse=True):
        optional_warnings = []
        live = load(path, optional_warnings)
        if live and live.get('status') == 'live_snapshot_unsealed':
            live_snapshot = {'json': source_link(path, output),
                'description': source_link(path.with_suffix('.md'), output),
                'cutoff_at_utc': live.get('cutoff_at_utc'),
                'cutoff_Asia_Shanghai': live.get('cutoff_Asia_Shanghai'),
                'not_final_measurement': True}
            break
    reasons = data.get('reasons', [])
    reasons = reasons if isinstance(reasons, list) else [str(reasons)]
    return {'task': task['task_name'], 'phase': phase, 'attempt': data.get('attempt', directory.name[len(phase)+1:]),
        'status': status, 'recorded_status': data.get('status'), 'sealed': sealed,
        'evaluation_valid': data.get('evaluation_valid') is True and status in ('task_success', 'task_failure'),
        'sealed_official_success': (normal_close and status == 'task_success'
            and closed.get('official_success') is True and closed.get('steps') == steps
            and type(steps) is int),
        'official_success': success, 'steps': steps,
        'started_at_utc': data.get('started_at_utc'), 'ended_at_utc': data.get('ended_at_utc'),
        'directory_modified_at_utc': datetime.fromtimestamp(directory.stat().st_mtime, timezone.utc).isoformat(),
        'termination_reason': closed.get('termination_reason'),
        'agent_finish_statement': closed.get('agent_finish'),
        'agent_status': (agent or {}).get('status'),
        'directory': directory.relative_to(output).as_posix(), 'timing': times,
        'agent_timing_analysis': analysis, 'videos': videos,
        'model_profile': profile, 'latest_live_time_snapshot': live_snapshot,
        'runner_queue_time': queue_to_ledger_time(directory, output, task, phase),
        'experience': experience_info, 'reasons': reasons, 'evidence_warnings': warnings,
        'sources': {key: source_link(directory / relative, output) for key, relative in (
            ('trial_record', 'trial_record.json'), ('worker_closed', 'episode_closed.json'),
            ('supervisor_timing', 'supervisor_timing/wall_timing.json'),
            ('worker_timing', 'worker_timing/wall_timing.json'),
            ('worker_timing_events', 'worker_timing/timing_events.jsonl'),
            ('failure_analysis', 'failure_analysis.md'),
            ('failure_analysis_json', 'failure_analysis.json'),
            ('final_time_breakdown', 'final_time_breakdown.md'),
            ('final_time_breakdown_json', 'final_time_breakdown.json'),
            ('agent_timing', 'agent/agent_timing.json'))}}


def model_profile_comparison(zero, retry):
    """Compare recorded sessions, never future launcher defaults or expert text."""
    rows = {'zero_shot': zero, 'one_expert_retry': retry}
    result = {'basis_directories': {phase: row['directory'] if row else None
                                  for phase, row in rows.items()},
              'scope': '各阶段最新尝试的会话启动配置；请求服务层不等于服务端已实现服务层。'}
    pending = [PHASE_LABELS[phase] for phase, row in rows.items()
               if row is None or not row['model_profile']['session_started']]
    if pending:
        return dict(result, status='pending', differing_fields=[], unknown_fields=[],
            message='尚无' + '、'.join(pending) + '的实际会话启动证据，暂不能比较两阶段模型配置；'
                    '不将当前默认配置或准备记录视为已经切换运行。')
    fields = {'model_selection': '模型', 'model_reasoning_effort': '推理强度',
              'model_service_tier_requested': '请求服务层'}
    profiles = [row['model_profile'] for row in (zero, retry)]
    unknown, different = [], []
    details = []
    for key, label in fields.items():
        values = [p.get(key) for p in profiles]
        if any(not isinstance(value, str) or not value.strip() for value in values):
            unknown.append(key)
        elif values[0] != values[1]:
            different.append(key)
            details.append(f'{label}：零样本 {values[0]} → 单专家重试 {values[1]}')
    if unknown:
        details.append('至少一阶段的' + '、'.join(fields[key] for key in unknown) + '未知/未显式指定，不能视为一致')
    if not all(p.get('server_service_tier_verified') is True for p in profiles):
        unknown.append('server_service_tier')
        details.append('至少一阶段的服务端服务层未证实')
    status = 'different' if different else 'unknown' if unknown else 'matching_recorded_config'
    details.append('不能把结果或耗时差异单独归因于专家经验。' if different or unknown else
                   '记录中的模型配置一致，但单次前后对比仍不能单独建立专家经验的因果效果。')
    if not all(row['sealed'] for row in (zero, retry)):
        details.append('至少一阶段尚未封存，此提示不是最终结果对比。')
    return dict(result, status=status, differing_fields=different, unknown_fields=unknown,
                message='；'.join(details))


def build_report(output):
    output = Path(output).expanduser().resolve()
    warnings = []
    protocol = load(output / 'protocol.json', warnings, required=True) or {}
    summary = load(output / 'summary.json', warnings) or {}
    tasks = protocol.get('tasks', [])
    tasks = tasks if isinstance(tasks, list) else []
    rows = []
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get('task_name'), str):
            warnings.append('protocol 包含无效任务条目')
            continue
        # Canonical task names cannot introduce traversal when reporting local files.
        name = task['task_name']
        if not name or any(char not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for char in name):
            warnings.append('protocol 包含非规范任务名称')
            continue
        attempts = []
        phases, latest_by_phase = {}, {}
        for phase in PHASES:
            candidates = [d for d in sorted((output / name).glob(phase + '_*')) if d.is_dir() and not d.is_symlink()]
            items = [attempt_row(d, output, task, phase) for d in candidates]
            items.sort(key=lambda row: (row['started_at_utc'] or row['directory_modified_at_utc'], row['directory']))
            attempts.extend(items)
            latest = items[-1] if items else None
            latest_by_phase[phase] = latest
            phases[phase] = {'status': latest['status'] if latest else 'not_run',
                             'latest_directory': latest['directory'] if latest else None,
                             'latest_steps': latest['steps'] if latest else None,
                             'latest_total_wall_seconds': latest['timing']['supervisor_total_seconds'] if latest else None,
                             'attempts': len(items)}
        expert = expert_stages(output, task)
        zero = latest_by_phase['zero_shot']
        if (zero is not None and zero['sealed'] and zero['evaluation_valid']
                and zero['status'] == 'task_success' and zero['official_success'] is True
                and zero['sealed_official_success']
                and latest_by_phase['one_expert_retry'] is None):
            reason = '最新零样本已封存为有效官方成功，无需单专家重试。'
            phases['one_expert_retry'].update(status='not_required', reason=reason,
                                             basis_directory=zero['directory'])
            episode = expert['fixed_episode_index']
            directory = output / name / f'expert_episode_{episode}'
            # Any existing expert directory may hold partial/error evidence. Do
            # not replace those stages, or read their demonstration contents.
            if (type(episode) is int and episode >= 0
                    and not directory.exists() and not directory.is_symlink()):
                for stage in ('export', 'analysis'):
                    if expert[stage]['status'] == 'pending':
                        expert[stage].update(status='not_required', reason=reason)
        rows.append({'task_name': name, 'task_index': task.get('task_index'),
                     'label': TASK_LABELS.get(name, name), 'prompt': task.get('prompt', ''),
                     'phases': phases, 'attempts': attempts, 'expert_stages': expert,
                     'model_profile_comparison': model_profile_comparison(
                         zero, latest_by_phase['one_expert_retry'])})
    counts = Counter(row['status'] for task in rows for row in task['attempts'])
    return {'schema_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'title': 'SeekAct × BEHAVIOR 原场景实验',
        'protocol': {key: protocol.get(key) for key in ('render_profile', 'variant', 'split', 'instance_id', 'seed', 'success_source')},
        'protocol_link': source_link(output / 'protocol.json', output),
        'summary_link': source_link(output / 'summary.json', output),
        'network_diagnostic_link': source_link(output / 'network_path_diagnostic.json', output),
        'runner_active_as_last_recorded': summary.get('active'),
        'runner_summary_updated_at_utc': summary.get('updated_at_utc'),
        'scope': '仅 protocol 任务的正式 zero_shot / one_expert_retry 目录；校准与其他历史实验不计入。',
        'counts_by_attempt': dict(counts),
        'not_run_task_phases': sum(p['status'] == 'not_run' for task in rows for p in task['phases'].values()),
        'not_required_task_phases': sum(p['status'] == 'not_required' for task in rows for p in task['phases'].values()),
        'tasks': rows, 'timing_note': TIMING_NOTE, 'warnings': warnings,
        'runner_queue_time': {'status': 'per_trial_records' if any(
                row['runner_queue_time']['status'] == 'measured' for task in rows for row in task['attempts'])
                else 'not_recorded', 'wall_seconds': None,
            'scope': '排队到 ledger 的区间按本轮匹配凭据单独展示，不汇总或加入运行墙钟；缺失为未知。worker gpu_wait 已含在运行墙钟中。'},
        'expert_text_read_by_report': False, 'tasks_started_by_report': 0}


def esc(value):
    return html.escape(str(value), quote=True)


def href(relative):
    return quote(relative, safe='/')


def seconds(value):
    value = number(value)
    if value is None:
        return '—'
    return f'{value:.2f} 秒' if value < 60 else f'{value / 60:.2f} 分钟（{value:.2f} 秒）'


def badge(status):
    css = 'success' if status in ('task_success', 'complete', 'not_required') else 'failure' if status == 'task_failure' else 'error' if 'error' in status else 'pending'
    return f'<span class="badge {css}">{esc(STATUS_LABELS.get(status, status))}</span>'


def official(value):
    return '已成功' if value is True else '未成功' if value is False else '未取得判定'


def link(relative, label):
    return f'<a href="{esc(href(relative))}">{esc(label)}</a>' if relative else ''


def video_panel(role, meta, steps):
    title = {'robot_cameras': '机器人相机', 'third_person': '第三人称'}[role]
    title += '全过程' if meta['full_decode_verified'] else '执行录像（尚未完整验收）'
    if not meta['path']:
        reason = '该轮没有执行动作，因此没有可播放动作视频。' if steps == 0 else '尚未生成或未找到视频；不以缺失视频推断任务结果。'
        return f'<div class="video"><h4>{title}</h4><p class="muted">{reason}</p></div>'
    status = '完整解码已验收' if meta['full_decode_verified'] else '视频存在，完整解码尚未验收'
    return (f'<div class="video"><h4>{title}</h4><video controls preload="metadata" playsinline '
            f'src="{esc(href(meta["path"]))}"></video><p>{esc(status)} · '
            f'{link(meta["path"], "打开视频")}</p></div>')


def phase_cell(phase):
    extra = ''
    if phase['latest_directory']:
        steps = phase['latest_steps']
        extra = '<small>' + (esc(steps) + ' 步 · ' if steps is not None else '') + \
                '执行总墙钟：' + seconds(phase['latest_total_wall_seconds']) + '</small>'
    return '<td>' + badge(phase['status']) + extra + '</td>'


def expert_panel(expert):
    if expert['fixed_episode_index'] is None:
        return ''
    parts = ['<details class="expert"><summary>单专家准备：导出与经验分析（独立计时）</summary>',
             '<p>协议固定 episode_index：<strong>' + esc(expert['fixed_episode_index']) +
             '</strong>；已记录 source_count：<strong>' + esc(expert['source_count'] if expert['source_count'] is not None else '—') +
             '</strong>。有效零样本已成功且专家阶段未开始时无需重试；其余没有完成标记的阶段保持待完成，不从视频或摘要文件的存在推断分析已结束。</p>',
             '<div class="table-wrap"><table><thead><tr><th>准备阶段</th><th>状态</th><th>真实墙钟</th></tr></thead><tbody>']
    for key, label in (('export', '导出 / 传输 / 完整解码验收'), ('analysis', '单条专家经验分析')):
        stage = expert[key]
        parts.append('<tr><td>' + label + '</td><td>' + badge(stage['status']) + '</td><td>' +
                     seconds(stage['wall_seconds']) + '</td></tr>')
    parts.append('</tbody></table></div><p class="muted">' + esc(expert['timing_scope']) + '</p>')
    parts.append('<p class="muted">' + esc(expert['analysis_scope']) + '</p>')
    if expert['analysis'].get('scope_recorded'):
        parts.append('<p class="muted">分析标记中的计时范围：' + esc(expert['analysis']['scope_recorded']) + '</p>')
    sub = expert['export'].get('phases_exclusive_seconds')
    if sub:
        parts.append('<p class="muted">导出内部 exclusive 分段：' + '；'.join(label + ' ' + seconds(sub.get(key))
                     for key, label in (('expert_export', '远端提取 / 编码及请求等待'),
                                        ('expert_transfer', '传输'), ('expert_verification', '本地校验'))) + '。</p>')
    warnings = expert['export']['reasons'] + expert['analysis']['reasons']
    if warnings:
        parts.append('<ul class="warnings">' + ''.join('<li>' + esc(value) + '</li>' for value in warnings) + '</ul>')
    if expert['metadata_links']:
        parts.append('<p>' + ' · '.join(link(path, name) for name, path in expert['metadata_links'].items()) + '</p>')
    if expert['experience_link']:
        parts.append('<p>' + link(expert['experience_link'], '查看单专家经验文本 experience.md') +
                     ' · 文本 SHA256（由分析标记记录，报告未重读正文）：<code>' +
                     esc(expert['experience_sha256_reported']) + '</code></p>')
    if expert['source_file_sha256']:
        parts.append('<p>单专家导出文件 SHA256（来源元数据，报告未读取这些文件内容）：</p><dl>' +
                     ''.join('<dt>' + esc(name) + '</dt><dd><code>' + esc(digest) + '</code></dd>'
                             for name, digest in expert['source_file_sha256'].items()) + '</dl>')
    if expert['expert_video_links']:
        names = {'head': '头部相机', 'left_wrist': '左腕相机', 'right_wrist': '右腕相机'}
        parts.append('<p><strong>以下为单专家示范，不是本次机器人执行视频：</strong> ' + ' · '.join(
            link(meta['path'], names[role]) for role, meta in expert['expert_video_links'].items()) + '</p>')
    parts.append('</details>')
    return ''.join(parts)


def html_report(report):
    overview = []
    details = []
    for task in report['tasks']:
        phases = task['phases']
        overview.append(f'<tr><td><a href="#{esc(task["task_name"])}">{esc(task["label"])}</a>'
                        f'<small>{esc(task["task_name"])} · ID {esc(task["task_index"])}</small></td>'
                        + phase_cell(phases['zero_shot']) + phase_cell(phases['one_expert_retry']) + '</tr>')
        section = [f'<section id="{esc(task["task_name"])}"><h2>{esc(task["label"])}</h2>'
                   f'<p class="prompt">{esc(task["prompt"])}</p>']
        section.append('<p class="notice"><strong>两阶段可比性：</strong>' +
                       esc(task['model_profile_comparison']['message']) + '</p>')
        section.append(expert_panel(task['expert_stages']))
        if not task['attempts']:
            section.append('<p class="empty">尚无正式实验记录。经验重试只会在有效零样本失败后进行。</p>')
        for trial in task['attempts']:
            t = trial['timing']
            section.append(f'<article><div class="attempt-heading"><h3>{esc(PHASE_LABELS[trial["phase"]])} · '
                           f'{esc(trial["attempt"])}</h3>{badge(trial["status"])}</div>'
                           f'<p>官方 BDDL：<strong>{official(trial["official_success"])}</strong> · '
                           f'动作步数：<strong>{esc(trial["steps"] if trial["steps"] is not None else "—")}</strong> · '
                           f'调度总墙钟：<strong>{seconds(t["supervisor_total_seconds"])}</strong></p>')
            queue = trial['runner_queue_time']
            section.append('<p class="muted">排队起点到本轮 ledger 启动：' +
                (seconds(queue['wall_seconds']) if queue['status'] == 'measured' else '未知') +
                '；含 launcher 开销，不是纯锁等待，未加入上述运行墙钟。' +
                ((' ' + link(queue['receipt'], '匹配排队凭据')) if queue.get('receipt') else '') + '</p>')
            profile = trial['model_profile']
            section.append('<p class="muted">本轮模型：' + esc(profile['model_selection'] or '未记录') +
                '；推理强度：' + esc(profile['model_reasoning_effort'] or '未显式指定') +
                '；请求服务层：' + esc(profile['model_service_tier_requested'] or '未显式指定') +
                '；服务端服务层' + ('已验证' if profile['server_service_tier_verified'] else '未证实') +
                '。这些字段来自本轮启动记录，不回填当前默认配置。</p>')
            live = trial['latest_live_time_snapshot']
            if live:
                section.append('<p class="notice">已保存进行中计时快照（截至 ' +
                    esc(live['cutoff_Asia_Shanghai'] or live['cutoff_at_utc'] or '记录中的截点') +
                    '，不是最终计时）：' + ' · '.join(filter(None, (
                        link(live['description'], '时间占比与耗时原因说明'),
                        link(live['json'], '原始实测快照 JSON')))) + '。</p>')
            if trial['status'] not in ('task_success', 'task_failure') and trial['official_success'] is not None:
                section.append('<p class="muted">这里保留官方原始状态；本轮未通过完整有效评测验收，不计入成功率。</p>')
            if trial['termination_reason']:
                section.append('<p>仿真记录的终止原因：<code>' + esc(trial['termination_reason']) + '</code>。</p>')
            statement = trial['agent_finish_statement']
            if isinstance(statement, dict) and statement.get('reason'):
                section.append('<p class="muted">Agent 自述（不作为成功判定）：' + esc(statement['reason']) + '</p>')
            if trial['reasons'] or trial['evidence_warnings']:
                reasons = [REASONS.get(reason, reason) for reason in trial['reasons']] + trial['evidence_warnings']
                section.append('<ul class="warnings">' + ''.join(f'<li>{esc(reason)}</li>' for reason in reasons) + '</ul>')
            section.append('<div class="table-wrap"><table><thead><tr><th>外层初始化 / 等 GPU</th><th>Agent 会话</th>'
                           '<th>外层收尾</th><th>Agent 进程墙钟</th></tr></thead><tbody><tr>' +
                           ''.join(f'<td>{seconds(t[key])}</td>' for key in ('worker_startup_seconds', 'agent_session_seconds',
                               'worker_finalize_seconds', 'agent_process_wall_seconds')) + '</tr></tbody></table></div>')
            analysis = trial['agent_timing_analysis']
            section.append('<h4 style="margin-top:20px">Agent 进程区间分解</h4>')
            if analysis['valid']:
                section.append('<p class="muted">完整封存、跨度配对与时钟一致性检查通过。工具墙钟取 Agent 区间内 '
                               'tool_dispatch 的交集并集，嵌套/重叠跨度只计一次。</p>')
                section.append('<div class="table-wrap"><table><thead><tr><th>工具执行墙钟（含子跨度）</th>'
                               '<th>其余墙钟（非纯 LLM）</th><th>其中：观察 / 图像采集</th>'
                               '<th>其中：物理动作执行</th></tr></thead><tbody><tr>' +
                               ''.join('<td>' + seconds(analysis[key]) + '</td>' for key in (
                                   'tool_dispatch_intersection_union_seconds', 'agent_residual_seconds',
                                   'observation_intersection_union_seconds', 'action_execution_intersection_union_seconds')) +
                               '</tr></tbody></table></div><p class="muted">' + esc(analysis['residual_scope']) + '。' +
                               esc(analysis['subspan_scope']) + '。工具并集 + 其余墙钟 = Agent 进程墙钟（纳秒精确核对）。</p>')
            else:
                label = '待封存' if analysis['status'] == 'pending' else '未知'
                section.append('<p class="muted">时间分解：' + label + '。' +
                               esc('；'.join(analysis['reasons'])) + '。不将缺失区间当作零，也不按动作步数估计。</p>')
            section.append('<details><summary>查看 worker 实测分段与证据</summary><p class="muted">下列为 worker 层的 exclusive 墙钟，'
                           '仅描述各段自身占用。它们与上面的外层/Agent 时间重叠，不可跨层求和。</p><dl>' +
                           ''.join(f'<dt>{label}</dt><dd>{seconds(t[key])}</dd>' for key, label in (
                               ('worker_gpu_wait_exclusive_seconds', '等待 GPU'),
                               ('worker_initialization_exclusive_seconds', '仿真初始化'),
                               ('worker_observation_exclusive_seconds', '观察 / 图像采集'),
                               ('worker_action_execution_exclusive_seconds', '物理动作执行'),
                               ('worker_tool_dispatch_exclusive_seconds', '工具分发自身开销'),
                               ('worker_video_finalize_exclusive_seconds', '视频收尾 / 解码验收'))) + '</dl>')
            section.append('<p class="links">' + ' · '.join(link(value, key) for key, value in trial['sources'].items() if value) + '</p>')
            if trial['experience']:
                section.append('<p>已记录本轮指定经验文本的来源校验：SHA256 <code>' +
                               esc(trial['experience'].get('sha256')) + '</code>。报告未读取其内容，也不推断专家分析已完成。</p>')
            section.append('</details><div class="videos">' + ''.join(video_panel(role, meta, trial['steps'])
                           for role, meta in trial['videos'].items()) + '</div></article>')
        section.append('</section>')
        details.append(''.join(section))
    counts = report['counts_by_attempt']
    counters = [('有效成功', counts.get('task_success', 0)), ('有效未完成', counts.get('task_failure', 0)),
                ('基础设施 / 证据异常', sum(counts.get(s, 0) for s in ('agent_infrastructure_error', 'infrastructure_error', 'evidence_error'))),
                ('尚未封存', counts.get('unsealed', 0)), ('未运行的任务×阶段', report['not_run_task_phases'])]
    protocol = report['protocol']
    warning = '<ul class="warnings">' + ''.join('<li>' + esc(w) + '</li>' for w in report['warnings']) + '</ul>' if report['warnings'] else ''
    active = report.get('runner_active_as_last_recorded')
    active_text = ('调度最近记录：' + json.dumps(active, ensure_ascii=False) + '；这是保存时的状态，不代表当前进程仍运行。') if active else '没有记录正在执行的正式任务。'
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SeekAct · BEHAVIOR 实验报告</title><style>
:root{color-scheme:light;--ink:#162532;--muted:#617383;--line:#dce4e9;--blue:#17618e}*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:15px/1.65 system-ui,sans-serif}main{max-width:1320px;margin:auto;padding:32px 24px 64px}header{padding:24px 0 16px}h1{font-size:30px;line-height:1.3;margin:8px 0 14px}h2{font-size:24px;margin:0 0 12px}h3,h4{margin:0}h3{font-size:19px}h4{font-size:16px;margin-bottom:9px}.eyebrow{color:var(--blue);font-weight:650;letter-spacing:.09em;font-size:12px}a{color:var(--blue);text-underline-offset:3px}.muted,small{color:var(--muted)}small{display:block;font-size:12px}.notice{background:#e8f1f6;border-left:3px solid var(--blue);padding:14px 18px}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:24px 0}.stat{background:white;padding:16px;border:1px solid var(--line);border-radius:10px}.stat strong{display:block;font-size:28px}.stat span{font-size:13px;color:var(--muted)}section,.overview{background:white;border:1px solid var(--line);border-radius:12px;padding:24px;margin:24px 0}article{padding-top:22px;margin-top:22px;border-top:1px solid var(--line)}.attempt-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:13px;white-space:nowrap}.success{background:#ddf2e6;color:#175f39}.failure{background:#fff0d4;color:#85570f}.error{background:#fbe3e3;color:#9a2d2d}.pending{background:#edf1f4;color:#516573}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;color:var(--muted);font-weight:550;background:#f6f8fa}td,th{padding:12px;border-bottom:1px solid var(--line);vertical-align:top}.videos{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:20px}.video{min-width:0;background:#f6f8fa;padding:14px;border-radius:8px}.video p{font-size:13px;margin-bottom:0}video{display:block;width:100%;max-height:420px;aspect-ratio:16/9;background:#10171d;border-radius:4px}.prompt{color:var(--muted);max-width:950px}.empty{padding:20px;background:#f6f8fa}.warnings{color:#963b28}details{margin-top:16px}summary{cursor:pointer;color:var(--blue)}dl{display:grid;grid-template-columns:250px 1fr;font-size:14px}dt,dd{margin:0;padding:5px 0}code{word-break:break-all;font-size:12px}.links{font-size:13px}footer{font-size:13px;color:var(--muted)}@media(max-width:850px){.stats{grid-template-columns:repeat(2,1fr)}.videos{grid-template-columns:1fr}main{padding:16px}section,.overview{padding:17px}h1{font-size:25px}dl{grid-template-columns:1fr 1fr}}
</style></head><body><main><header><div class="eyebrow">SEEKACT-AGENT / BEHAVIOR R1PRO</div><h1>原场景任务实验报告</h1>''' + \
        f'<p>场景：{esc(protocol.get("variant"))} · split：{esc(protocol.get("split"))} · instance：{esc(protocol.get("instance_id"))} · 渲染：{esc(protocol.get("render_profile"))}</p>' + \
        f'<p class="muted">报告生成时间（UTC）：{esc(report["generated_at_utc"])}。{esc(active_text)}</p></header>' + warning + \
        '<div class="notice">最终任务成功只取官方 BDDL；搜索接近、动作 ACK 或 Agent 自述均不替代成功。零样本与单专家经验重试分别报告；基础设施故障不计能力失败。</div>' + \
        '<div class="stats">' + ''.join(f'<div class="stat"><strong>{value}</strong><span>{label}</span></div>' for label, value in counters) + '</div>' + \
        '<div class="overview"><h2>三任务对照</h2><p class="muted">此表显示各阶段最新记录；下方保留所有正式尝试，未挑选成功记录。</p>' + \
        '<div class="table-wrap"><table><thead><tr><th>任务</th><th>零样本</th><th>单专家经验重试</th></tr></thead><tbody>' + ''.join(overview) + \
        '</tbody></table></div></div><div class="notice">' + esc(TIMING_NOTE) + '</div>' + ''.join(details) + \
        '<footer><p>' + esc(report['scope']) + '</p><p>' + ' · '.join(filter(None, (
            link(report['protocol_link'], '实验协议'), link(report['summary_link'], '调度汇总'),
            link(report.get('network_diagnostic_link'), '网络路径实测快照'), link('report.json', '报告数据')))) + \
        '</p><p>该页面只读取已有证据；不启动任务，不读取专家文本，不自动判断新的视频内容。</p></footer></main></body></html>\n'


def write_report(output):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = build_report(output)
    for name, text in (('report.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('index.html', html_report(report))):
        temporary = output / (name + '.tmp')
        temporary.write_text(text)
        temporary.replace(output / name)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = write_report(args.output)
    print(json.dumps({'report': str(args.output.resolve() / 'index.html'),
                      'counts_by_attempt': report['counts_by_attempt'],
                      'not_run_task_phases': report['not_run_task_phases']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
