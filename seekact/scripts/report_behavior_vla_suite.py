#!/usr/bin/env python3
"""Report the new three-task tool-augmented study, without expert-retry gates."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from urllib.parse import quote

import report_behavior_experiment as timing_report


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError, ValueError):return None


def write_report(root):
    root=Path(root);protocol=read(root/'protocol_current.json') or read(root/'protocol_toy_resolved.json') or read(root/'protocol.json')
    if protocol is None:return
    native_toy=any(t['task_name']=='collecting_childrens_toys' and not t.get('champion_cross_task_transfer') for t in protocol['tasks'])
    strict=bool(protocol.get('vla_tools',{}).get('vla_only'))
    control_note=('GPT仅观察并调度：AP搜索接近，声明视觉找到目标后交接冠军执行。直接底盘、手臂、夹爪和躯干控制被隐藏且由仿真端硬拒绝。' if strict else '同一上层可选择两个学习控制器及原有机器人恢复工具。')
    if protocol.get('vla_tools',{}).get('closed_loop'):
        control_note='反复视觉闭环调度：每段步数由GPT决定，无120步上限（保留任务总预算），GPT检查新画面、记录进展与子目标，再选择AP或冠军；连续冠军停滞要求AP恢复。原生直接控制硬禁用，成功只认官方BDDL。'
    if protocol.get('vla_tools',{}).get('interaction_only'):
        control_note='AP负责搜索、接近与转移；冠军只按当前交互分段执行。每段后GPT检查新画面并判定完成/继续/切AP；完成后必须先执行AP。取消总步数预算，每任务最长3600秒墙钟（含初始化及模型等待，不含视频封口）。'
    if protocol.get('vla_tools',{}).get('qwen_monitor'):
        control_note='AP独占搜索、接近与转移；冠军仅执行当前局部交互。每批冠军动作后Qwen检查当前帧及交互目标，连续两次已完成立即交回GPT，并要求先执行AP。无任务总步数限制、无实验超时；成功只认官方BDDL。视频青色放大镜为AP，金色星形为冠军。'
    toy_note=('当前玩具项已按用户确认改为冠军原生支持的collecting_childrens_toys21；鞋子和收音机保留原有效试验。旧task54迁移试验只作历史记录。' if native_toy else '玩具原任务54不在冠军表内；已说明的操作默认使用相关技能迁移，非冠军原生支持。')
    if strict:toy_note='本轮任务按当前协议列表执行，均为冠军原生支持；每条从新场景启动，不沿用旧结果。'
    rows=[]
    for task in protocol['tasks']:
        name=task['task_name'];attempts=[]
        for directory in sorted((root/name).glob('zero_shot_*')):
            record=read(directory/'trial_record.json');progress=read(directory/'sim/progress.json')
            raw=directory/'execution.jsonl';counts=Counter()
            if raw.exists():
                for line in raw.read_text().splitlines():
                    try:r=json.loads(line)
                    except ValueError:continue
                    if r.get('event')=='submitted':counts[r.get('owner',{}).get('controller','native_tools')]+=1
            entry={'directory':str(directory.relative_to(root)),'sealed':record is not None,
                'status':record['status']if record else 'running','steps':record['steps']if record else (progress or {}).get('step'),
                'official_success':record['official_success']if record else None,'controller_steps':dict(counts)}
            if record:
                entry.update(evaluation_valid=record['evaluation_valid'],wall_seconds=record['supervisor_timing']['total_seconds'],
                    vla_evidence=record['episode_closed'].get('vla_evidence'),vla_tools=record.get('vla_tools'))
                normal=record['agent_exit_code']==record['worker_exit_code']==0
                entry['agent_time']=timing_report.analyze_agent_tool_time(read(directory/'agent/agent_timing.json'),
                    read(directory/'worker_timing/wall_timing.json'),timing_report.load_timing_events(directory/'worker_timing/timing_events.jsonl'),
                    sealed=True,normal_close=normal)
                entry['videos']=record.get('videos',{})
            attempts.append(entry)
        rows.append({'task':name,'task_index':task['task_index'],'champion_skills':task.get('champion_skills',{}),
            'champion_cross_task_transfer':task.get('champion_cross_task_transfer',False),'attempts':attempts})
    status=read(root/'suite_status.json') or read(root/'study_status.json') or {}
    superseded=read(root/'cohort_superseded.json')
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'planned_tasks':len(rows),'tasks':rows,
        'suite_status':status,'models':read(root/'models_ready.json'),'not_pure_llm_time':True,
        'video_wall_time_note':'30fps物理过程视频不包含冻结的模型/网络等待；墙钟单独计量。'}
    (root/'vla_usage.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    esc=lambda x:html.escape(str(x))
    url=lambda x:quote(str(x),safe='/')
    parts=['<!doctype html><meta charset="utf-8"><title>SeekAct + AP / Champion VLA</title>',
        '<style>body{font:16px system-ui;margin:32px auto;max-width:1350px;padding:0 20px;background:#f7f8fb;color:#172033}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border:1px solid #d7deea;text-align:left}video{width:48%;background:#111;margin:1%}article{background:white;padding:20px;margin:24px 0;border-radius:12px}small{color:#59657a}a{color:#1764c0}</style>',
        '<h1>SeekAct：主动感知 VLA + 冠军 VLA 工具实验</h1>',
        '<p>'+html.escape(control_note)+'原场景 public_test/301，seed20261020，fast；成功只认官方 BDDL。每个任务计划一条有效试验，不以能力失败重跑凑成功。</p>',
        '<p>模型：gpt-6-astra / low / Fast requested（服务端tier未验证）。AP checkpoint024000支持限定语义；冠军checkpoint2按固定任务ID控制，不理解自由语言。'+html.escape(toy_note)+'</p>',
        '<p>运行状态：'+esc(status.get('state','准备中'))+' '+esc(status.get('task',''))+'</p>',
        '<p><a href="final_acceptance.json">最终验收汇总</a> · <a href="vla_usage.json">完整控制器用量和计时JSON</a> · <a href="protocol_current.json">当前任务协议</a> · <a href="models_ready.json">真实模型推理烟测</a> · <a href="execution_audit.json">原始动作与视频审计</a> · <a href="index_cross_task_history.html">旧迁移试验历史</a></p>',
        '<table><tr><th>任务</th><th>状态 / 官方成功</th><th>步数</th><th>实际墙钟（秒）</th><th>各控制器步数</th></tr>']
    for row in rows:
        a=row['attempts'][-1]if row['attempts']else{}
        parts.append('<tr><td>'+esc(row['task'])+'</td><td>'+esc(a.get('status','未开始'))+' / '+esc(a.get('official_success','—'))+'</td><td>'+esc(a.get('steps','—'))+'</td><td>'+esc(round(a['wall_seconds'],3)if 'wall_seconds'in a else '未封口')+'</td><td>'+esc(a.get('controller_steps',{}))+'</td></tr>')
    parts.append('</table><p><small>基础设施故障单列；动作步数占比不等于墙钟占比。模型外部等待包含网络及编排，不能称纯推理耗时。'+esc(summary['video_wall_time_note'])+'</small></p>')
    if superseded:
        parts.insert(4,'<p><strong>此批次已因用户更改控制协议而停止，仅作历史记录。</strong> <a href="../seekact_vla_only_20261024/index.html">查看新的纯VLA调度三任务试验</a></p>')
    for row in rows:
        for a in row['attempts']:
            d=a['directory'];parts.append('<article><h2>'+esc(row['task'])+' / '+esc(Path(d).name)+'</h2><p>冠军技能：'+esc(row['champion_skills'])+'；跨任务迁移：'+esc(row['champion_cross_task_transfer'])+'</p>')
            if a['sealed']:
                parts.append('<p><a href="'+url(d+'/trial_record.json')+'">验收</a> · <a href="'+url(d+'/agent/agent_messages.md')+'">上层执行摘要</a> · <a href="'+url(d+'/vla/summary.json')+'">VLA调用记录</a> · <a href="'+url(d+'/worker_timing/wall_timing.json')+'">工具真实计时</a></p>')
                evidence=a.get('vla_evidence')or{};decisions=evidence.get('decisions',[])
                if decisions:
                    switches=sum(x['kind']!=y['kind']for x,y in zip(decisions,decisions[1:]))
                    parts.append('<p>逐段视觉决策：'+esc(len(decisions))+'次；AP/冠军切换：'+esc(switches)+'次。<a href="'+url(d+'/vla/summary.json')+'">查看每段步数、进展、理由与下一子目标</a></p>')
                time=a['agent_time'];parts.append('<p>工具区间并集（含VLA等待）：'+esc(time.get('tool_dispatch_intersection_union_seconds'))+'秒；Agent其余处理/等待：'+esc(time.get('agent_residual_seconds'))+'秒。两者在Agent区间内不重叠。</p>')
                if a.get('videos',{}).get('playable_video_available'):
                    for role in ('third_person','robot_cameras'):
                        annotated=root/d/'sim/videos_annotated'/ (role+'.mp4')
                        annotation=read(root/d/'video_annotation.json')
                        use_annotated=bool(annotation and annotation.get('status')=='verified' and annotated.is_file())
                        relative=d+('/sim/videos_annotated/'if use_annotated else '/sim/videos/')+role+'.mp4'
                        parts.append('<video controls preload="metadata" src="'+url(relative)+'"></video>')
                        if use_annotated:parts.append('<p>青色放大镜：AP；金色星形：冠军。<a href="'+url(d+'/sim/videos/'+role+'.mp4')+'">未标注原片</a> · <a href="'+url(d+'/video_annotation.json')+'">标注对齐与完整性</a></p>')
            else:parts.append('<p>执行尚未封口，暂不发布未完成的MP4。</p>')
            parts.append('</article>')
    temp=root/'index.html.tmp';temp.write_text('\n'.join(parts));temp.replace(root/'index.html')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();write_report(args.output)
