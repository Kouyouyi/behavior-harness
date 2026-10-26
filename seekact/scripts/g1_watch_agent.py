"""Read-only one-second view of agent tool decisions and the unified G1 owner."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError,ValueError):
        return None


def short(value,limit=220):
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
    # Do not let logged task text inject terminal control sequences.
    text=' '.join(''.join(c for c in text if c.isprintable() or c=='\n').split())
    return text[:limit]+('…' if len(text)>limit else '')


def event_line(row):
    kind=row.get('event');tool=row.get('tool','')
    if kind not in ('mcp_request','mcp_result','mcp_error'):
        return None
    stamp=time.strftime('%H:%M:%S',time.localtime(row.get('time',0)))
    if kind=='mcp_request':
        return f'{stamp} 调用 {tool} {short(row.get("arguments",{}))}'
    if kind=='mcp_error':
        return f'{stamp} 错误 {tool} {short(row.get("error",row))}'
    value=row.get('result',{})
    if isinstance(value,str):
        try:value=json.loads(value)
        except ValueError:pass
    if isinstance(value,dict):
        fields=('job_id','completed','reason','status','frame_id','steps_published')
        selected={k:value[k] for k in fields if k in value}
        if not selected:return None
        value=selected
    return f'{stamp} 返回 {tool} {short(value)}'


class AuditTail:
    def __init__(self,path):
        self.path=path;self.offset=0;self.identity=None
        self.lines=deque(maxlen=12)

    def poll(self):
        try:
            stat=self.path.stat();identity=(stat.st_dev,stat.st_ino)
            if identity!=self.identity or stat.st_size<self.offset:
                self.offset=max(0,stat.st_size-1024*1024);self.identity=identity
                if self.offset:
                    with self.path.open('rb') as f:
                        f.seek(self.offset);f.readline();self.offset=f.tell()
            with self.path.open('rb') as f:
                f.seek(self.offset)
                # Keep the refresh bounded even when catching up with a large log.
                for _ in range(500):
                    before=f.tell();line=f.readline()
                    if not line.endswith(b'\n'):
                        f.seek(before);break
                    try:
                        display=event_line(json.loads(line))
                        if display:self.lines.append(display)
                    except (ValueError,TypeError,AttributeError):pass
                self.offset=f.tell()
        except OSError:pass


def snapshot(run,config,tail):
    owner=read_json(run/'ankle_status.json') or {}
    age=time.monotonic()-owner.get('updated_monotonic',0)
    live=0<=age<=.25
    vla=owner.get('vla') or {};job=vla.get('active_job') or {}
    active=owner.get('active_job') or {}
    movement=owner.get('locomotion') or {};move_job=movement.get('active_job') or {}
    lines=[f'G1 agent / VLA  {time.strftime("%H:%M:%S")}  只读，Ctrl+C 仅关闭此窗口',
           f'运行目录: {run}',f'会话配置: {config}',
           f'owner: {owner.get("phase","未启动")} | 反馈: {"实时" if live else "过期/未启动"} | 租约: {"不限时" if owner.get("lease_unlimited") else str(owner.get("lease_remaining_seconds"))+" 秒"}',
           f'VLA: {vla.get("phase","未启用")} | 已初始化: {vla.get("initialized",False)}',
           f'当前任务: {job.get("task",job.get("action","无"))} | 步数预算: {job.get("steps","—")}',
           f'踝温动作: {active.get("side","无")} | 阶段: {owner.get("stage")}',
           f'故障: {owner.get("fault") or "无"}',
           f'最近 VLA 结果: {short(vla.get("last_result"))}',
           f'移动: {movement.get("phase","未启用")} | 方向: {move_job.get("command",{}).get("direction","无")} | 上身保持: {bool(movement.get("upper_body_hold"))}',
           f'最近移动结果: {short(movement.get("last_result"))}',
           '', '最近工具调用（evidence 为 agent 给出的交接依据；完整计划看 agent 交互窗口）：']
    tail.poll();lines.extend(tail.lines or ['等待 agent 工具调用…'])
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--once',action='store_true')
    a=p.parse_args();config=read_json(a.config)
    if not config or 'root' not in config:p.error('Provide a valid RAX session config')
    tail=AuditTail(Path(config['root'])/'mcp.jsonl')
    try:
        while True:
            if sys.stdout.isatty() and not a.once:print('\033[2J\033[H',end='')
            print(snapshot(a.run,a.config,tail),flush=True)
            if a.once:break
            time.sleep(1)
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
