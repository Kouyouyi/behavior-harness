import html
from pathlib import Path

def render_report(out,data):
    rows=[]
    by_id={r['trial']:r for r in data['trials']}
    for i in range(1,11):
        name=f'trial_{i:02d}';r=by_id.get(name,{})
        active=data.get('active') or {}
        status=r.get('status',active.get('phase') if active.get('trial')==name else 'pending')
        links=[]
        for stage in ('search','completion'):
            run=r.get(stage+'_run') or str(out/name/(stage+'_v1'))
            if Path(run).exists():
                relative=Path(run).relative_to(out)
                for filename,label in [('acceptance.json','验收'),('sim/videos/robot_cameras.mp4','机器人视频'),('sim/videos/third_person.mp4','第三人称视频')]:
                    if (Path(run)/filename).exists():links.append(f'<a href="{relative}/{filename}">{stage} {label}</a>')
        rows.append(f'<tr><td>{name}</td><td>{"陌生位置" if i<=5 else "柜内遮挡"}</td><td>{html.escape(str(status))}</td><td>{r.get("search_success","—")}</td><td>{r.get("original_task_success","—")}</td><td>{html.escape(str(r.get("reason","")))}</td><td>{" · ".join(links)}</td></tr>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>收音机10次两阶段评测</title><style>body{font:16px system-ui;margin:30px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>收音机10次独立OOD评测</h1><p>5次陌生位置、5次柜内遮挡；搜索接近与官方开启收音机分别验收。模型等待不算能力失败。</p><p>已完成记录 '+str(data['completed_records'])+'/10；搜索成功 '+str(data['search_successes'])+'；原任务成功 '+str(data['original_task_successes'])+'。尚未完成时不报告最终成功率。</p><table><tr><th>试验</th><th>场景</th><th>状态</th><th>找到并接近</th><th>开启收音机</th><th>原因</th><th>证据</th></tr>'+''.join(rows)+'</table>')
if __name__=='__main__':
    import json,sys
    out=Path(sys.argv[1]).resolve();render_report(out,json.loads((out/'summary.json').read_text()))
