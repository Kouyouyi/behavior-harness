"""Evidence-bounded per-task report; incomplete runs never enter capability rates."""
import collections
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
def read(p):
    return json.loads(p.read_text()) if p.exists() else None

def build(out):
    out=Path(out)
    summary=read(out/'summary.json') or {'tasks':[]}
    indexed={r['task']:r for r in summary['tasks']}
    rows=[]
    for spec in read(ROOT/'configs/ood99_manifest.json'):
        name=spec['task_name'];r=indexed.get(name)
        if not r:
            rows.append({'task':name,'status':'pending','reason':'尚未完成测试','videos':[]});continue
        run=out/r['run'];a=read(run/'acceptance.json') or {}
        status=r['status'];reason=str(r.get('detail',''))
        if status=='success':reason='目标可见、水平距离不超过1米，且planner声明到达'
        elif status=='not_successful_within_budget':
            stop=(a.get('control_summary') or {}).get('reason')
            reason={'arrival_verification_failed':'planner声明到达，但独立可见性/距离验收未通过',
                    'observation_request_limit':'重复请求观察，未继续完成搜索接近',
                    'round_limit':'耗尽指令轮次，未通过到达验收',
                    'max_steps':'耗尽物理步数，未通过到达验收'}.get(stop,'在规定预算内未通过到达验收：'+str(stop))
        elif status=='infrastructure_failed':
            planners=sorted((run/'control').glob('round_*/planner'))
            if planners:
                p=planners[-1];fail=read(p/'transport_failures.json')
                if fail:
                    reason='planner中断：'+'; '.join(x['error'] for x in fail)
                    if all('timeout' in x['error'].lower() or 'exited' in x['error'].lower() for x in fail):
                        status='awaiting_model_retry'
                        reason='历史模型响应中断，等待重测；不计任务能力失败，原始日志保留'
                elif (p/'answer.json').exists() and not (p/'result.json').exists():reason='planner输出未通过指令契约验证，未下发该指令'
        elif status=='initialization_infrastructure_failed':reason='场景初始化/可见性测量未完成：'+reason
        elif status=='scene_invalid':reason='候选OOD场景未通过验收：'+reason
        videos=[{'path':v['path'],'fully_decoded':v['fully_decoded']} for v in a.get('videos',[])]
        terminal=[x for x in a.get('terminal_audits',[]) if x]
        rows.append({'task':name,'status':status,'reason':reason,'steps':r.get('steps',0),
            'run':r['run'],'acceptance':str((run/'acceptance.json').resolve()) if (run/'acceptance.json').exists() else None,
            'terminal_distance_m':terminal[-1].get('horizontal_distance_m') if terminal else None,
            'videos':videos,'video_note':None if videos else '没有可交付视频；查看初始化或运行错误，不能据此声称完成仿真',
            'prior_attempts_preserved':bool(r.get('previous_attempts'))})
    counts=dict(collections.Counter(r['status'] for r in rows))
    success=counts.get('success',0);failed=counts.get('not_successful_within_budget',0)
    report={'counts':counts,'total':99,'completed_capability_trials':success+failed,
            'capability_success_rate':success/(success+failed) if success+failed else None,
            'all_99_capability_evaluated':success+failed==99,'active':summary.get('active'),
            'note':'基础设施/场景失败与未测试任务不计入能力成功率。每任务一个实例/种子；仅验收找到并接近。', 'tasks':rows}
    (out/'task_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    body=[]
    for r in rows:
        links=[]
        if r.get('acceptance'):links.append('<a href="'+html.escape(str(Path(r['acceptance']).relative_to(out.resolve())),quote=True)+'">验收</a>')
        for i,v in enumerate(r['videos']):
            links.append('<a href="'+html.escape(str(Path(v['path']).relative_to(out.resolve())),quote=True)+f'">视频{i+1}</a>')
        body.append('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in ['task','status','steps','reason'])+'<td>'+' · '.join(links)+'</td></tr>')
    (out/'task_results.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>逐任务验收</title><style>body{font:15px system-ui;margin:24px}td{padding:10px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>99任务逐项验收（持续更新）</h1><p>'+html.escape(json.dumps(counts,ensure_ascii=False))+'</p><p>'+report['note']+'</p><table><tr><th>任务</th><th>状态</th><th>步数</th><th>原因</th><th>证据</th></tr>'+''.join(body)+'</table>')
    return report

if __name__=='__main__':
    r=build(Path(sys.argv[1]).resolve());print(json.dumps({k:v for k,v in r.items() if k!='tasks'},ensure_ascii=False))
