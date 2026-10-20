import json,sys,html
from pathlib import Path
from audit_rollout import audit
root=Path(sys.argv[1]).resolve();run=root/'completion_v2'
a=audit(run)
summary=a['control_summary'] or {}
restored=a['handoff_restore'] or {}
rounds=a['rounds'];videos=a['videos']
complete=bool(a['simulator_result'] and a['simulator_result'].get('success') is not None and not a['error'] and not a['launcher_failure'])
integrity=bool(rounds and all(r['unacknowledged_submissions']==0 and r['executed_wrapper_prefix_exact'] is True for r in rounds) and a['acknowledged_steps']==summary.get('steps'))
video_ok=bool(len(videos)>=2 and all(v['fully_decoded'] and v['probe'] and all(s.get('codec_name')=='h264' and s.get('pix_fmt')=='yuv420p' for s in v['probe']['streams'] if s.get('codec_type')=='video') for v in videos))
valid=complete and integrity and video_ok and restored.get('full_state_checked') and restored.get('physics_steps_added')==0
success=bool(a['original_task_success'] and valid)
final=dict(status='success' if success else ('task_not_successful' if valid else 'infrastructure_or_incomplete'),original_task_success=success,valid_evaluation=bool(valid),steps=a['acknowledged_steps'],reason=summary.get('reason'),execution_integrity_passed=integrity,video_integrity_passed=video_ok,handoff_restore=restored,raw_actions='30x23 model output; saved separately',executed_actions='20x23 published champion wrapper output; corrections and resampling enabled',prior_attempt='completion_v1: zero physical steps, nonfinite stage-score JSON serialization failure',comparison='Same trial_01 search handoff; original official radio VLA failed at 10000 steps. This is one final-stage matched-state test, not a new end-to-end search benchmark.',videos=[v['path'] for v in videos])
(root/'final_acceptance.json').write_text(json.dumps(final,ensure_ascii=False,indent=2))
label='官方开机成功' if success else ('预算内未成功开机' if valid else '基础设施或证据不完整，不能判能力失败')
links=''.join(f'<li><a href="{html.escape(str(Path(v["path"]).relative_to(root)))}">{html.escape(Path(v["path"]).name)}</a></li>' for v in videos)
(root/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>冠军模型收音机测试</title><h1>2025 冠军模型：{label}</h1><p>实际执行 {a['acknowledged_steps']} 步。checkpoint_2，task_id=0，保留发布版动作修正与重采样。</p><p>从 trial_01/search_repair_v3 的完整交接状态开始，仅比较最终开机阶段。旧官方收音机模型同状态执行 10000 步未成功。</p><p>有效评测：{bool(valid)}；动作 ACK 完整：{integrity}；视频完整解码：{video_ok}。</p><ul>{links}<li><a href="final_acceptance.json">最终验收</a></li><li><a href="completion_v2/acceptance.json">逐步验收与视频信息</a></li><li><a href="protocol.json">测试协议</a></li><li><a href="interface_contract.json">输入与动作处理契约</a></li></ul>''')
print(json.dumps(final,ensure_ascii=False))
