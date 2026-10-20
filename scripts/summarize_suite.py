"""Summarize heterogeneous diagnostics without inventing a benchmark success rate."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
rows=[]
for directory in sorted((root/'outputs').iterdir()):
    if not directory.is_dir() or not directory.name.endswith('20261018') and '20261018_' not in directory.name:
        continue
    acceptance=directory/'acceptance.json'
    if not acceptance.exists():continue
    a=json.loads(acceptance.read_text()); config=json.loads((directory/'run_config.json').read_text())
    summary=a['control_summary'] or {}; rounds=a['rounds']; private=[r['private_evidence'] for r in rounds if r['private_evidence']]
    opening=[r for r in rounds if r.get('command') and any(t.startswith('open_') for t in r['command']['motion_tags'])]
    joints=[p['container_joint_positions'] for p in private if 'container_joint_positions' in p]
    initial_path=directory/'initial/scorer_private.json'
    initial=json.loads(initial_path.read_text()) if initial_path.exists() else {}
    joint_delta=None
    if joints and 'container_joint_positions' in initial:
        joint_delta=np.max(np.abs(np.asarray(joints)-np.asarray(initial['container_joint_positions'])),axis=0).tolist()
    videos=[]
    for v in a['videos']:
        streams=v['probe']['streams'] if v['probe'] else []
        stream=next((s for s in streams if s.get('codec_type')=='video'),{})
        videos.append({'path':v['path'],'fully_decoded':v['fully_decoded'],
                       **{k:stream.get(k) for k in ['codec_name','profile','codec_tag_string','pix_fmt','nb_read_frames']}})
    rows.append({'run':directory.name,'checkpoint_step':config.get('checkpoint_step',10000),
                 'initial_yaw_offset_deg':config.get('initial_yaw_offset_deg',0),
                 'initial_offset_x':config.get('initial_offset_x',0),'initial_offset_y':config.get('initial_offset_y',0),
                 'fixed_instruction_diagnostic':config.get('diagnostic_open_drawer',False),
                 'opening_commit_chunks':config.get('opening_commit_chunks',1),
                 'reason':summary.get('reason','incomplete'),'ack_steps':a['acknowledged_steps'],
                 'opening_ack_steps':sum(r['ack_steps'] for r in opening),
                 'opening_motion_tags':sorted(set(t for r in opening for t in r['command']['motion_tags'])),
                 'find_and_approach_success':a['find_and_approach_success'],
                 'any_container_open_observed':any(p.get('container_open') is True for p in private),
                 'max_container_joint_delta':joint_delta,
                 'all_submitted_actions_exact_prediction_prefix':all(r['raw_prefix_exact'] is not False for r in rounds),
                 'unacknowledged_submissions':sum(r['unacknowledged_submissions'] for r in rounds),
                 'terminal_audits':a['terminal_audits'],
                 'diagnostic_terminal_evidence':a.get('diagnostic_terminal_evidence'),
                 'videos':videos,'acceptance_path':str(acceptance)})
report={'scope':'real local OmniGibson with remote trained VLA; heterogeneous iterative diagnostics, not independent benchmark trials',
        'strict_arrival_criterion':{'max_horizontal_distance_m':1.0,'minimum_exact_target_pixels':16,'planner_reached_required_for_autonomous_success':True},
        'notes':['Original turning_on_radio success is not the requested find-and-approach objective.',
                 'Several runs were interrupted by planner communication failures.',
                 'Local opening diagnostics modify only the initial robot pose and do not establish full-room search success.',
                 'Fixed instruction diagnostic excludes autonomous planning; its visibility/distance score is not autonomous task success.',
                 'Planner prompts evolved during debugging; per-call prompts and hashes are retained.'],
        'runs':rows}
metadata_path=root/'outputs/suite_annotations_20261018.json'
metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
for row in rows:
    row['interpretation_note']=metadata.get('run_annotations',{}).get(row['run'],'See run config and termination reason.')
videos=[v for row in rows for v in row['videos']]
report['final_checks']={
    'unit_tests_passed':metadata.get('unit_tests_passed'),
    'all_action_prefixes_match':all(row['all_submitted_actions_exact_prediction_prefix'] for row in rows),
    'unacknowledged_submissions':sum(row['unacknowledged_submissions'] for row in rows),
    'video_count':len(videos), 'all_videos_fully_decoded':all(v['fully_decoded'] for v in videos),
    'all_videos_h264_high_avc1_yuv420p':all(v['codec_name']=='h264' and v['profile']=='High' and v['codec_tag_string']=='avc1' and v['pix_fmt']=='yuv420p' for v in videos)}
report['temporary_checkpoint_024000_server']=metadata.get('temporary_checkpoint_024000_server')
path=root/'outputs/real_simulation_report_20261018.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'report':str(path),'runs':len(rows),'ack_steps':sum(r['ack_steps'] for r in rows),'autonomous_successes':sum(r['find_and_approach_success'] for r in rows)},ensure_ascii=False))
