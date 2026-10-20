"""Audit persisted physical execution and decode delivered videos."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np


def audit(root):
    root=Path(root)
    read=lambda p:json.loads(p.read_text()) if p.exists() else None
    config=read(root/'run_config.json') or {}
    summary=read(root/'control/summary.json')
    result=read(root/'sim/result.json')
    rounds=[]
    for directory in sorted((root/'control').glob('round_*')):
        planner=read(directory/'planner/result.json')
        events=[json.loads(line) for line in (directory/'execution.jsonl').read_text().splitlines()] if (directory/'execution.jsonl').exists() else []
        submissions=[e for e in events if e['event']=='submitted']
        acks=[e for e in events if e['event']=='ack']
        exact=None
        if submissions:
            predictions=np.load(directory/'predicted_actions.npy')
            exact=all(np.array_equal(np.asarray(e['action'],dtype=np.float32),predictions[e['offset']]) for e in submissions)
        private=read(directory/'after/scorer_private.json') or read(directory/'scorer_private.json')
        distance=None
        if private:
            distance=float(np.linalg.norm(np.array(private['target_position'])[:2]-np.array(private['base_position'])[:2]))
        rounds.append({'round':directory.name,'instruction':planner['instruction'] if planner else None,
                       'command':planner['command'] if planner else None,
                       'ack_steps':len(acks),'unacknowledged_submissions':len(submissions)-len(acks),
                       'raw_prefix_exact':exact if not config.get('champion_executor') else None,
                       'executed_wrapper_prefix_exact':exact if config.get('champion_executor') else None,
                       'horizontal_distance_m':distance,'private_evidence':private})
    videos=[]
    for path in sorted((root/'sim').rglob('*.mp4')):
        probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(path)],capture_output=True,text=True)
        decode=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],capture_output=True,text=True)
        videos.append({'path':str(path.resolve()),'probe':json.loads(probe.stdout) if probe.returncode==0 else None,
                       'fully_decoded':decode.returncode==0,'errors':probe.stderr+decode.stderr})
    report={'run':str(root.resolve()),'control_summary':summary,'simulator_result':result,
            'run_config':config,'fixed_instruction_diagnostic':config.get('diagnostic_open_drawer',False),
            'diagnostic_terminal_evidence':read(root/'diagnostic_terminal/result.json'),
            'terminal_geometry_without_reached_claim':read(root/'terminal_geometry_only/result.json'),
            'launcher_failure':read(root/'launcher_failure.json'),'error':read(root/'error.json'),
            'acknowledged_steps':sum(r['ack_steps'] for r in rounds),'rounds':rounds,'videos':videos,
            'terminal_audits':[read(p) for p in sorted((root/'control').glob('round_*/terminal_audit/result.json'))],
            'find_and_approach_success':bool(summary and summary.get('find_and_approach_success')),
            'original_task_success':bool(summary and summary.get('official_task_success')),
            'objective':summary.get('objective') if summary else None,
            'handoff_restore':read(root/'sim/handoff_restore.json')}
    (root/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'run':str(root),'steps':report['acknowledged_steps'],'success':report['find_and_approach_success'],
                      'reason':summary['reason'] if summary else 'incomplete','videos':len(videos)},ensure_ascii=False))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directories',nargs='+',type=Path)
    for directory in parser.parse_args().directories: audit(directory)
