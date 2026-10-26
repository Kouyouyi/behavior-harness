#!/usr/bin/env python3
"""Portable public entrypoint. Never copies credentials into a study."""
import argparse,json,os,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'seekact'))

def main():
    p=argparse.ArgumentParser(description='SeekAct: AP / champion / Qwen tool scheduling')
    commands=p.add_subparsers(dest='command',required=True)
    commands.add_parser('doctor',help='Read-only runtime and configuration checks')
    prepare=commands.add_parser('prepare',help='Create a new three-task protocol, no physics')
    prepare.add_argument('--output',required=True,type=Path)
    prepare.add_argument('--qwen-base-url',required=True)
    prepare.add_argument('--qwen-key-file',required=True,type=Path)
    prepare.add_argument('--qwen-model')
    prepare.add_argument('--ap-url',default='http://127.0.0.1:18770')
    prepare.add_argument('--champion-url',default='http://127.0.0.1:18778')
    prepare.add_argument('--ap-checkpoint',required=True,help='Exact remote path reported by AP /health')
    run=commands.add_parser('run',help='Preflight/calibrate then run all three fresh episodes')
    run.add_argument('--output',required=True,type=Path)
    commands.add_parser('test',help='Run BEHAVIOR focused regression tests; no physical simulation')
    a=p.parse_args()
    if a.command=='doctor':
        from scripts import run_behavior_experiment as base
        facts=dict(repo=str(REPO),agent_python=str(base.AGENT_PYTHON),sim_python=str(base.SIM_PYTHON),
            simulator_root=str(base.SIM_ROOT),assets_root=str(base.ASSET_ROOT),gpu_lock=str(base.GPU_LOCK),
            sudo_helper=str(base.SUDO_HELPER)if base.SUDO_HELPER else None)
        for label,python,code in [('agent',base.AGENT_PYTHON,'import PIL; import services.behavior_ap.vla'),
              ('simulator',base.SIM_PYTHON,'import behavior_ap_sim; import av,cv2,numpy,scipy; import omnigibson')]:
            # Read-only imports; no simulation environment is constructed.
            result=subprocess.run([str(python),'-c',code],env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(REPO/'seekact'),str(base.SIM_ROOT)))),capture_output=True,text=True)
            facts[label+'_imports_ok']=result.returncode==0
            facts[label+'_error']=result.stderr[-1500:]if result.returncode else None
        print(json.dumps(facts,ensure_ascii=False,indent=2));return 0 if facts['agent_imports_ok']and facts['simulator_imports_ok']else 1
    if a.command=='prepare':
        key=a.qwen_key_file.expanduser().resolve()
        if not key.is_file():p.error('Qwen key file does not exist')
        if key.is_relative_to(REPO):p.error('Store the credential outside the Git repository')
        if key.stat().st_mode&0o077:p.error('Qwen key file must be private (chmod600)')
        root=a.output.expanduser().resolve();root.mkdir(parents=True,exist_ok=False)
        protocol=json.loads((REPO/'configs/seekact/three_tasks.json').read_text())
        protocol['vla_tools'].update(ap_url=a.ap_url,champion_url=a.champion_url,ap_checkpoint=a.ap_checkpoint)
        protocol['vla_tools']['qwen_monitor'].update(base_url=a.qwen_base_url.rstrip('/'),api_key_file=str(key),model=a.qwen_model)
        for name in ('protocol.json','protocol_current.json'):(root/name).write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(output=str(root),planned_tasks=[x['task_name']for x in protocol['tasks']],physical_steps=0),indent=2));return 0
    if a.command=='run':
        root=a.output.expanduser().resolve()
        if not(root/'protocol.json').is_file():p.error('Run prepare first')
        from services.controller.config import Configuration
        os.environ['RAX_ARTIFACT_ROOT']=str(root)
        os.execv(sys.executable,[sys.executable,'-u',str(REPO/'seekact/scripts/run_qwen_interaction_study.py'),'--output',str(root)])
    if a.command=='test':
        names=['test_behavior_qwen_monitor','test_behavior_interaction','test_behavior_closed_loop',
            'test_behavior_launcher','test_behavior_experiment','test_behavior_worker',
            'test_behavior_vla','test_behavior_vla_only','test_behavior_video_overlay','test_behavior_video_recording','test_seekact_release']
        return subprocess.call([sys.executable,'-m','unittest',*['local_tests.unit.'+n for n in names]],cwd=REPO/'seekact')

if __name__=='__main__':raise SystemExit(main())
