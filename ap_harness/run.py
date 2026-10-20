import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .paths import ROOT, PYTHON
from .control import run_loop, save_json
from .policy import PinnedRemotePolicy

def user_planner(images, goal, history, output):
    request = Path(output).parent / 'planner_request.json'
    save_json(request, {'images': images, 'goal': goal, 'history': history,
                       'output': str(output), 'timeout': 480})
    command = [PYTHON, '-m', 'ap_harness.planner', '--request', str(request)]
    if os.geteuid() == 0:
        import pwd
        owner = pwd.getpwnam('cody')
        os.chown(Path(output).parent, owner.pw_uid, owner.pw_gid)
        command = ['runuser', '-u', 'cody', '--', 'env',
                   'PATH=/home/cody/.nvm/versions/node/v25.2.1/bin:/usr/local/bin:/usr/bin:/bin',
                   *command]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads((Path(output) / 'result.json').read_text())

def diagnostic_planner(images, goal, history, output):
    from planner_contract import render
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    command={'status':'execute','motion_tags':['open_drawer'],
             'target_description':'电视下方矮柜抽屉','goal_kind':'inspect_container_interior',
             'visual_evidence':'固定指令故障隔离诊断；不是自主视觉判断。','target_visible':False}
    result={'command':command,'instruction':render(command),'seconds':0,
            'planner_source':'fixed_instruction_diagnostic','autonomous_search':False}
    save_json(output/'result.json',result)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--goal', default='寻找收音机，必要时搜索可见通路或打开可疑容器，观察并接近收音机。')
    parser.add_argument('--max-steps', type=int, default=800)
    parser.add_argument('--max-rounds', type=int, default=200)
    parser.add_argument('--planner-max-chunks',type=int,choices=[1,2,3,4],default=4)
    parser.add_argument('--checkpoint-step',type=int,choices=[10000,24000],default=10000)
    parser.add_argument('--opening-commit-chunks',type=int,choices=[1,4,8,12],default=1)
    parser.add_argument('--ood', choices=['none', 'radio_on_tv_cabinet', 'radio_on_tv_cabinet_facing_room', 'radio_inside_closed_container'], default='none')
    parser.add_argument('--container-name')
    parser.add_argument('--scenario-spec',type=Path)
    parser.add_argument('--preflight-dir',type=Path)
    parser.add_argument('--resume-handoff',type=Path)
    parser.add_argument('--champion-executor',action='store_true')
    parser.add_argument('--export-handoff',action='store_true')
    parser.add_argument('--initial-yaw-offset-deg', type=float, default=0.0)
    parser.add_argument('--initial-offset-x',type=float,default=0.0)
    parser.add_argument('--initial-offset-y',type=float,default=0.0)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--task-name', default='turning_on_radio')
    parser.add_argument('--instance-id', type=int, default=301)
    parser.add_argument('--target-key', default='radio_receiver.n.01_1')
    parser.add_argument('--initialize-only', action='store_true')
    parser.add_argument('--diagnostic-open-drawer',action='store_true')
    parser.add_argument('--initial-visibility-audit', action='store_true')
    args = parser.parse_args()
    scenario = None
    if args.scenario_spec:
        args.scenario_spec = args.scenario_spec.resolve()
        scenario=json.loads(args.scenario_spec.read_text())
        args.task_name=scenario['task_name'];args.target_key=scenario['target_key']
        args.instance_id=scenario['instance_id'];args.goal=scenario['goal']
        if not args.initialize_only and not args.preflight_dir:
            parser.error('Scenario rollout requires a successful isolated visibility preflight')
    if args.max_steps < 1 or args.max_rounds < 1:
        parser.error('Positive step and round limits required')
    if args.initial_visibility_audit and not args.initialize_only:
        parser.error('Visibility audit is isolated from physics stepping because of an Isaac synthetic-data crash')
    if args.ood == 'radio_inside_closed_container' and not args.container_name:
        parser.error('Closed-container fixture requires an explicit --container-name from the private inventory')
    if args.ood != 'none' and (args.task_name != 'turning_on_radio' or args.target_key != 'radio_receiver.n.01_1'):
        parser.error('Built-in OOD fixtures are radio-specific; other tasks require goal-preserving scene setup')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'run_config.json', {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    if scenario: save_json(args.output/'scenario_spec.json',scenario)
    import hashlib
    save_json(args.output / 'source_manifest.json', {
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('ap_harness', 'runtime/behavior_harness', 'pi05_harness_20261017/checkpoint_metadata')
        for p in (ROOT/folder).glob('*') if p.is_file()})
    from .simulator import Simulator
    sim = Simulator(args.output / 'sim', max_steps=args.max_steps, ood='suite_ood' if scenario else args.ood,
                    container_name=args.container_name,task_name=args.task_name,
                    instance_id=args.instance_id,target_key=args.target_key,
                    initial_visibility_audit=args.initial_visibility_audit, initial_yaw_offset_deg=args.initial_yaw_offset_deg, initial_offset_x=args.initial_offset_x, initial_offset_y=args.initial_offset_y,
                    scenario_spec=str(args.scenario_spec) if scenario else None,seed_override=scenario['seed'] if scenario else args.seed)
    reason = 'infrastructure_error'
    try:
        sim.start()
        if scenario and not args.initialize_only:
            from .ood99 import fingerprint, match_fingerprint
            preflight=args.preflight_dir
            if (preflight/'error.json').exists(): raise RuntimeError('Failed preflight cannot authorize rollout')
            if json.loads((preflight/'scenario_spec.json').read_text()) != scenario:
                raise RuntimeError('Scenario specification changed since preflight')
            evidence=json.loads((preflight/'sim/initial_occlusion_private.json').read_text())
            if any(r['target_visible'] for r in evidence['equivalent_target_evidence']):
                raise RuntimeError('Initial target visibility invalidates OOD condition')
            match_fingerprint(fingerprint(sim),json.loads((preflight/'sim/initial_fingerprint_private.json').read_text()))
            save_json(args.output/'preflight_match.json',{'passed':True,'preflight':str(preflight),'tolerance_m':.02})
        if args.preflight_dir and not scenario:
            from .ood99 import fingerprint, match_fingerprint
            pre=args.preflight_dir
            prior=json.loads((pre/'run_config.json').read_text())
            for key in ('ood','container_name','seed','initial_yaw_offset_deg','initial_offset_x','initial_offset_y','task_name','instance_id','target_key'):
                if prior[key]!=getattr(args,key):raise ValueError('Preflight config mismatch: '+key)
            evidence=json.loads((pre/'sim/initial_occlusion_private.json').read_text())
            if evidence['target_visible']:raise ValueError('Initially visible target invalidates OOD')
            match_fingerprint(fingerprint(sim),json.loads((pre/'sim/initial_fingerprint_private.json').read_text()))
            save_json(args.output/'preflight_match.json',{'passed':True,'preflight':str(pre)})
        if args.resume_handoff:
            sim.restore_handoff(args.resume_handoff)
        sim.capture(args.output / 'initial')
        if args.export_handoff:
            sim.save_handoff(args.output/'task_handoff')
        if args.initialize_only:
            reason = 'initialize_only'
        elif args.champion_executor:
            from .champion_execution import run_task
            reason=run_task(sim,args.task_name,args.output/'control')
        elif args.resume_handoff:
            from .task_execution import run_task
            reason=run_task(sim,args.task_name,args.output/'control')
        else:
            reason = run_loop(sim, diagnostic_planner if args.diagnostic_open_drawer else user_planner, PinnedRemotePolicy(timeout=120, retry_log=args.output/'model_wait.jsonl', url='http://127.0.0.1:'+('18770' if args.checkpoint_step==24000 else '18769'), checkpoint='/cs/kyy/behavior_ap_experiment/runs/pi05_goal_horizon_v2_b8/checkpoint_'+f'{args.checkpoint_step:06d}'), args.goal,
                              args.output / 'control', max_rounds=args.max_rounds, planner_max_chunks=args.planner_max_chunks, opening_commit_chunks=args.opening_commit_chunks)
        if args.diagnostic_open_drawer and not args.initialize_only:
            evidence = sim.verify_approach(args.output/'diagnostic_terminal')
            save_json(args.output/'diagnostic_evaluation.json', {
                'fixed_instruction_diagnostic':True, 'autonomous_search_success':False,
                'geometry_and_visibility_passed':evidence['find_and_approach_success'],
                'planner_reached_claim':False, 'terminal_evidence':evidence})
        if scenario and not args.initialize_only and not args.resume_handoff and not getattr(sim,'_terminal_audit_started',False):
            # A timeout can still leave the target visible/near. Keep this
            # independent measurement separate from an autonomous reached claim.
            evidence=sim.verify_approach(args.output/'terminal_geometry_only')
            save_json(args.output/'terminal_geometry_only/interpretation.json',{
                'planner_reached_claim':False,'autonomous_find_and_approach_success':False,
                'visible_and_near_without_claim':evidence['find_and_approach_success']})
    except BaseException as exc:
        save_json(args.output / 'error.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        sim.close(reason)
        sim.shutdown()

if __name__ == '__main__': main()
