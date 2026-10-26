"""Interactive Codex using the existing login and one RAX stdio MCP frontend.

No global Codex config changes, credential copies or native robot startup.
This G1 entry uses direct MCP calls; shell, browser and parallel agents are disabled.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from services.controller.config import Configuration
from services.g1_sonic import vla
from services.workspace import ROBOT_FACTS

ROOT=Path(__file__).resolve().parents[2]
HARNESS_CODEX=Path('/home/user/code/HumanoidHarnessDeploy/.venv/lib/python3.12/site-packages/codex_cli_bin/bin/codex')


def binary(explicit=None):
    candidates=[explicit,os.environ.get('CODEX_BIN'),shutil.which('codex'),HARNESS_CODEX]
    for value in candidates:
        if value and Path(value).is_file() and os.access(value,os.X_OK):return str(Path(value).resolve())
    raise RuntimeError('Codex binary unavailable; provide --codex-bin')


def inherited_mcp():
    try:import tomllib
    except ImportError:import tomli as tomllib
    path=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'config.toml'
    if not path.exists():return []
    with path.open('rb') as file:return list(tomllib.load(file).get('mcp_servers',{}))


def settings(config,config_path,python,other_servers):
    values=[('features.'+name,False) for name in (
        'shell_tool','unified_exec','shell_snapshot','multi_agent','multi_agent_v2',
        'apps','plugins','browser_use','computer_use')]
    # Current model tool routing needs the local Code Mode host, even when
    # shell tools are disabled. It can invoke only the configured tool set.
    values+=[('features.code_mode_host',True),('web_search','disabled')]
    values += [(f'mcp_servers.{json.dumps(name)}.enabled',False)
               for name in other_servers if name!='rax_robot']
    values += [
        ('mcp_servers.rax_robot.command',str(python)),
        ('mcp_servers.rax_robot.args',['-m','services.controller.frontend','--config',str(config_path)]),
        ('mcp_servers.rax_robot.cwd',str(ROOT)),
        ('mcp_servers.rax_robot.env.PYTHONPATH',str(ROOT)),
        ('mcp_servers.rax_robot.env.PYTHONDONTWRITEBYTECODE','1'),
        ('mcp_servers.rax_robot.enabled',True),
        ('mcp_servers.rax_robot.required',True),
        ('mcp_servers.rax_robot.startup_timeout_sec',30),
        ('mcp_servers.rax_robot.tool_timeout_sec',30),
        ('mcp_servers.rax_robot.default_tools_approval_mode','approve'),
        ('mcp_servers.rax_robot.enabled_tools',['status',*ROBOT_FACTS['g1_sonic']['tools']]),
    ]
    return [f'{key}={json.dumps(value,ensure_ascii=False)}' for key,value in values]


def prepare(config,config_path,python,codex):
    if config.robot!='g1_sonic' or config.agent_provider!='codex':
        raise ValueError('This entry requires --robot g1_sonic --agent-provider codex')
    if config.subagents:raise ValueError('Use --subagents 0 for this G1 owner')
    workspace=config.root/'agent-workspace'
    workspace.mkdir(parents=True,exist_ok=True)
    (workspace/'runtime/frames').mkdir(parents=True,exist_ok=True)
    skill=ROOT/'agent_skills/control/g1_sonic/robot-interface/SKILL.md'
    # Include the operational skill directly: no shell/file tool is needed to read it.
    instructions='''# G1 operator session via Codex

You control this robot only through the rax_robot MCP tools. Use tool results and
images; never infer that submitted or completed reference commands prove grasp
success. Follow the operator's current task and existing authorization. Start by
reading status and observing, then wait for the operator to name the object before
any prepare/run command. Do not start, stop or reset the native controller.

Before each meaningful action, give a short Chinese explanation of the observed
scene, chosen tool and purpose. After it, report measured outcome and what needs
visual confirmation. These public action summaries should be visible in the terminal.
When clarification is needed, ask directly in this chat. Never invent site evidence.

This session uses direct MCP calls, not shell scripts or subagents. Do not call
request_operator or skill_library_merge. Pause a VLA segment when asked. On a tool
error, inspect status and the current view instead of silently repeating motion.
A thermal interruption discards the queue; it is not permission to open a hand
holding an object. Native O remains the physical stop; ending Codex does not stop SONIC.

## Deployment skill

'''+skill.read_text()+'\n## Operator task\n\n'+config.task_description+'\n'
    (workspace/'AGENTS.md').write_text(instructions)
    (workspace/'TASK.md').write_text(config.task_description+'\n')
    overrides=settings(config,config_path,python,inherited_mcp())
    command=[codex]
    for value in overrides:command+=['-c',value]
    command+=['--cd',str(workspace),'--sandbox','read-only','--ask-for-approval','never','--no-alt-screen']
    if config.agent_model:command+=['--model',config.agent_model]
    command+=['先用中文说明你已接入 G1 的 RAX 工具，读取状态并观察；在我指定物品前不要执行 prepare 或 run。']
    (config.root/'codex-launch.json').write_text(json.dumps(dict(command=command,workspace=str(workspace),
        model=config.agent_model or 'existing Codex default',robot_start_commands=0),indent=2)+'\n')
    return command


def pause_active(config):
    directory=config.backend_options.get('control_directory')
    movement=None
    try:
        from services.g1_sonic import locomotion
        if locomotion.status(directory)['locomotion'].get('active_job'):
            movement=locomotion.submit(directory,'stop')
    except (OSError,ValueError,RuntimeError,KeyError):
        pass
    try:
        state=vla.status(directory)
        if state['vla'].get('active_job'):
            response=vla.submit(directory,'pause')
            if movement:response['movement_stop']=movement
            return response
        return dict(status='movement_stop_requested',movement_stop=movement) if movement else dict(status='no_active_vla_job')
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        if movement:return dict(status='movement_stop_requested',movement_stop=movement)
        return dict(status='pause_not_confirmed',reason=str(exc))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--codex-bin')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();path=a.config.resolve();config=Configuration.load(path)
    config.root.mkdir(parents=True,exist_ok=True)
    with (config.root/'codex-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        codex=binary(a.codex_bin)
        command=prepare(config,path,sys.executable,codex)
        if a.prepare_only:
            print(config.root/'codex-launch.json');return
        subprocess.run([codex,'login','status'],check=True)
        print('Codex → RAX MCP 已准备。原生站立和 O 停止由现场窗口保留。',flush=True)
        try:
            completed=subprocess.run(command,cwd=config.root/'agent-workspace')
            returncode=completed.returncode
        except KeyboardInterrupt:
            returncode=130
        finally:
            outcome=pause_active(config)
            (config.root/'codex-exit.json').write_text(json.dumps(outcome,indent=2)+'\n')
            print('Codex 已退出；原生 SONIC 未停止。VLA 暂停请求状态：'+json.dumps(outcome,ensure_ascii=False),flush=True)
        raise SystemExit(returncode)


if __name__=='__main__':main()
