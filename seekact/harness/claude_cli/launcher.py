"""Run Claude Code on the host as the robot agent, separated from the robot infrastructure.

The launcher (trusted, operator-run) prepares a session under the artifact root:

    <session>/
      configuration.json     operator configuration (services.controller.config)
      model-relay.json       provider, models and the credential *path* (never the token)
      agent-workspace/       the agent's cwd and its only writable directory
      claude-home/           CLAUDE_CONFIG_DIR and HOME of the agent: settings.json, mcp.json
      state.json, mcp.jsonl  robot-service ledger and audit log (unreadable to the agent)
      agent-lifecycle.json   launches, usage waits and exit status

and per launch starts two host services the agent cannot read or modify:

- the Messages relay on 127.0.0.1:<port>, which holds the model credential;
- the robot service (services.controller.frontend --socket), the only robot owner.

Claude Code runs with ``--permission-mode dontAsk`` and an explicit allow list, so
the Read/Edit/Write tools reach only the workspace, and every Bash command runs in
Claude Code's bubblewrap sandbox: writes only in the workspace, internet only through the
sandbox proxy (which refuses the robot controller), the host sockets that would bypass the
robot service hidden, and the repository, credentials, the operator's RAX_PROTECTED_PATHS
(robot-controller checkouts, SDK environments) and the session's private files unreadable.
The robot is reachable only through the MCP tools.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from harness.claude_cli.messages_relay import ANY_MODEL, Credential, MessagesRelay, PROVIDERS
from services.controller.config import Configuration
from services.json_io import write_private_json

PROJECT = Path(__file__).resolve().parents[2]
HOME_NAME = 'claude-home'
WORKSPACE_NAME = 'agent-workspace'
MCP_SERVER = 'robot'
# Tools the robot agent never needs: scheduling, orchestration and worktrees. WebSearch is a
# server tool the Messages relay refuses; WebFetch runs locally and is allowed.
DENIED_TOOLS = ('WebSearch', 'Workflow', 'CronCreate', 'CronDelete', 'CronList', 'ScheduleWakeup',
                'RemoteTrigger', 'SendMessage', 'ListAgents', 'EnterWorktree', 'ExitWorktree', 'PushNotification')
SUBAGENT_TOOLS = ('Agent', 'Task')  # 'Task' is the tool's former name.
RELAY_TOKEN = 'host-relay'  # Placeholder the relay replaces; not a credential.
# Session files only the launcher and the robot service read.
PRIVATE_FILES = ('configuration.json', 'model-relay.json', 'state.json', 'mcp.jsonl', 'frontend.log',
                 'model-usage.jsonl', 'model-calls.json', 'agent-lifecycle.json', 'workspace-manifest.json')
# Credentials and the harness itself: the agent's tools never read them.
HOME = Path.home()
PROTECTED_PATHS = (PROJECT, HOME/'.config/rax', HOME/'.claude', HOME/'.claude.json', HOME/'.codex', HOME/'.ssh',
                   HOME/'.aws', HOME/'.config/gcloud', HOME/'.netrc', HOME/'.git-credentials')
# Paths on this host that grant direct robot access are site-specific, so the operator lists them:
# robot-controller checkouts, vendor SDK environments and other robot servers, whose code could reach
# the controller past the backend. Colon-separated absolute paths.
SITE_PROTECTED_ENVIRONMENT = 'RAX_PROTECTED_PATHS'
# On Linux the Bash sandbox can only allow all Unix sockets or none (seccomp cannot filter
# by path), and the agent's scripts need the robot service socket. These host sockets
# would bypass the robot service (docker controls robot containers) or leak the
# desktop session, so they are hidden from the sandbox.
UID = os.getuid()
HIDDEN_SOCKETS = (Path('/run/docker.sock'), Path('/var/run/docker.sock'), Path('/run/containerd'),
                  Path('/run/dbus'), Path(f'/run/user/{UID}'), Path('/tmp/.X11-unix'))
# Hosts the sandbox proxy refuses even though other internet hosts are allowed: the RB-Y1's
# controller serves unauthenticated gRPC (:50051) and its web UI (:5173), so a script with the
# vendor SDK could command the robot past the backend. The proxy matches the literal host
# (no CIDR ranges), so every address that reaches a controller must be listed: the factory
# address here, plus the host of the session's `address` backend option (denied_hosts).
DENIED_HOSTS = ('192.168.30.1',)
MAX_USAGE_WAITS = 10
# OpenRouter key for gemini_generate, read only by the host robot service (never the agent).
DEFAULT_GEMINI_KEY = Path.home()/'.config/rax/openrouter.key'
USAGE_STOP_SECONDS = 600  # Let Claude Code end its own turn first; then stop it.
RESUME_PROMPT = ("This session was interrupted and has now been resumed in the same workspace. The harness "
                 "may have been updated meanwhile: reread AGENTS.md and the skills it points to for your current "
                 "step (tools, subagents and procedures can have changed), then reread memory/PROGRESS.md, call "
                 "status, and continue from the live state. Do not repeat finished work.")
USAGE_PROMPT = ("The model usage limit paused this session until it reset. Call status, reread "
                "memory/PROGRESS.md, and continue from the live state.")


def claude_binary(path=None):
    binary = Path(path or shutil.which('claude') or '').resolve()
    if not binary.is_file():
        raise RuntimeError('Claude Code binary is required (install claude on the host)')
    return binary


def socket_path(root):
    """Short, per-session Unix socket path (Linux limits socket paths to 107 bytes)."""
    directory = Path(f'/tmp/rax-{os.getuid()}')
    directory.mkdir(mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    return directory/(hashlib.sha256(str(root).encode()).hexdigest()[:16] + '.sock')


def agent_python(config):
    """Interpreter directory put first on the agent's PATH: the session's agent_python, else
    RAX_AGENT_PYTHON (the operator scripts set it to the SDK-free agent environment), else this
    process's own interpreter (the robot service's, which has the robot SDK: avoid)."""
    return Path(config.agent_python or os.environ.get('RAX_AGENT_PYTHON') or sys.executable).parent


def robot_model_dir(config):
    """The robot's model folder (URDF + meshes) under the artifact root, if the operator installed it."""
    from services.storage_root import artifact_root
    from services.workspace.setup import ROBOT_FACTS
    relative = ROBOT_FACTS.get(config.robot, {}).get('model_dir')
    path = artifact_root()/relative if relative else None
    return path if path is not None and path.is_dir() else None


def site_protected_paths():
    """The operator's site paths that grant direct robot access (RAX_PROTECTED_PATHS)."""
    paths = [Path(part).expanduser() for part in os.environ.get(SITE_PROTECTED_ENVIRONMENT, '').split(':') if part]
    relative = [str(path) for path in paths if not path.is_absolute()]
    if relative:
        raise ValueError(f'{SITE_PROTECTED_ENVIRONMENT} takes absolute paths; got {relative}')
    return paths


def protected_paths(config):
    return [*PROTECTED_PATHS, *site_protected_paths(), *(config.root/name for name in PRIVATE_FILES)]


def denied_hosts(config):
    """DENIED_HOSTS plus the host of the session's controller address (backend option `address`)."""
    hosts = list(DENIED_HOSTS)
    address = config.backend_options.get('address')
    if address:
        host = (str(address).rpartition(':')[0] or str(address)).strip('[]')
        if host not in hosts:
            hosts.append(host)
    return hosts


def warn_unsafe_defaults(config):
    """Say so when the agent could reach robot software: no site paths listed, or no agent environment."""
    if not (config.agent_python or os.environ.get('RAX_AGENT_PYTHON')):
        print('warning: neither the session\'s agent_python nor RAX_AGENT_PYTHON is set, so the agent\'s scripts '
              'run in the robot service\'s interpreter, which has the robot SDK. Point RAX_AGENT_PYTHON at an '
              'environment without it.', file=sys.stderr)
    if not site_protected_paths():
        print(f'warning: {SITE_PROTECTED_ENVIRONMENT} is empty, so the agent\'s sandbox can read every robot '
              'controller checkout, vendor SDK environment and robot server on this host. List each of them '
              f'(colon-separated absolute paths) before running a session on the robot.', file=sys.stderr)


def settings(config, workspace, socket):
    """Claude Code settings: dontAsk permissions confined to the workspace, sandboxed Bash."""
    ws = f'/{workspace}'  # Permission rules spell absolute paths //like/this.
    allow = [f'Read({ws}/**)', f'Edit({ws}/**)',  # Edit rules cover every file-editing tool.
             f'Read(/{config.root}/stream/**)',  # the read-only live recording (runtime/stream)
             *([f'Read(/{robot_model_dir(config)}/**)'] if robot_model_dir(config) else []),  # robot-model/
             'Bash', 'Glob', 'Grep', 'TodoWrite', 'WebFetch', f'mcp__{MCP_SERVER}']
    if config.subagents:  # Claude Code's native subagents, every type, any model
        allow.append('Agent')
    protected = protected_paths(config)
    hosts = denied_hosts(config)
    deny = [*DENIED_TOOLS, *(f'WebFetch(domain:{host})' for host in hosts)]
    for path in protected:
        for tool in ('Read', 'Edit'):
            deny.append(f'{tool}(/{path})')
            if not path.suffix:  # Directories (or not yet created): everything below too.
                deny.append(f'{tool}(/{path}/**)')
    timeout_ms = (config.session_seconds + 300) * 1000
    return {
        'permissions': {'defaultMode': 'dontAsk', 'allow': allow, 'deny': deny},
        'sandbox': {
            'enabled': True, 'failIfUnavailable': True, 'autoAllowBashIfSandboxed': True,
            'allowUnsandboxedCommands': False,
            'filesystem': {'allowWrite': [str(workspace)],
                           'denyRead': [str(p) for p in (*protected, *HIDDEN_SOCKETS)]},
            # Internet over the sandbox's HTTP(S) proxy; loopback (the robot bridges) stays closed.
            'network': {'allowAllUnixSockets': True, 'allowLocalBinding': False, 'allowedDomains': ['*'],
                        'deniedDomains': hosts},
        },
        'autoUpdates': False, 'includeCoAuthoredBy': False, 'cleanupPeriodDays': 3650,
        'env': {
            'RAX_MCP_SOCKET': str(socket),
            # .pylib: packages the agent adds itself (pip install --target .pylib); its env is read-only.
            'PYTHONPATH': f'{workspace}:{workspace}/.pylib', 'PYTHONDONTWRITEBYTECODE': '1',
            'PATH': f'{agent_python(config)}:/usr/local/bin:/usr/bin:/bin',
            'API_TIMEOUT_MS': str(timeout_ms), 'MCP_TOOL_TIMEOUT': str(timeout_ms), 'MCP_TIMEOUT': '60000',
            'BASH_DEFAULT_TIMEOUT_MS': '600000', 'BASH_MAX_TIMEOUT_MS': str(config.session_seconds * 1000),
            'CLAUDE_CODE_MAX_RETRIES': '20', 'DISABLE_AUTOUPDATER': '1', 'DISABLE_TELEMETRY': '1', 'DISABLE_ERROR_REPORTING': '1',
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY': '1',
            'CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS': '0',
            # Every MCP tool stays a local tool definition (the relay refuses server tools).
            'ENABLE_TOOL_SEARCH': 'false'}}


def populate_workspace(config, workspace, *, gemini=True):
    """Template, skills, TASK.md and the contract for a new agent workspace."""
    from services.workspace import Setup, build, recipe_for
    for name in ('code', 'memory', 'output', 'runtime/frames', 'tmp', '.pylib'):
        (workspace/name).mkdir(parents=True, exist_ok=True)
    model = robot_model_dir(config)
    if model is not None:  # the vendor's URDF and meshes, shared read-only by every session
        (workspace/'robot-model').symlink_to(model, target_is_directory=True)
    # The live camera/robot recording lives outside the workspace: readable, never writable.
    (config.root/'stream').mkdir(mode=0o755, exist_ok=True)
    (workspace/'runtime'/'stream').symlink_to(config.root/'stream', target_is_directory=True)
    built = build(recipe_for(), Setup.from_config(config, gemini=gemini), workspace)
    (workspace/'TASK.md').write_text(f'# Task: {config.task}\n\n{config.task_description.strip()}\n')
    (workspace/'CLAUDE.md').write_text('@AGENTS.md\n')
    (workspace/'.claude').mkdir(exist_ok=True)
    (workspace/'.claude/skills').symlink_to('../.agents/skills', target_is_directory=True)
    from services import skill_library  # the persistent, agent-written task skills (separate from .agents/skills)
    built['skill_library'] = skill_library.checkout(config, workspace)
    return built


def deploy(config, *, key_file=None, gemini_key_file=None, claude=None):
    """Prepare a new session without starting the robot service or a model."""
    binary = claude_binary(claude)
    root = config.root
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    workspace, home = root/WORKSPACE_NAME, root/HOME_NAME
    if (root/'state.json').exists() or workspace.exists():
        raise FileExistsError('Use a fresh session root')
    provider = config.agent_provider
    model = config.agent_model or PROVIDERS[provider]['model']
    account = None
    if provider == 'claude-login':
        from harness.claude_cli import experiment_login
        key_file = key_file or experiment_login.token_file()  # Never the personal ~/.claude login.
        account = experiment_login.status()
    Credential(provider, key_file=key_file)  # Validate before provisioning.
    write_private_json(root/'model-relay.json', {
        'provider': provider, 'base_url': PROVIDERS[provider]['base_url'], 'model': model,
        'models': [ANY_MODEL],  # the agent and its subagents may use every model the provider serves
        'key_file': str(key_file) if key_file else None,
        'gemini_key_file': str(gemini_key_file) if gemini_key_file else None,
        # Identity only (no token): which dedicated account this session bills.
        'account': {k: account.get(k) for k in ('email', 'organization')} if account else None})
    write_private_json(root/'configuration.json', config.to_dict())
    workspace.mkdir(mode=0o700)
    home.mkdir(mode=0o700)
    write_private_json(root/'workspace-manifest.json', populate_workspace(config, workspace,
                                                                          gemini=bool(gemini_key_file)))
    socket = socket_path(root)
    (home/'settings.json').write_text(json.dumps(settings(config, workspace, socket), indent=2))
    (home/'.claude.json').write_text(json.dumps({
        'hasCompletedOnboarding': True,
        'projects': {str(workspace): {'hasTrustDialogAccepted': True, 'hasCompletedProjectOnboarding': True}}}))
    (home/'mcp.json').write_text(json.dumps({'mcpServers': {MCP_SERVER: {
        'type': 'stdio', 'command': sys.executable,
        'args': [str(PROJECT/'harness/claude_cli/mcp_bridge.py'), str(socket)]}}}))
    return deployed_paths(config, binary, model)


def deployed_paths(config, binary, model):
    root = config.root
    return {'root': root, 'workspace': root/WORKSPACE_NAME, 'claude_home': root/HOME_NAME, 'claude': binary,
            'model': model, 'configuration': root/'configuration.json',
            'relay_configuration': root/'model-relay.json', 'socket': socket_path(root),
            'subagents': config.subagents, 'effort': config.agent_effort, 'resume_session': None}


# Files and folders of a workspace that the harness writes and the agent does not own: refreshed on
# --resume so a resumed session runs with the current instructions, skills and tool contract.
HARNESS_OWNED = ('AGENTS.md', 'CLAUDE.md', 'START_PROMPT.md', 'MCP_SESSION.md', 'MCP_CONTRACT.json', '.agents/skills',
                 'api')


def refresh_harness(config, deployed):
    """Bring a stopped session's harness up to date without touching the agent's own work.

    Rewrites the Claude Code settings from the current configuration and code, removes subagent
    definitions an older harness wrote, lets the relay forward every model, and re-renders the
    harness-owned workspace files (HARNESS_OWNED) from a fresh build. memory/, code/, output/,
    runtime/, skill-library/, TASK.md and everything else the agent wrote stay as they are.
    Returns the refreshed paths.
    """
    import tempfile
    workspace, home = deployed['workspace'], deployed['claude_home']
    (home/'settings.json').write_text(json.dumps(settings(config, workspace, deployed['socket']), indent=2))
    for stale in ('perception-check.md', 'analyst.md'):  # custom types of an earlier harness
        (home/'agents'/stale).unlink(missing_ok=True)
    relay = json.loads(deployed['relay_configuration'].read_text())
    relay['models'] = [ANY_MODEL]
    write_private_json(deployed['relay_configuration'], relay)
    gemini = bool(relay.get('gemini_key_file'))
    with tempfile.TemporaryDirectory(prefix='rax-refresh-') as scratch:
        fresh = Path(scratch)/WORKSPACE_NAME
        fresh.mkdir()
        from services.workspace import Setup, build, recipe_for
        build(recipe_for(), Setup.from_config(config, gemini=gemini), fresh)
        for name in HARNESS_OWNED:
            source, target = fresh/name, workspace/name
            if not source.exists():
                continue
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
    for name in ('.pylib',):
        (workspace/name).mkdir(exist_ok=True)
    model = robot_model_dir(config)
    if model is not None and not (workspace/'robot-model').exists():
        (workspace/'robot-model').symlink_to(model, target_is_directory=True)
    if not (workspace/'skill-library'/'.git').is_dir():
        from services import skill_library
        skill_library.checkout(config, workspace)
    return [*HARNESS_OWNED, 'settings.json', 'model-relay.json']


def redeploy(config, claude=None):
    """Reattach an existing session for a resumed launch, with its harness refreshed (refresh_harness)."""
    relay_file = config.root/'model-relay.json'
    deployed = deployed_paths(config, claude_binary(claude), json.loads(relay_file.read_text())['model'])
    for key in ('workspace', 'claude_home', 'configuration', 'relay_configuration'):
        if not deployed[key].exists():
            raise FileNotFoundError(f'Cannot resume: {deployed[key]} is missing')
    deployed['refreshed'] = refresh_harness(config, deployed)
    deployed['resume_session'] = latest_claude_session(deployed['claude_home'])
    return deployed


def latest_claude_session(home):
    """Id of the newest top-level Claude Code conversation (subagent transcripts live below it)."""
    transcripts = sorted(Path(home, 'projects').glob('*/*.jsonl'), key=lambda p: p.stat().st_mtime)
    if not transcripts:
        raise RuntimeError('No Claude Code conversation to resume in '+str(home))
    return transcripts[-1].stem


def claude_arguments(deployed, *, prompt, interactive=False, json_output=False):
    denied = DENIED_TOOLS + (() if deployed.get('subagents') else SUBAGENT_TOOLS)
    home = deployed['claude_home']
    args = ['--model', deployed['model'], '--effort', deployed.get('effort', 'xhigh'),
            '--settings', str(home/'settings.json'),
            '--mcp-config', str(home/'mcp.json'), '--strict-mcp-config',
            '--permission-mode', 'dontAsk', '--disallowedTools', *denied]
    if deployed.get('resume_session'):
        args += ['--resume', deployed['resume_session']]  # Continue the session's own conversation.
    if not interactive:
        args = ['-p', *args, '--output-format', 'stream-json' if json_output else 'text', '--verbose']
    return [*args, '--', prompt]


def agent_environment(deployed, relay_address):
    """The agent CLI's process environment: nothing inherited but the terminal basics."""
    home = deployed['claude_home']
    environment = {k: os.environ[k] for k in ('TERM', 'LANG', 'LC_ALL', 'COLORTERM') if k in os.environ}
    environment.update({
        'HOME': str(home), 'CLAUDE_CONFIG_DIR': str(home),
        'PATH': f'{deployed["claude"].parent}:/usr/local/bin:/usr/bin:/bin',
        'ANTHROPIC_BASE_URL': f'http://{relay_address[0]}:{relay_address[1]}',
        'ANTHROPIC_AUTH_TOKEN': RELAY_TOKEN, 'RAX_MCP_SOCKET': str(deployed['socket'])})
    # Pin what Claude Code's model aliases (used by subagents: model opus / sonnet / haiku) resolve to.
    relay = json.loads(deployed['relay_configuration'].read_text())
    aliases = PROVIDERS.get(relay.get('provider'), {}).get('aliases', {})
    for alias, variable in (('opus', 'ANTHROPIC_DEFAULT_OPUS_MODEL'), ('sonnet', 'ANTHROPIC_DEFAULT_SONNET_MODEL'),
                            ('haiku', 'ANTHROPIC_DEFAULT_HAIKU_MODEL')):
        if alias in aliases:
            environment[variable] = aliases[alias]
    return environment


def relay_credential(deployed):
    relay = json.loads(deployed['relay_configuration'].read_text())
    return relay, Credential(relay['provider'], key_file=relay['key_file'])


def start_relay(config, deployed, *, max_model_calls):
    relay, credential = relay_credential(deployed)
    model = MessagesRelay(('127.0.0.1', 0), credential, base_url=relay['base_url'],
                          models=relay.get('models') or [relay['model']], max_calls=max_model_calls,
                          calls_file=config.root/'model-calls.json', client_write_timeout=config.session_seconds,
                          usage_log=config.root/'model-usage.jsonl')
    model.start()
    return model


def start_robot_service(config, deployed, log, *, started_at, timeout=120):
    """The robot service: sole robot owner, serving the agent CLI and the agent's scripts."""
    relay = json.loads(deployed['relay_configuration'].read_text())
    socket = deployed['socket']
    socket.unlink(missing_ok=True)
    args = [sys.executable, '-m', 'services.controller.frontend', '--config', str(deployed['configuration']),
            '--socket', str(socket), '--started-at', str(started_at)]
    if relay.get('gemini_key_file'):
        args += ['--key-file', relay['gemini_key_file']]
    environment = {**os.environ, 'PYTHONPATH': str(PROJECT)}
    server = subprocess.Popen(args, cwd=PROJECT, env=environment, stdout=log, stderr=log)
    deadline = time.monotonic() + timeout
    while not socket.exists():
        if server.poll() is not None or time.monotonic() >= deadline:
            stop(server)
            raise RuntimeError(f'Robot service startup failed; inspect {config.root}/frontend.log')
        time.sleep(.1)
    return server


def stop(process, grace=10):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def watch(agent, model, timeout, *, stop_on_limit=True):
    """Wait for the agent; stop it once the usage window has been exhausted for a while."""
    deadline, limited = time.monotonic() + timeout, None
    while agent.poll() is None:
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(agent.args, timeout)
        if stop_on_limit and model.limited_until > time.time():
            limited = limited or time.monotonic()
            if time.monotonic() - limited > USAGE_STOP_SECONDS:
                stop(agent, grace=30)
        time.sleep(1)


def until_usage_resets(run, model, *, deployed, prompt, budget, interactive, on_wait=None, sleep=time.sleep):
    """Run the agent. If the subscription usage window ran out during a run, wait for the reset
    (off the session clock; the robot service and relay stay up), then continue the agent's own
    conversation."""
    began, waited, waits = time.monotonic(), 0.0, 0
    while True:
        run_started = time.time()
        returncode = run(prompt, budget - (time.monotonic() - began - waited))
        if interactive or model.limited_until <= run_started or waits >= MAX_USAGE_WAITS:
            return returncode
        pause = max(0.0, model.limited_until + 60 - time.time())
        if on_wait:
            on_wait(model.limited_until, waited)
        sleep(pause)
        waited, waits = waited + pause, waits + 1
        if on_wait:
            on_wait(None, waited)
        deployed['resume_session'] = latest_claude_session(deployed['claude_home'])
        prompt = USAGE_PROMPT


def launch(config, deployed, *, prompt, interactive=True, seconds=None, max_model_calls=None, on_wait=None):
    """Start the relay and robot service, run Claude Code in the workspace, clean up."""
    relay_credential(deployed)  # Fail before starting anything.
    started_at = time.time()
    server = model = agent = None
    with (config.root/'frontend.log').open('ab') as log:
        try:
            model = start_relay(config, deployed, max_model_calls=max_model_calls or config.max_model_calls)
            server = start_robot_service(config, deployed, log, started_at=started_at)
            environment = agent_environment(deployed, model.address)
            budget = seconds or config.session_seconds

            def run(prompt, remaining):
                nonlocal agent
                command = [str(deployed['claude']), *claude_arguments(deployed, prompt=prompt, interactive=interactive)]
                agent = subprocess.Popen(command, cwd=deployed['workspace'], env=environment)
                watch(agent, model, remaining, stop_on_limit=not interactive)
                return agent.returncode
            return until_usage_resets(run, model, deployed=deployed, prompt=prompt, budget=budget,
                                      interactive=interactive, on_wait=on_wait)
        finally:
            stop(agent)
            stop(server, grace=30)
            if model:
                model.close()


def attempt_seconds(attempts):
    return sum(max(0.0, a['finished_at'] - a['started_at'] - a.get('usage_wait_seconds', 0.0)) for a in attempts
               if isinstance(a.get('started_at'), (int, float)) and isinstance(a.get('finished_at'), (int, float)))


def resumable_lifecycle(record):
    """Earlier attempts of a session, or a reason it cannot be resumed."""
    if not record.is_file():
        raise ValueError('Nothing to resume: this session was never launched')
    previous = json.loads(record.read_text())
    if previous.get('status') == 'prepared-only':
        raise ValueError('The session was only prepared; launch it instead')
    if not previous.get('finished_at'):
        # The caller holds launch.lock, so the previous launcher died without cleanup.
        logs = [p.stat().st_mtime for p in (record.parent/'frontend.log', record.parent/'state.json') if p.exists()]
        previous.update(status='crashed', finished_at=max(logs, default=previous.get('started_at', time.time())))
    return [*previous.pop('attempts', []), previous]


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True, help='Operator configuration JSON')
    parser.add_argument('--key-file', type=Path, help='Model credential (openrouter/anthropic key, or the '
                        'claude-login experiment token; default for claude-login: the experiment token)')
    parser.add_argument('--gemini-key-file', type=Path,
                        help=f'OpenRouter key enabling gemini_generate (default: {DEFAULT_GEMINI_KEY} if present)')
    parser.add_argument('--no-gemini', action='store_true', help='Run without gemini_generate')
    parser.add_argument('--prepare-only', action='store_true', help='Build the session; start no service or model')
    parser.add_argument('--exec-prompt', help='Run non-interactively with this prompt')
    parser.add_argument('--non-interactive', action='store_true', help='Run START_PROMPT.md without a terminal UI')
    parser.add_argument('--resume', action='store_true', help="Continue a stopped session's own conversation")
    args = parser.parse_args()
    config = Configuration.load(args.config)
    signal.signal(signal.SIGTERM, _interrupt)
    config.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (config.root/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = config.root/'agent-lifecycle.json'
        lifecycle = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'preparing'}
        seconds = config.session_seconds
        if args.resume:
            try:
                attempts = resumable_lifecycle(record)
            except ValueError as exc:
                raise SystemExit(str(exc))
            seconds -= int(attempt_seconds(attempts))
            if seconds < 60:
                raise SystemExit('The session has no wall-clock budget left to resume')
            lifecycle.update(resumed=True, attempts=attempts)
        elif record.exists():
            raise FileExistsError('This session was already launched; use --resume or a fresh configuration')
        write_private_json(record, lifecycle)
        try:
            if config.agent_provider == 'claude-login' and not args.key_file:
                from harness.claude_cli import experiment_login
                experiment_login.check()  # Fail before provisioning if the token is missing or revoked.
            gemini_key = None if args.no_gemini else args.gemini_key_file or (
                DEFAULT_GEMINI_KEY if DEFAULT_GEMINI_KEY.is_file() else None)
            warn_unsafe_defaults(config)
            if args.resume:
                deployed = redeploy(config)
            else:
                deployed = deploy(config, key_file=args.key_file, gemini_key_file=gemini_key)
            if args.prepare_only:
                print(json.dumps({k: str(v) for k, v in deployed.items()}, indent=2))
                lifecycle['status'] = 'prepared-only'
                return
            interactive = not (args.non_interactive or args.exec_prompt)
            prompt = args.exec_prompt or (RESUME_PROMPT if args.resume else
                                          (deployed['workspace']/'START_PROMPT.md').read_text().strip())
            base_wait = lifecycle.get('usage_wait_seconds', 0.0)

            def usage_wait(until, waited):
                if until is None:
                    lifecycle.update(status='running', usage_wait_seconds=base_wait + waited)
                    lifecycle.pop('usage_limited_until', None)
                else:
                    lifecycle.update(status='usage_wait', usage_limited_until=until)
                write_private_json(record, lifecycle)
            lifecycle['status'] = 'running'
            write_private_json(record, lifecycle)
            returncode = launch(config, deployed, prompt=prompt, interactive=interactive, seconds=seconds,
                                on_wait=usage_wait)
            lifecycle.update(status='exited', returncode=returncode)
            raise SystemExit(returncode)
        except KeyboardInterrupt:
            lifecycle['status'] = 'stopped'
            raise SystemExit(130)
        except Exception as exc:
            lifecycle.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            lifecycle['finished_at'] = time.time()
            write_private_json(record, lifecycle)
            if lifecycle['status'] != 'prepared-only':
                start_shipper(config.root)


def start_shipper(root):
    """Zip and ship this run off the robot once the launcher lets go of it (services.runtime_ship).

    Detached so the operator's terminal returns at once. Off unless RAX_SHIP_DEST names
    a destination; RAX_SHIP=0 disables it as well.
    """
    import os as _os
    from pathlib import Path as _Path
    import subprocess as _subprocess
    import sys as _sys
    from services.storage_root import artifact_root
    if _os.environ.get('RAX_SHIP', '1') == '0' or not _os.environ.get('RAX_SHIP_DEST'):
        return
    project = _Path(__file__).resolve().parents[2]
    try:
        log = open(artifact_root()/'ship.log', 'ab')
        _subprocess.Popen([_sys.executable, '-m', 'services.runtime_ship', '--session', str(root), '--sweep'],
                          cwd=project, stdin=_subprocess.DEVNULL, stdout=log, stderr=_subprocess.STDOUT,
                          start_new_session=True)
        print(f'shipping this run in the background; see {artifact_root()/"ship.log"}', file=_sys.stderr)
    except OSError as exc:
        print(f'could not start the runtime shipper: {exc}', file=_sys.stderr)


if __name__ == '__main__':
    main()
