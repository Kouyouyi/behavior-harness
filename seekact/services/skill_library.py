"""The agent's persistent skill library: task skills learned across sessions, one git repository per robot.

Separate from the harness's read-only skills (`agent_skills/`, rendered into `.agents/skills/`):
those are written by people; this library is written by agents, one session at a time.

    <artifact root>/skill-library/<robot>/      the library (git, branch main); only the harness writes it
    <session>/agent-workspace/skill-library/    the session's clone, on branch session/<session name>

The agent reads the library, runs its skills, and commits new or improved skills on its
session branch. `merge()` (the harness tool `skill_library_merge`) validates the branch and
merges it into main. The harness never runs git inside the agent's clone (its config and
hooks are the agent's): it copies the clone's object store to a scratch directory with a
fresh config and no hooks, fetches the branch from there into the library, validates the
files of that commit as data (code is only compiled, never run), and merges.

Format (enforced by `validate_tree`, documented in the library README and the agent's
`continual-learning` skill):

    INDEX.md                         one table row per skill: slug link, status, what, when
    skills/<slug>/SKILL.md           frontmatter + the required sections below
    skills/<slug>/code/*.py          the executable procedure; the entry defines main(ctx, ...)
    skills/<slug>/evidence/runs.jsonl  one JSON object per run of the code
    skills/<slug>/evidence/images/   visual references (jpg / png) the runs and SKILL.md cite
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

LIBRARY_DIR = 'skill-library'
ENVIRONMENT = 'RAX_SKILL_LIBRARY'
STATUSES = ('draft', 'verified', 'deprecated')
SECTIONS = ('When to use', 'Preconditions', 'Procedure', 'Code', 'Checks', 'Failure modes', 'Evidence')
RUN_KEYS = ('session', 'date', 'commit', 'outcome', 'unattended', 'conditions', 'checks', 'images')
VERIFIED_MIN_RUNS = 2       # unattended successful runs of the committed code
MAX_IMAGE_BYTES = 400_000
MAX_FILE_BYTES = 2_000_000
MAX_SKILL_BYTES = 25_000_000  # saved frames (RGB + depth + meta) let later sessions replay perception offline
SLUG = re.compile(r'^[a-z0-9][a-z0-9-]{1,60}$')
GIT_ENV = {'GIT_CONFIG_NOSYSTEM': '1', 'GIT_TERMINAL_PROMPT': '0', 'HOME': '/nonexistent'}

README = """# {robot} skill library

Skills agents learned on this robot, kept across sessions. Agents read and use them, and
add or improve them on their session branch; the harness tool `skill_library_merge`
checks the format and evidence and merges into `main`. The protocol is the agent skill
`continual-learning`; the rules the merge enforces:

- `INDEX.md`: one row per skill, `| [slug](skills/slug/SKILL.md) | status | what it does | when to use |`.
- `skills/<slug>/SKILL.md` begins with frontmatter (`name` = the folder name, `description`,
  `status`: draft | verified | deprecated, `robot`, `entry`: the code file, e.g.
  `code/run.py`) and has the sections {sections}.
- The entry is Python defining `main(ctx, ...)` (ctx: `api.runtime.Context`) that returns
  `{{"ok": bool, "measured": {{...}}, "why": str}}`.
- `evidence/runs.jsonl`: one JSON object per run of the code with {run_keys}; `images` are
  paths under `evidence/images/` (jpg / png, at most {image_kb} kB each).
- `## Checks` holds a `### Perception checkpoints` table: for each checkpoint the step, the
  perception subagent, frames, question, expected answer, pass criterion and action on
  fail (or "None").
- At least one image per skill. `verified` needs at least {verified} unattended successful
  runs. A branch may only change `INDEX.md` and files under `skills/`.
- Never delete another session's skill: set its status to `deprecated` and say why.
"""
INDEX = """# Skill index

| Skill | Status | What it does | When to use |
|---|---|---|---|
"""


class LibraryError(RuntimeError):
    pass


def _git(*args, cwd=None, check=True):
    result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, timeout=60,
                            env={**os.environ, **GIT_ENV})
    if check and result.returncode:
        raise LibraryError(f"git {' '.join(args)}: {(result.stderr or result.stdout).strip()[:500]}")
    return result


def library_path(config):
    """The robot's library: RAX_SKILL_LIBRARY, else <artifact root>/skill-library/<robot> next to sessions/."""
    configured = os.environ.get(ENVIRONMENT)
    if configured:
        return Path(configured)/config.robot
    sessions = Path(config.root).resolve().parent
    base = sessions.parent if sessions.name == 'sessions' else sessions
    return base/LIBRARY_DIR/config.robot


def ensure_library(path, robot):
    """Create the library repository (README, INDEX, skills/) on first use."""
    path = Path(path)
    if (path/'.git').is_dir():
        return path
    path.mkdir(parents=True, exist_ok=True)
    _git('init', '-q', '-b', 'main', cwd=path)
    _git('config', 'user.name', 'rax harness', cwd=path)
    _git('config', 'user.email', 'harness@rax.local', cwd=path)
    (path/'README.md').write_text(README.format(robot=robot, sections=', '.join(f'`## {s}`' for s in SECTIONS),
                                                run_keys=', '.join(f'`{k}`' for k in RUN_KEYS),
                                                image_kb=MAX_IMAGE_BYTES // 1000, verified=VERIFIED_MIN_RUNS))
    (path/'INDEX.md').write_text(INDEX)
    (path/'skills').mkdir(exist_ok=True)
    (path/'skills'/'.gitkeep').write_text('')
    _git('add', '-A', cwd=path)
    _git('commit', '-q', '-m', f'Initialize the {robot} skill library', cwd=path)
    return path


def branch_name(config):
    return f'session/{Path(config.root).name}'


def checkout(config, workspace):
    """Clone the library into the new workspace on the session's branch; returns its summary."""
    library = ensure_library(library_path(config), config.robot)
    clone = Path(workspace)/LIBRARY_DIR
    _git('clone', '-q', '--no-hardlinks', str(library), str(clone))
    _git('checkout', '-q', '-b', branch_name(config), cwd=clone)
    _git('config', 'user.name', f'agent {Path(config.root).name}', cwd=clone)
    _git('config', 'user.email', 'agent@rax.local', cwd=clone)
    return {'library': str(library), 'clone': str(clone), 'branch': branch_name(config),
            'skills': len(list((clone/'skills').glob('*/SKILL.md')))}


# ----- validation (files of a commit, read as data) --------------------------------------------
def _frontmatter(text):
    match = re.match(r'^---\n(.*?)\n---\n', text, re.S)
    if not match:
        return None
    fields = {}
    for line in match.group(1).splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            value = re.sub(r'\s+#.*$', '', value)  # a trailing YAML comment
            fields[key.strip()] = value.strip().strip('"\'')
    return fields


def _list_field(value):
    """A frontmatter list written inline: `[a, b]` or `a, b`."""
    return [item.strip().strip('"\'') for item in value.strip().strip('[]').split(',') if item.strip()]


def validate_skill(folder):
    """Problems of one skill folder (a list of strings; empty when it follows the format)."""
    folder = Path(folder)
    slug = folder.name
    problems = []

    def bad(message):
        problems.append(f'skills/{slug}: {message}')
    links = [p for p in [folder, *folder.rglob('*')] if p.is_symlink()]
    if links:  # refuse before reading anything through a link
        bad(f'symlinks are not allowed: {[str(p.relative_to(folder.parent)) for p in links][:5]}')
        return problems
    if not SLUG.match(slug):
        bad('folder name must be lower-case letters, digits and hyphens')
    total = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    if total > MAX_SKILL_BYTES:
        bad(f'{total / 1e6:.1f} MB is over the {MAX_SKILL_BYTES / 1e6:g} MB per skill')
    for p in folder.rglob('*'):
        if p.is_file() and p.stat().st_size > MAX_FILE_BYTES:
            bad(f'{p.relative_to(folder)} is over {MAX_FILE_BYTES / 1e6:g} MB')
    skill = folder/'SKILL.md'
    if not skill.is_file():
        bad('SKILL.md missing')
        return problems
    text = skill.read_text(errors='replace')
    meta = _frontmatter(text)
    if meta is None:
        bad('SKILL.md must begin with --- frontmatter ---')
        return problems
    for key in ('name', 'description', 'status', 'robot', 'entry'):
        if not meta.get(key):
            bad(f'frontmatter needs {key}')
    if meta.get('name') and meta['name'] != slug:
        bad(f"frontmatter name {meta['name']!r} must equal the folder name")
    status = meta.get('status')
    if status and status not in STATUSES:
        bad(f'status must be one of {", ".join(STATUSES)}')
    headings = set(re.findall(r'^##\s+(.+?)\s*$', text, re.M))
    for section in SECTIONS:
        if section not in headings:
            bad(f'SKILL.md needs a "## {section}" section')
    if not re.search(r'^###\s+Perception checkpoints\s*$', text, re.M):
        bad('SKILL.md needs a "### Perception checkpoints" table under "## Checks" (what each perception '
            'subagent checks: step, subagent, frames, question, expected answer, pass criterion, on fail; '
            'or "None")')
    entry = meta.get('entry')
    if entry:
        code = (folder/entry).resolve()
        if not code.is_relative_to(folder.resolve()) or not code.is_file() or code.suffix != '.py':
            bad(f'entry {entry} must be a .py file inside the skill folder')
        else:
            try:
                compile(code.read_text(errors='replace'), entry, 'exec')  # syntax only: nothing runs
            except (SyntaxError, ValueError) as exc:
                bad(f'entry {entry} does not compile: {str(exc)[:200]}')
            if not re.search(r'^def main\(\s*ctx\b', code.read_text(errors='replace'), re.M):
                bad(f'entry {entry} must define main(ctx, ...)')
    images = folder/'evidence'/'images'
    image_files = [p for p in images.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')] if images.is_dir() else []
    if not image_files:
        bad('evidence/images/ needs at least one image (a visual reference of the start state or result)')
    for p in image_files:
        if p.stat().st_size > MAX_IMAGE_BYTES:
            bad(f'{p.relative_to(folder)} is {p.stat().st_size // 1000} kB (at most {MAX_IMAGE_BYTES // 1000} kB)')
    runs_file = folder/'evidence'/'runs.jsonl'
    runs = []
    if runs_file.is_file():
        for n, line in enumerate(runs_file.read_text(errors='replace').splitlines(), start=1):
            if not line.strip():
                continue
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                bad(f'evidence/runs.jsonl line {n} is not JSON')
                continue
            missing = [k for k in RUN_KEYS if k not in run]
            if missing:
                bad(f'evidence/runs.jsonl line {n} lacks {missing}')
                continue
            if run['outcome'] not in ('success', 'failure'):
                bad(f'evidence/runs.jsonl line {n}: outcome must be success or failure')
            for image in run['images'] if isinstance(run['images'], list) else []:
                path = (folder/image).resolve()
                if not path.is_relative_to(images.resolve()) or not path.is_file():
                    bad(f'evidence/runs.jsonl line {n}: image {image} is not a file under evidence/images/')
            runs.append(run)
    elif status == 'verified':
        bad('verified needs evidence/runs.jsonl')
    if status == 'verified':
        good = [r for r in runs if r.get('outcome') == 'success' and r.get('unattended') is True and r.get('images')]
        if len(good) < VERIFIED_MIN_RUNS:
            bad(f'verified needs at least {VERIFIED_MIN_RUNS} unattended successful runs with images in '
                f'evidence/runs.jsonl (has {len(good)}); keep it draft until then')
    return problems


def validate_tree(tree, changed):
    """Problems of a library tree, for the files a branch changed."""
    tree = Path(tree)
    problems = []
    outside = [p for p in changed if not (p == 'INDEX.md' or p.startswith('skills/'))]
    if outside:
        problems.append(f'a session branch may only change INDEX.md and skills/; it changed {sorted(outside)[:10]}')
    slugs = sorted({p.split('/')[1] for p in changed if p.startswith('skills/') and p.count('/') >= 2})
    index_path = tree/'INDEX.md'
    if index_path.is_symlink():
        return problems + ['INDEX.md must be a file, not a symlink'], []
    index = index_path.read_text(errors='replace') if index_path.is_file() else ''
    for slug in slugs:
        folder = tree/'skills'/slug
        if not folder.is_dir():
            problems.append(f'skills/{slug} was deleted; set its status to deprecated instead')
            continue
        problems += validate_skill(folder)
        if f'(skills/{slug}/SKILL.md)' not in index:
            problems.append(f'INDEX.md needs a row linking skills/{slug}/SKILL.md')
        meta = _frontmatter((folder/'SKILL.md').read_text(errors='replace')) if (folder/'SKILL.md').is_file() else None
        for used in _list_field((meta or {}).get('uses', '')):
            if not (tree/'skills'/used/'SKILL.md').is_file():
                problems.append(f'skills/{slug}: uses {used!r}, which is not a skill in the library')
    return problems, slugs


# ----- merge (the harness tool) ---------------------------------------------------------------
def _copy_objects(source, target):
    """Copy a git object store without symlinks and without alternates (no reads outside the clone)."""
    source = Path(source)
    for directory, names, files in os.walk(source, followlinks=False):
        here = Path(directory)
        out = Path(target)/here.relative_to(source)
        out.mkdir(parents=True, exist_ok=True)
        names[:] = [n for n in names if not (here/n).is_symlink()]
        for name in files:
            path = here/name
            if path.is_symlink() or (here.name == 'info' and name in ('alternates', 'http-alternates')):
                continue
            shutil.copyfile(path, out/name)



def merge(config, workspace, message):
    """Validate the session branch of the workspace's clone and merge it into the library's main."""
    library = ensure_library(library_path(config), config.robot)
    clone = Path(workspace)/LIBRARY_DIR
    branch = branch_name(config)
    if not (clone/'.git').is_dir():
        raise LibraryError('the workspace has no skill-library clone')
    lock = open(library/'.git'/'rax-merge.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    scratch = Path(tempfile.mkdtemp(prefix='skill-merge-'))
    try:
        # The agent's objects only: no config, hooks or info from its .git.
        objects = scratch/'agent.git'
        objects.mkdir()
        _copy_objects(clone/'.git'/'objects', objects/'objects')
        refs = clone/'.git'/'refs'/'heads'/Path(branch)
        packed = clone/'.git'/'packed-refs'
        head = refs.read_text().strip() if refs.is_file() else None
        if head is None and packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(f' refs/heads/{branch}'):
                    head = line.split()[0]
        if not head or not re.fullmatch(r'[0-9a-f]{40}', head):
            raise LibraryError(f'no commit on {branch}: commit your skill first (git add, git commit)')
        (objects/'refs'/'heads').mkdir(parents=True)
        (objects/'HEAD').write_text('ref: refs/heads/main\n')
        (objects/'config').write_text('[core]\n\trepositoryformatversion = 0\n\tbare = true\n')
        _git('fetch', '-q', '--no-tags', str(objects), head, cwd=library)
        base = _git('merge-base', 'main', head, cwd=library).stdout.strip()
        changed = [p for p in _git('diff', '--name-only', base, head, cwd=library).stdout.splitlines() if p]
        if not changed:
            return {'merged': False, 'reason': 'no committed changes on the session branch since its base',
                    'branch': branch, 'head': head}
        tree = scratch/'tree'
        tree.mkdir()
        archive = subprocess.run(['git', 'archive', '--format=tar', head], cwd=library, capture_output=True,
                                 timeout=60, env={**os.environ, **GIT_ENV})
        if archive.returncode:
            raise LibraryError('git archive failed: ' + archive.stderr.decode(errors='replace')[:300])
        subprocess.run(['tar', '-x', '-C', str(tree)], input=archive.stdout, check=True, timeout=60)
        problems, slugs = validate_tree(tree, changed)
        if problems:
            return {'merged': False, 'branch': branch, 'head': head, 'problems': problems,
                    'do': 'fix these in skill-library/, commit, and call skill_library_merge again'}
        before = _git('rev-parse', 'main', cwd=library).stdout.strip()
        result = _git('merge', '--no-ff', '-m', f'Merge {branch}: {message.strip()[:200]}', head, cwd=library,
                      check=False)
        if result.returncode:
            conflicts = _git('diff', '--name-only', '--diff-filter=U', cwd=library, check=False).stdout.split()
            _git('merge', '--abort', cwd=library, check=False)
            return {'merged': False, 'branch': branch, 'head': head, 'conflicts': conflicts,
                    'do': 'in skill-library/: git fetch origin && git merge origin/main, resolve the conflicts '
                          '(keep both sessions\' evidence), commit, and call skill_library_merge again'}
        after = _git('rev-parse', 'main', cwd=library).stdout.strip()
        return {'merged': True, 'branch': branch, 'head': head, 'library_main': after, 'previous_main': before,
                'skills': slugs, 'changed_files': len(changed),
                'next': 'git fetch origin in skill-library/ to see the merged main'}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
