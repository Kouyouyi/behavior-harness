"""Workspace generation for the real-robot agent: template, skills, blocks and contract."""
import json
from pathlib import Path
import re

import pytest

from services.workspace import ROBOT_FACTS, Setup, build, recipe_for, render
from services.workspace.templates import registry

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('profile', ['rgbd', 'rgb-only'])
def test_rby1_workspace_renders_its_own_robot_skills(tmp_path, profile):
    workspace = tmp_path/'ws'
    manifest = build(recipe_for(), Setup('rby1', 'tidy_table', 'claude', profile), workspace)
    for path in workspace.rglob('*.md'):
        text = path.read_text()
        assert '{{' not in text and '{%' not in text, path
        assert not re.search(r'[Rr]obo[Dd]ojo|BEHAVIOR|rehears|formal batch|container|[Ff]ranka|[Dd]exmate|Vega', text), path
    skills = set(p.name for p in (workspace/'.agents/skills').iterdir())
    assert {'manual-robot-control', 'experience-memory', 'live-stream', 'gemini', 'robot-interface',
            'whole-body-control', 'base-navigation', 'grasping', 'object-localization'} <= skills
    assert 'motion-planning' not in skills  # no planner is wired into this backend
    localization = (workspace/'.agents/skills/object-localization/SKILL.md').read_text()
    assert ('cannot back-project' in localization) == (profile == 'rgb-only')
    assert ('triangulat' in localization) == (profile == 'rgb-only')
    assert (workspace/'api/stream.py').is_file() and (workspace/'api/runtime.py').is_file()
    assert 'rby1_stream' in (workspace/'.agents/skills/live-stream/SKILL.md').read_text()
    assert 'real hardware' in (workspace/'MCP_SESSION.md').read_text()
    interface = (workspace/'.agents/skills/robot-interface/SKILL.md').read_text()
    assert 'RB-Y1' in interface and 'mock_move' not in interface  # no other robot's tools
    contract = json.loads((workspace/'MCP_CONTRACT.json').read_text())
    assert contract['robot'] == 'rby1' and set(ROBOT_FACTS['rby1']['tools']) <= set(contract['tools'])
    assert manifest['contract'] == contract
    agents = (workspace/'AGENTS.md').read_text()
    assert 'Rainbow RB-Y1' in agents and 'mock robot' not in agents


def test_subagent_text_only_with_subagents(tmp_path):
    build(recipe_for(), Setup('rby1', subagents=0), tmp_path/'none')
    build(recipe_for(), Setup('rby1', subagents=2), tmp_path/'two')
    assert 'subagents' not in (tmp_path/'none/AGENTS.md').read_text()
    assert "Claude Code's subagents" in (tmp_path/'two/AGENTS.md').read_text()


def test_blocks_registry_and_unknown_names_fail():
    assert set(registry()) == {'session.contract-pointer', 'images.forwarding'}
    with pytest.raises(ValueError, match='Unknown workspace template name'):
        render('{{ nonexistent }}', Setup('rby1'))
    with pytest.raises(ValueError, match='Unknown workspace block'):
        render('{{ block("missing") }}', Setup('rby1'))


def test_every_template_variable_is_a_setup_fact():
    """Sources only use names Setup.context() and vocab provide (StrictUndefined renders all)."""
    for path in [*ROOT.glob('auto_research_agent/*.md'), *ROOT.glob('agent_skills/**/*.md'),
                 *ROOT.glob('agent_blocks/**/*.md')]:
        if path.name == 'README.md' and path.parent.name == 'agent_skills':
            continue
        render(path.read_text(), Setup('rby1'))


def test_agents_md_links_the_rby1_limits_and_posture_skills(tmp_path):
    workspace = tmp_path/'ws'
    build(recipe_for(), Setup('rby1', 'tidy_table', 'claude', 'rgbd'), workspace)
    text = (workspace/'AGENTS.md').read_text()
    links = set(re.findall(r'\]\((\.agents/skills/[^)]+)\)', text))
    assert all((workspace/link).is_file() for link in links)
    for skill in ('robot-interface', 'whole-body-control'):
        assert f'.agents/skills/{skill}/SKILL.md' in links
    for tool in ('rby1_fk', 'rby1_check'):
        assert tool in (workspace/'.agents/skills/robot-interface/SKILL.md').read_text()
