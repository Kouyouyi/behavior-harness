"""The workspace recipe for the agent that controls the real robot."""
from .recipe import PROJECT, Recipe, SkillRule, Vocab
from .setup import ROBOT_FACTS

VOCAB = Vocab(memory='memory', notes='memory/PROGRESS.md', frames='runtime/frames/', inventory='status')

# Skill sources by axis. Every workspace takes the task skills, the perception skills of
# its observation mode ('depth' or 'rgb-only') and robot, the control skills of its robot,
# and the auto-research skills. Agents see one flat set with robot-neutral names.
SKILL_LAYERS = ('task', 'perception/common', 'perception/{perception}', 'perception/{perception}-{robot}',
                'control/common', 'control/{robot}', 'auto-research')

# No OpenRouter key, no gemini_generate tool: leave its skill out instead of advertising it.
# Likewise a robot without a live recording does not get that skill.
AGENT = Recipe('real-robot/agent', PROJECT / 'auto_research_agent', SKILL_LAYERS, vocab=VOCAB,
               omit_skills=(SkillRule('gemini', lambda setup: not setup.gemini),
                            SkillRule('live-stream', lambda setup: not ROBOT_FACTS[setup.robot]['stream'])))


def recipe_for(options=frozenset()):
    return AGENT
