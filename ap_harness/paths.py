from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / 'pi05_harness_20261017'
LEGACY = ROOT / 'runtime'
PYTHON = '/home/cody/miniconda3/envs/behavior-2026/bin/python'
# Keep Isaac's pinned dependencies ahead of the extra planner-only packages.
sys.path.extend([str(ROOT / 'deps'), str(HANDOFF / 'scripts'), str(LEGACY)])
