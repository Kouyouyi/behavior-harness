"""Load existing SONIC public serializers without importing teleop publishers."""
import hashlib
import importlib.util
from pathlib import Path

SONIC = Path('/home/user/GR00T-WholeBodyControl')
WIRE = SONIC/'gear_sonic/utils/teleop/zmq/zmq_planner_sender.py'


def wire_module():
    spec = importlib.util.spec_from_file_location('rax_sonic_wire', WIRE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
