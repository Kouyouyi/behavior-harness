"""Simulation-only contact ablation around the existing DDS 77 / lo scene."""
import argparse
import json
from pathlib import Path
import sys

from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
from shared_control import sim_scene


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--disable-hand-collisions', action='store_true')
    parser.add_argument('--disable-arm-collisions', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args, remaining = parser.parse_known_args()
    original = DefaultEnv.init_scene

    def initialize(self):
        original(self)
        changed = []
        if args.disable_hand_collisions or args.disable_arm_collisions:
            model = self.mj_model
            for g in range(model.ngeom):
                body = model.body(int(model.geom_bodyid[g])).name
                parts = ('_hand_', '_wrist_', '_elbow_', '_shoulder_') if args.disable_arm_collisions else ('_hand_',)
                if any(part in body for part in parts) and (model.geom_contype[g] or model.geom_conaffinity[g]):
                    changed.append(dict(geom_id=g, body=body, contype=int(model.geom_contype[g]),
                                        conaffinity=int(model.geom_conaffinity[g])))
                    model.geom_contype[g] = 0
                    model.geom_conaffinity[g] = 0
            if not changed:
                raise ValueError('No hand collision geometry found for diagnostic')
        (args.output/'collision_override.json').write_text(json.dumps(dict(disabled=changed,
            mass_and_inertia_unchanged=True, diagnostic_only=True, deploy_to_hardware=False), indent=2)+'\n')

    DefaultEnv.init_scene = initialize
    sys.argv = [sys.argv[0], '--output', str(args.output), *remaining]
    sim_scene.main()


if __name__ == '__main__':
    main()
