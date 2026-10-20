"""Export the minimal inference closure for a trained G1 SONIC checkpoint."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

UPSTREAM_TAG = "v0.6.0"
UPSTREAM_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
PI05_EXPANDED_TENSORS = (
    "model.action_in_proj.weight",
    "model.action_out_proj.weight",
    "model.action_out_proj.bias",
)


def export_runtime(
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    upstream_commit: str = UPSTREAM_COMMIT,
) -> None:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing runtime export: {output_dir}")

    config_path = checkpoint_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    policy_type = config.get("type", "act")
    if policy_type not in {"act", "diffusion", "multi_task_dit", "pi05"}:
        raise ValueError(f"unsupported G1 SONIC policy type: {policy_type!r}")
    horizon_key = "horizon" if policy_type in {"diffusion", "multi_task_dit"} else "chunk_size"
    horizon = int(config.get(horizon_key, 40))

    output_dir.mkdir(parents=True)
    shutil.copytree(checkpoint_dir, output_dir / "checkpoint")
    package_dir = output_dir / "g1_sonic_runtime"
    package_dir.mkdir()
    source_dir = Path(__file__).resolve().parent
    for filename in ("__init__.py", "contract.py", "runtime.py"):
        shutil.copy2(source_dir / filename, package_dir / filename)

    lerobot_requirement = {
        "act": "lerobot==0.6.0",
        "diffusion": "lerobot[diffusion]==0.6.0",
        "multi_task_dit": "lerobot[multi_task_dit]==0.6.0",
        "pi05": "lerobot[pi]==0.6.0",
    }[policy_type]
    (output_dir / "requirements.txt").write_text(
        "--extra-index-url https://download.pytorch.org/whl/cu128\n"
        'torch==2.7.0+cu128; sys_platform == "linux"\n'
        'torchvision==0.22.0+cu128; sys_platform == "linux"\n'
        "transformers==5.5.4\n"
        f"{lerobot_requirement}\n"
    )
    manifest = {
        "schema_version": f"humano.g1_sonic.{policy_type}.v2",
        "policy": policy_type,
        "upstream_tag": UPSTREAM_TAG,
        "upstream_commit": upstream_commit,
        "observation": {
            "observation.images.ego_view": {"dtype": "uint8|float32", "shape": [360, 640, 3]},
            "observation.state": {"dtype": "float32", "shape": [43]},
            "observation.projected_gravity": {"dtype": "float32", "shape": [3]},
        },
        "model_input": {
            "observation.state": {
                "dtype": "float32",
                "shape": [46],
                "layout": ["joint_state[43]", "projected_gravity[3]"],
            },
        },
        "action": {
            "dtype": "float32",
            "shape": [horizon, 78],
            "layout": ["motion_token[64]", "left_hand[7]", "right_hand[7]"],
        },
    }
    if policy_type == "pi05":
        manifest["task"] = "move forward to the office chair and push it to the table"
        manifest["pretrained_initialization"] = {
            "base": "lerobot/pi05_base",
            "strategy": "weight_surgery",
            "preserved_action_dimensions": 32,
            "extra_action_dimensions": 46,
            "extra_weight_initialization": "xavier_uniform",
            "extra_output_bias_initialization": "zeros",
            "expanded_tensors": list(PI05_EXPANDED_TENSORS),
        }
    (output_dir / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "observation_fixture.npz",
        **{
            "observation.images.ego_view": np.zeros((360, 640, 3), dtype=np.uint8),
            "observation.state": np.zeros(43, dtype=np.float32),
            "observation.projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        },
    )
    (output_dir / "expected_output.json").write_text(
        json.dumps(
            {
                "dtype": "float32",
                "finite": True,
                "shape": [horizon, 78],
            },
            indent=2,
        )
        + "\n"
    )

    license_path = source_dir.parents[2] / "LICENSE"
    if license_path.is_file():
        shutil.copy2(license_path, output_dir / "LICENSE")
