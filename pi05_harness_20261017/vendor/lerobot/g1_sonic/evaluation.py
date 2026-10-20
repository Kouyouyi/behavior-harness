"""Offline checkpoint smoke evaluation for G1 SONIC baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contract import G1SonicObservationSpec
from .runtime import DEFAULT_TASK, G1SonicRuntime, load_policy
from .training import validate_prepared_dataset


def evaluate_sample(runtime: G1SonicRuntime, sample: dict[str, Any]) -> dict[str, Any]:
    """Run one LeRobot dataset sample through the fixed-schema runtime."""

    image = sample["observation.images.ego_view"]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().permute(1, 2, 0).numpy()
    state = sample["observation.state"]
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    spec = G1SonicObservationSpec()
    if state.shape != (spec.total_dim,):
        raise ValueError(f"prepared observation.state must have shape ({spec.total_dim},), got {state.shape}")

    action = runtime.predict(
        {
            "observation.images.ego_view": image,
            "observation.state": state[: spec.joint_state_dim],
            "observation.projected_gravity": state[spec.joint_state_dim :],
        }
    )
    return {"finite": bool(np.isfinite(action).all()), "shape": list(action.shape)}


def evaluate_checkpoint(
    checkpoint_dir: Path,
    dataset_root: Path,
    device: str = "cuda",
    *,
    expected_policy_type: str | None = None,
    task: str = DEFAULT_TASK,
) -> dict[str, Any]:
    """Load one local dataset sample and verify a trained checkpoint."""

    dataset_root = Path(dataset_root).resolve()
    validate_prepared_dataset(dataset_root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=f"g1_sonic/{dataset_root.name}",
        root=dataset_root,
        episodes=[0],
    )
    runtime = load_policy(
        checkpoint_dir,
        device=device,
        expected_policy_type=expected_policy_type,
        task=task,
    )
    return evaluate_sample(runtime, dataset[0])
