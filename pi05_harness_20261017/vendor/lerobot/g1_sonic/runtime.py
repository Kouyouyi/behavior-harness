"""Checkpoint loading and fixed-schema inference for G1 SONIC baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contract import compose_state

DEFAULT_TASK = "move forward to the office chair and push it to the table"


def _observation_batch(
    observation: dict[str, Any],
    device: torch.device,
    *,
    task: str = "",
) -> dict[str, Any]:
    state = compose_state(observation)

    image = np.asarray(observation.get("observation.images.ego_view"))
    if image.shape != (360, 640, 3):
        raise ValueError(f"observation.images.ego_view must have shape (360, 640, 3), got {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("observation.images.ego_view must contain only finite values")
    image_tensor = torch.from_numpy(image)
    if image_tensor.dtype == torch.uint8:
        image_tensor = image_tensor.to(torch.float32) / 255.0
    else:
        image_tensor = image_tensor.to(torch.float32)

    return {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "observation.images.ego_view": image_tensor.permute(2, 0, 1).contiguous().unsqueeze(0).to(device),
        "task": task,
        "robot_type": "g1_sonic",
    }


@dataclass
class G1SonicRuntime:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    device: torch.device
    horizon: int
    policy_type: str = "act"
    task: str = ""

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        task = observation.get("task", self.task)
        text_conditioned_policies = {"pi05", "multi_task_dit"}
        if self.policy_type in text_conditioned_policies and not isinstance(task, str):
            raise TypeError(f"{self.policy_type} task must be a string, got {type(task).__name__}")
        if self.policy_type in text_conditioned_policies and not task.strip():
            raise ValueError(f"{self.policy_type} requires a non-empty task")
        batch = self.preprocessor(_observation_batch(observation, self.device, task=task))
        if self.policy_type == "diffusion":
            batch = {
                **batch,
                "observation.state": batch["observation.state"].unsqueeze(1),
            }
        if self.policy_type == "multi_task_dit":
            action = torch.stack(
                [self.policy.select_action(batch) for _ in range(self.horizon)],
                dim=1,
            )
        else:
            action = self.policy.predict_action_chunk(batch)
        action = self.postprocessor(action)
        if not isinstance(action, torch.Tensor):
            raise TypeError(
                f"{self.policy_type} postprocessor must return torch.Tensor, got {type(action).__name__}"
            )
        action_array = action.detach().to(device="cpu", dtype=torch.float32).numpy()
        if action_array.shape != (1, self.horizon, 78):
            raise ValueError(
                f"{self.policy_type} checkpoint must output shape "
                f"(1, {self.horizon}, 78), got {action_array.shape}"
            )
        result = action_array[0]
        if not np.isfinite(result).all():
            raise ValueError(f"{self.policy_type} checkpoint returned non-finite actions")
        return result


G1SonicACTRuntime = G1SonicRuntime


def load_policy(
    checkpoint_dir: Path,
    device: str = "cuda",
    *,
    expected_policy_type: str | None = None,
    task: str = DEFAULT_TASK,
) -> G1SonicRuntime:
    """Load a supported checkpoint and its serialized processors from one directory."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    supported_policy_types = {"act", "diffusion", "multi_task_dit", "pi05"}
    if config.type not in supported_policy_types:
        raise ValueError(f"expected one of {sorted(supported_policy_types)}, got policy type {config.type!r}")
    if expected_policy_type is not None and config.type != expected_policy_type:
        raise ValueError(f"expected {expected_policy_type} checkpoint, got policy type {config.type!r}")
    config.device = device
    policy = get_policy_class(config.type).from_pretrained(checkpoint_dir, config=config)
    policy = policy.to(torch.device(device)).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    horizon = int(config.horizon if config.type in {"diffusion", "multi_task_dit"} else config.chunk_size)
    return G1SonicRuntime(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=torch.device(device),
        horizon=horizon,
        policy_type=config.type,
        task=task if config.type in {"pi05", "multi_task_dit"} else "",
    )
