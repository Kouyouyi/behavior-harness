from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .compat import load_psi_module
from .configuration_psi0 import Psi0Config


def _as_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).tolist()


def _select_stats(stats: dict[str, Any], preferred_key: str, fallbacks: tuple[str, ...]) -> dict[str, Any]:
    for key in (preferred_key, *fallbacks):
        if key in stats:
            return stats[key]
    raise KeyError(f"Missing Psi0 statistics. Tried {(preferred_key, *fallbacks)}")


def _resolve_stats(
    config: Psi0Config, dataset_stats: dict[str, dict[str, torch.Tensor]] | None
) -> tuple[list[float], list[float], list[float] | None, list[float] | None]:
    stats: dict[str, Any] = dataset_stats or {}
    if config.stats_path is not None:
        path = Path(config.stats_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Psi0 statistics file does not exist: {path}")
        stats = json.loads(path.read_text())

    action_stats = _select_stats(stats, config.action_stats_key, (ACTION,))
    state_stats = _select_stats(stats, config.state_stats_key, (config.state_key, "states"))
    low_key, high_key = ("q01", "q99") if config.action_norm_type == "bounds_q99" else ("min", "max")

    action_min = _as_list(action_stats[low_key])
    action_max = _as_list(action_stats[high_key])
    state_min = _as_list(state_stats[low_key]) if config.normalize_state else None
    state_max = _as_list(state_stats[high_key]) if config.normalize_state else None
    return action_min, action_max, state_min, state_max


@dataclass
@ProcessorStepRegistry.register(name="psi0_batch_processor")
class Psi0BatchProcessorStep(ProcessorStep):
    """Run the canonical Psi0 normalize/image/Qwen transform on a LeRobot batch."""

    psi0_source_dir: str
    model_name_or_path: str
    image_keys: list[str]
    state_key: str
    task_key: str
    action_dim: int
    chunk_size: int
    model_max_length: int
    action_min: list[float]
    action_max: list[float]
    state_min: list[float] | None
    state_max: list[float] | None
    normalize_state: bool
    image_resize: tuple[int, int]
    image_center_crop: tuple[int, int]
    image_augmentation: bool

    _model_transform: Any = field(default=None, init=False, repr=False)
    _vlm_processor: Any = field(default=None, init=False, repr=False)
    _pt_to_pil: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        transform_module = load_psi_module(self.psi0_source_dir, "config.transform")
        utils_module = load_psi_module(self.psi0_source_dir, "utils")
        transformers = __import__("transformers", fromlist=["AutoProcessor"])
        self._vlm_processor = transformers.AutoProcessor.from_pretrained(self.model_name_or_path)
        self._pt_to_pil = utils_module.pt_to_pil
        self._model_transform = transform_module.Psi0ModelTransform(
            resize={"size": self.image_resize},
            center_crop={"size": self.image_center_crop},
            img_aug=self.image_augmentation,
        )

    @staticmethod
    def _normalize(values: np.ndarray, low: list[float], high: list[float], *, state: bool) -> np.ndarray:
        low_np = np.asarray(low, dtype=np.float32)
        high_np = np.asarray(high, dtype=np.float32)
        ill = np.abs(high_np - low_np) < 1e-4 * (np.abs(high_np) + np.abs(low_np) + 1e-8)
        safe_high = high_np.copy()
        safe_high[ill] = 1.0
        normalized = np.where(
            ill,
            0.0 if state else values,
            (values - low_np) / (safe_high - low_np) * 2.0 - 1.0,
        )
        normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32)
        return np.nan_to_num(normalized, nan=0.0) if state else normalized

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        if self.state_key not in observation:
            raise KeyError(f"Psi0 requires {self.state_key!r}")
        for key in self.image_keys:
            if key not in observation:
                raise KeyError(f"Psi0 requires image feature {key!r}")

        states = observation[self.state_key]
        if states.ndim == 1:
            states = states.unsqueeze(0)
        batch_size = states.shape[0]
        tasks = complementary.get(self.task_key)
        if tasks is None:
            raise KeyError(f"Psi0 requires complementary field {self.task_key!r}")
        if isinstance(tasks, str):
            tasks = [tasks]

        actions = transition.get(TransitionKey.ACTION)
        if actions is None:
            actions = torch.zeros(batch_size, self.chunk_size, self.action_dim, dtype=torch.float32)
        elif actions.ndim == 2:
            actions = actions.unsqueeze(0)
        raw_actions = actions.detach().cpu().numpy().astype(np.float32)

        pad = complementary.get("action_is_pad")
        if pad is None:
            action_masks = np.ones_like(raw_actions, dtype=bool)
        else:
            pad_np = pad.detach().cpu().numpy().astype(bool)
            while pad_np.ndim < raw_actions.ndim:
                pad_np = pad_np[..., None]
            action_masks = np.broadcast_to(~pad_np, raw_actions.shape).copy()

        instances: list[dict[str, Any]] = []
        for index in range(batch_size):
            state_np = states[index].detach().cpu().numpy().astype(np.float32)
            if state_np.ndim == 1:
                state_np = state_np[None, :]
            if self.normalize_state:
                if self.state_min is None or self.state_max is None:
                    raise ValueError("normalize_state=True requires state min/max statistics")
                state_np = self._normalize(state_np, self.state_min, self.state_max, state=True)
            action_np = self._normalize(raw_actions[index], self.action_min, self.action_max, state=False)
            images = [self._pt_to_pil(observation[key][index], normalized=False) for key in self.image_keys]
            sample = {
                "observations": images,
                "states": state_np,
                "actions": action_np,
                "raw_actions": raw_actions[index],
                "instruction": str(tasks[index]).lower(),
                "dataset": "humano",
                "actions_mask": action_masks[index],
            }
            instances.append(self._model_transform(sample, vlm_processor=self._vlm_processor))

        pad_token_id = self._vlm_processor.tokenizer.pad_token_id
        input_ids = [
            item["input_ids"].squeeze(0) if item["input_ids"].dim() == 2 else item["input_ids"]
            for item in instances
        ]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        input_ids = input_ids[:, : self.model_max_length]
        pixel_values = torch.stack([item["pixel_values"] for item in instances]).to(torch.float32)
        image_grid_thw = torch.stack([item["image_grid_thw"].squeeze(0) for item in instances])
        normalized_actions = torch.stack(
            [torch.from_numpy(np.asarray(item["actions"], dtype=np.float32)) for item in instances]
        )
        normalized_states = torch.stack(
            [torch.from_numpy(np.asarray(item["states"], dtype=np.float32)) for item in instances]
        )
        masks = torch.stack(
            [torch.from_numpy(np.asarray(item["actions_mask"], dtype=bool)) for item in instances]
        )

        observation[self.state_key] = normalized_states
        transition[TransitionKey.OBSERVATION] = observation
        if transition.get(TransitionKey.ACTION) is not None:
            transition[TransitionKey.ACTION] = normalized_actions
        complementary.update(
            input_ids=input_ids,
            attention_mask=input_ids.ne(pad_token_id),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            actions_mask=masks,
            raw_actions=torch.from_numpy(raw_actions),
        )
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "psi0_source_dir": self.psi0_source_dir,
            "model_name_or_path": self.model_name_or_path,
            "image_keys": self.image_keys,
            "state_key": self.state_key,
            "task_key": self.task_key,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "model_max_length": self.model_max_length,
            "action_min": self.action_min,
            "action_max": self.action_max,
            "state_min": self.state_min,
            "state_max": self.state_max,
            "normalize_state": self.normalize_state,
            "image_resize": self.image_resize,
            "image_center_crop": self.image_center_crop,
            "image_augmentation": self.image_augmentation,
        }


@dataclass
@ProcessorStepRegistry.register(name="psi0_denormalize_action")
class Psi0DenormalizeActionStep(ProcessorStep):
    action_min: list[float]
    action_max: list[float]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        low = torch.as_tensor(self.action_min, device=action.device, dtype=action.dtype)
        high = torch.as_tensor(self.action_max, device=action.device, dtype=action.dtype)
        transition[TransitionKey.ACTION] = 0.5 * (action + 1.0) * (high - low) + low
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"action_min": self.action_min, "action_max": self.action_max}


def make_psi0_pre_post_processors(
    config: Psi0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    action_min, action_max, state_min, state_max = _resolve_stats(config, dataset_stats)
    preprocess = Psi0BatchProcessorStep(
        psi0_source_dir=config.psi0_source_dir,
        model_name_or_path=config.model_name_or_path,
        image_keys=config.image_keys,
        state_key=config.state_key,
        task_key=config.task_key,
        action_dim=config.action_dim,
        chunk_size=config.chunk_size,
        model_max_length=config.model_max_length,
        action_min=action_min,
        action_max=action_max,
        state_min=state_min,
        state_max=state_max,
        normalize_state=config.normalize_state,
        image_resize=config.image_resize,
        image_center_crop=config.image_center_crop,
        image_augmentation=config.image_augmentation,
    )
    return (
        PolicyProcessorPipeline(
            steps=[AddBatchDimensionProcessorStep(), preprocess, DeviceProcessorStep(device=config.device)],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline(
            steps=[
                Psi0DenormalizeActionStep(action_min=action_min, action_max=action_max),
                DeviceProcessorStep(device="cpu"),
            ],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
