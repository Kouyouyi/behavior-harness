"""Training configurations for G1 SONIC policies in official LeRobot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import G1SonicActionSpec, G1SonicObservationSpec

PI05_BASE_MODEL = "lerobot/pi05_base"
CLIP_BASE_MODEL = "openai/clip-vit-base-patch16"


def validate_prepared_dataset(root: Path) -> None:
    """Validate the shared GR00T-aligned contract before launching training."""

    root = Path(root).resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"prepared dataset metadata does not exist: {info_path}")

    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"prepared dataset must use codebase_version 'v3.0', got {info.get('codebase_version')!r}"
        )
    if info.get("robot_type") != "g1_sonic":
        raise ValueError(f"prepared dataset must use robot_type 'g1_sonic', got {info.get('robot_type')!r}")

    action = info.get("features", {}).get("action")
    state = info.get("features", {}).get("observation.state")
    expected_state_dim = G1SonicObservationSpec().total_dim
    expected_dim = G1SonicActionSpec().total_dim
    if action is None:
        raise ValueError("prepared dataset metadata is missing the explicit action feature")
    if action.get("dtype") != "float32" or action.get("shape") != [expected_dim]:
        raise ValueError(
            "prepared dataset action feature must have dtype float32 and shape "
            f"[{expected_dim}], got {action!r}"
        )
    if state is None:
        raise ValueError("prepared dataset metadata is missing the GR00T-aligned observation.state feature")
    if state.get("dtype") != "float32" or state.get("shape") != [expected_state_dim]:
        raise ValueError(
            "prepared dataset observation.state feature must have dtype float32 and shape "
            f"[{expected_state_dim}], got {state!r}"
        )

    stats_path = root / "meta" / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"prepared dataset stats do not exist: {stats_path}")
    stats = json.loads(stats_path.read_text())
    required_stats = {
        "action": (expected_dim,),
        "observation.images.ego_view": (3, 1, 1),
        "observation.state": (expected_state_dim,),
    }
    for feature, expected_shape in required_stats.items():
        feature_stats = stats.get(feature)
        if feature_stats is None:
            raise ValueError(f"prepared dataset stats are missing {feature!r}")
        stat_names = ("mean", "std")
        if feature in {"action", "observation.state"}:
            stat_names += ("q01", "q99")
        for stat_name in stat_names:
            values = np.asarray(feature_stats.get(stat_name), dtype=np.float32)
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise ValueError(
                    f"prepared dataset {feature!r} {stat_name} must be finite with "
                    f"shape {expected_shape}, got {values.shape}"
                )

    parquet_paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise ValueError(f"no official v3 data parquet files found under {root / 'data'}")
    for parquet_path in parquet_paths:
        table = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
        if table.empty:
            raise ValueError(f"prepared dataset parquet is empty: {parquet_path}")
        actions = np.stack(table["action"].to_numpy()).astype(np.float32, copy=False)
        if actions.ndim != 2 or actions.shape[1] != expected_dim:
            raise ValueError(
                f"{parquet_path} action values must have shape (N, {expected_dim}), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"{parquet_path} action values must contain only finite values")
        states = np.stack(table["observation.state"].to_numpy()).astype(np.float32, copy=False)
        if states.ndim != 2 or states.shape[1] != expected_state_dim:
            raise ValueError(
                f"{parquet_path} observation.state values must have shape "
                f"(N, {expected_state_dim}), got {states.shape}"
            )
        if not np.isfinite(states).all():
            raise ValueError(f"{parquet_path} observation.state values must contain only finite values")


@dataclass(frozen=True)
class ACTTrainingConfig:
    dataset_root: Path
    output_dir: Path
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 4
    steps: int = 30_000
    save_freq: int = 2_000
    horizon: int = 40
    job_name: str = "g1_sonic_act"

    def command_args(self) -> tuple[str, ...]:
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        return (
            f"--dataset.repo_id=g1_sonic/{self.dataset_root.name}",
            f"--dataset.root={self.dataset_root.resolve()}",
            "--policy.type=act",
            f"--policy.device={self.device}",
            f"--policy.chunk_size={self.horizon}",
            f"--policy.n_action_steps={self.horizon}",
            "--policy.push_to_hub=false",
            f"--output_dir={self.output_dir.resolve()}",
            f"--job_name={self.job_name}",
            f"--batch_size={self.batch_size}",
            f"--num_workers={self.num_workers}",
            f"--steps={self.steps}",
            f"--save_freq={self.save_freq}",
            "--wandb.enable=false",
        )


@dataclass(frozen=True)
class DiffusionTrainingConfig:
    dataset_root: Path
    output_dir: Path
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 16
    steps: int = 30_000
    save_freq: int = 1_000
    save_total_limit: int = 5
    log_freq: int = 100
    horizon: int = 40
    num_inference_steps: int = 10
    job_name: str = "g1_sonic_diffusion"
    tracking_project: str = "g1-sonic-diffusion"
    wandb_mode: str = "online"
    swanlab_mode: str = "online"

    def command_args(self) -> tuple[str, ...]:
        if self.horizon <= 0 or self.horizon % 8:
            raise ValueError(f"diffusion horizon must be a positive multiple of 8, got {self.horizon}")
        return (
            f"--dataset.repo_id=g1_sonic/{self.dataset_root.name}",
            f"--dataset.root={self.dataset_root.resolve()}",
            "--policy.type=diffusion",
            f"--policy.device={self.device}",
            "--policy.n_obs_steps=1",
            f"--policy.horizon={self.horizon}",
            f"--policy.n_action_steps={self.horizon}",
            "--policy.drop_n_last_frames=0",
            f"--policy.num_inference_steps={self.num_inference_steps}",
            "--policy.push_to_hub=false",
            f"--output_dir={self.output_dir.resolve()}",
            f"--job_name={self.job_name}",
            f"--batch_size={self.batch_size}",
            f"--num_workers={self.num_workers}",
            f"--steps={self.steps}",
            f"--save_freq={self.save_freq}",
            f"--save_total_limit={self.save_total_limit}",
            f"--log_freq={self.log_freq}",
            "--wandb.enable=true",
            "--wandb.disable_artifact=true",
            f"--wandb.project={self.tracking_project}",
            f"--wandb.mode={self.wandb_mode}",
            "--swanlab.enable=true",
            f"--swanlab.project={self.tracking_project}",
            f"--swanlab.mode={self.swanlab_mode}",
        )


@dataclass(frozen=True)
class FlowMatchingTrainingConfig:
    dataset_roots: tuple[Path, ...]
    output_dir: Path
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 16
    steps: int = 30_000
    save_freq: int = 1_000
    save_total_limit: int = 5
    log_freq: int = 100
    horizon: int = 40
    clip_model: str = CLIP_BASE_MODEL
    num_integration_steps: int = 100
    image_target_shape: tuple[int, int] = (224, 224)
    shortest_image_edge: int = 224
    crop_fraction: float = 0.95
    job_name: str = "g1_sonic_flow_matching"
    tracking_project: str = "g1-sonic-flow-matching"
    wandb_mode: str = "online"
    swanlab_mode: str = "online"

    def single_dataset_root(self) -> Path:
        if not self.dataset_roots:
            raise ValueError("single-task Flow Matching requires one dataset root")
        if len(self.dataset_roots) > 1:
            raise NotImplementedError(
                "multi-task Flow Matching dataset loading is reserved but not implemented; "
                "pass exactly one --dataset-root"
            )
        return self.dataset_roots[0]

    def command_args(self) -> tuple[str, ...]:
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.num_integration_steps <= 0:
            raise ValueError(f"num_integration_steps must be positive, got {self.num_integration_steps}")
        dataset_root = self.single_dataset_root()
        return (
            f"--dataset.repo_id=g1_sonic/{dataset_root.name}",
            f"--dataset.root={dataset_root.resolve()}",
            "--policy.type=multi_task_dit",
            "--policy.objective=flow_matching",
            f"--policy.device={self.device}",
            "--policy.n_obs_steps=1",
            f"--policy.horizon={self.horizon}",
            f"--policy.n_action_steps={self.horizon}",
            "--policy.drop_n_last_frames=0",
            "--policy.do_mask_loss_for_padding=true",
            "--policy.timestep_sampling_strategy=beta",
            "--policy.timestep_sampling_alpha=1.5",
            "--policy.timestep_sampling_beta=1.0",
            "--policy.timestep_sampling_s=0.999",
            f"--policy.num_integration_steps={self.num_integration_steps}",
            "--policy.integration_method=euler",
            f"--policy.vision_encoder_name={self.clip_model}",
            f"--policy.text_encoder_name={self.clip_model}",
            f"--policy.image_resize_shape={list(self.image_target_shape)}",
            f"--policy.shortest_image_edge={self.shortest_image_edge}",
            f"--policy.crop_fraction={self.crop_fraction}",
            "--policy.image_crop_is_random=true",
            "--policy.push_to_hub=false",
            f"--output_dir={self.output_dir.resolve()}",
            f"--job_name={self.job_name}",
            f"--batch_size={self.batch_size}",
            f"--num_workers={self.num_workers}",
            f"--steps={self.steps}",
            f"--save_freq={self.save_freq}",
            f"--save_total_limit={self.save_total_limit}",
            f"--log_freq={self.log_freq}",
            "--wandb.enable=true",
            "--wandb.disable_artifact=true",
            f"--wandb.project={self.tracking_project}",
            f"--wandb.mode={self.wandb_mode}",
            "--swanlab.enable=true",
            f"--swanlab.project={self.tracking_project}",
            f"--swanlab.mode={self.swanlab_mode}",
        )


@dataclass(frozen=True)
class PI05TrainingConfig:
    dataset_root: Path
    output_dir: Path
    device: str = "cuda"
    batch_size: int = 64
    num_workers: int = 16
    steps: int = 30_000
    save_freq: int = 1_000
    save_total_limit: int = 5
    log_freq: int = 100
    horizon: int = 40
    base_model: str = PI05_BASE_MODEL
    job_name: str = "g1_sonic_pi05"
    tracking_project: str = "g1-sonic-pi05"
    wandb_mode: str = "online"
    swanlab_mode: str = "online"

    def command_args(self) -> tuple[str, ...]:
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        return (
            f"--dataset.repo_id=g1_sonic/{self.dataset_root.name}",
            f"--dataset.root={self.dataset_root.resolve()}",
            f"--policy.path={self.base_model}",
            "--policy.input_features=null",
            f"--policy.device={self.device}",
            "--policy.dtype=bfloat16",
            f"--policy.chunk_size={self.horizon}",
            f"--policy.n_action_steps={self.horizon}",
            "--policy.optimizer_lr=0.001",
            "--policy.scheduler_decay_lr=0.00001",
            f"--policy.max_state_dim={G1SonicObservationSpec().total_dim}",
            f"--policy.max_action_dim={G1SonicActionSpec().total_dim}",
            "--policy.train_expert_only=true",
            "--policy.gradient_checkpointing=true",
            "--policy.push_to_hub=false",
            f"--output_dir={self.output_dir.resolve()}",
            f"--job_name={self.job_name}",
            f"--batch_size={self.batch_size}",
            f"--num_workers={self.num_workers}",
            f"--steps={self.steps}",
            f"--save_freq={self.save_freq}",
            f"--save_total_limit={self.save_total_limit}",
            f"--log_freq={self.log_freq}",
            "--wandb.enable=true",
            "--wandb.disable_artifact=true",
            f"--wandb.project={self.tracking_project}",
            f"--wandb.mode={self.wandb_mode}",
            "--swanlab.enable=true",
            f"--swanlab.project={self.tracking_project}",
            f"--swanlab.mode={self.swanlab_mode}",
        )
