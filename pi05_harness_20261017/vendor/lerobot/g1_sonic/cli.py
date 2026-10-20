"""Command-line entrypoints for the independent G1 SONIC baseline project."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .dataset_inspection import inspect_v21_dataset
from .evaluation import evaluate_checkpoint
from .export import UPSTREAM_COMMIT, export_runtime
from .prepare import prepare_dataset, prepare_dataset_collection
from .training import (
    ACTTrainingConfig,
    DiffusionTrainingConfig,
    FlowMatchingTrainingConfig,
    PI05TrainingConfig,
    validate_prepared_dataset,
)


def inspect_main() -> None:
    parser = argparse.ArgumentParser(description="Validate a HUMANO G1 SONIC LeRobot v2.1 dataset")
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asdict(inspect_v21_dataset(args.source)), indent=2))


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare non-destructive v3.0 data for G1 SONIC ACT")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect_v21_dataset(args.source)
    prepare_dataset(args.source, args.output)


def prepare_collection_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and aggregate a G1 SONIC dataset collection")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_dataset_collection(args.source_root, args.output)


def train_act_main() -> None:
    parser = argparse.ArgumentParser(description="Train official LeRobot ACT on prepared G1 SONIC data")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--save-freq", type=int, default=2_000)
    args = parser.parse_args()
    config = ACTTrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.steps,
        save_freq=args.save_freq,
    )
    validate_prepared_dataset(config.dataset_root)
    subprocess.run(
        _make_training_command(config.command_args(), args.num_processes),
        check=True,
    )


def train_diffusion_main() -> None:
    parser = argparse.ArgumentParser(description="Train official LeRobot Diffusion on G1 SONIC data")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--save-freq", type=int, default=1_000)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--tracking-project", default=os.environ.get("TRACKING_PROJECT", "g1-sonic-diffusion"))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
        default=os.environ.get("SWANLAB_MODE", "online"),
    )
    args = parser.parse_args()
    config = DiffusionTrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.steps,
        save_freq=args.save_freq,
        save_total_limit=args.save_total_limit,
        log_freq=args.log_freq,
        tracking_project=args.tracking_project,
        wandb_mode=args.wandb_mode,
        swanlab_mode=args.swanlab_mode,
    )
    validate_prepared_dataset(config.dataset_root)
    subprocess.run(
        _make_training_command(config.command_args(), args.num_processes),
        check=True,
    )


def train_flow_matching_main() -> None:
    parser = argparse.ArgumentParser(description="Train official LeRobot Flow Matching on G1 SONIC data")
    parser.add_argument("--dataset-root", type=Path, required=True, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--save-freq", type=int, default=1_000)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--num-integration-steps", type=int, default=100)
    parser.add_argument(
        "--tracking-project", default=os.environ.get("TRACKING_PROJECT", "g1-sonic-flow-matching")
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
        default=os.environ.get("SWANLAB_MODE", "online"),
    )
    args = parser.parse_args()
    config = FlowMatchingTrainingConfig(
        dataset_roots=tuple(args.dataset_root),
        output_dir=args.output_dir,
        clip_model=args.clip_model,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.steps,
        save_freq=args.save_freq,
        save_total_limit=args.save_total_limit,
        log_freq=args.log_freq,
        num_integration_steps=args.num_integration_steps,
        tracking_project=args.tracking_project,
        wandb_mode=args.wandb_mode,
        swanlab_mode=args.swanlab_mode,
    )
    dataset_root = config.single_dataset_root()
    validate_prepared_dataset(dataset_root)
    subprocess.run(
        _make_training_command(config.command_args(), args.num_processes),
        check=True,
    )


def _make_training_command(command_args: tuple[str, ...], num_processes: int) -> tuple[str, ...]:
    if num_processes <= 0:
        raise ValueError(f"num_processes must be positive, got {num_processes}")
    if num_processes == 1:
        return (sys.executable, "-m", "lerobot.scripts.lerobot_train", *command_args)
    return (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_processes}",
        "--module",
        "lerobot.scripts.lerobot_train",
        *command_args,
    )


def train_pi05_main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune official LeRobot Pi0.5 on G1 SONIC data")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="lerobot/pi05_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--save-freq", type=int, default=1_000)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--tracking-project", default=os.environ.get("TRACKING_PROJECT", "g1-sonic-pi05"))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local", "offline", "disabled"),
        default=os.environ.get("SWANLAB_MODE", "online"),
    )
    args = parser.parse_args()
    config = PI05TrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        base_model=args.base_model,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.steps,
        save_freq=args.save_freq,
        save_total_limit=args.save_total_limit,
        log_freq=args.log_freq,
        tracking_project=args.tracking_project,
        wandb_mode=args.wandb_mode,
        swanlab_mode=args.swanlab_mode,
    )
    validate_prepared_dataset(config.dataset_root)
    subprocess.run(
        _make_training_command(config.command_args(), args.num_processes),
        check=True,
    )


def evaluate_act_main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-evaluate a G1 SONIC ACT checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = evaluate_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        dataset_root=args.dataset_root,
        device=args.device,
        expected_policy_type="act",
    )
    print(json.dumps(report, indent=2))


def _evaluate_main(policy_type: str) -> None:
    parser = argparse.ArgumentParser(description=f"Smoke-evaluate a G1 SONIC {policy_type} checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="move forward to the office chair and push it to the table")
    args = parser.parse_args()
    report = evaluate_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        dataset_root=args.dataset_root,
        device=args.device,
        expected_policy_type=policy_type,
        task=args.task,
    )
    print(json.dumps(report, indent=2))


def evaluate_diffusion_main() -> None:
    _evaluate_main("diffusion")


def evaluate_flow_matching_main() -> None:
    _evaluate_main("multi_task_dit")


def evaluate_pi05_main() -> None:
    _evaluate_main("pi05")


def export_main() -> None:
    parser = argparse.ArgumentParser(description="Export a minimal G1 SONIC inference closure")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-commit", default=UPSTREAM_COMMIT)
    args = parser.parse_args()
    export_runtime(
        args.checkpoint_dir,
        args.output_dir,
        upstream_commit=args.upstream_commit,
    )
