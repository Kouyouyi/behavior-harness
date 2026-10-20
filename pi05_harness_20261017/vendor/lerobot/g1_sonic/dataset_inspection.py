"""Strict inspection for HUMANO G1 SONIC LeRobot v2.1 datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .contract import compose_action, compose_state, validate_dataset_info, validate_episode_rows


@dataclass(frozen=True)
class DatasetInspectionReport:
    source: str
    episodes: int
    frames: int
    fps: int
    action_dim: int
    video_files: int


def _validate_state(row: dict[str, Any], row_number: int) -> None:
    try:
        compose_state(row)
    except ValueError as exc:
        raise ValueError(f"invalid observation state at row {row_number}: {exc}") from exc


def inspect_v21_dataset(root: Path) -> DatasetInspectionReport:
    """Inspect metadata, numeric frames, episode boundaries and video coverage."""

    root = Path(root).resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing dataset metadata: {info_path}")
    info = json.loads(info_path.read_text())
    manifest = validate_dataset_info(info)

    parquet_paths = sorted((root / "data").glob("*/episode_*.parquet"))
    if not parquet_paths:
        raise ValueError(f"no v2.1 episode parquet files found under {root / 'data'}")

    boundary_rows: list[dict[str, Any]] = []
    episode_indices: set[int] = set()
    total_frames = 0
    for parquet_path in parquet_paths:
        table = pd.read_parquet(parquet_path)
        required = {
            "episode_index",
            "frame_index",
            "index",
            "observation.state",
            "observation.projected_gravity",
            "action.motion_token",
            "teleop.left_hand_joints",
            "teleop.right_hand_joints",
        }
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(f"{parquet_path} is missing required columns: {missing}")

        records = table.to_dict(orient="records")
        for row_number, row in enumerate(records):
            _validate_state(row, total_frames + row_number)
            compose_action(row)
        boundary_rows.extend(table[["episode_index", "frame_index", "index"]].to_dict(orient="records"))
        episode_indices.update(int(value) for value in table["episode_index"].unique())
        total_frames += len(table)

    validate_episode_rows(boundary_rows)
    expected_episodes = int(info.get("total_episodes", -1))
    expected_frames = int(info.get("total_frames", -1))
    if episode_indices != set(range(expected_episodes)):
        raise ValueError(
            f"episode indices must cover 0..{expected_episodes - 1}, got {sorted(episode_indices)}"
        )
    if len(parquet_paths) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} episode parquet files, got {len(parquet_paths)}")
    if total_frames != expected_frames:
        raise ValueError(f"expected {expected_frames} frames, got {total_frames}")

    video_template = info.get("video_path")
    chunk_size = int(info.get("chunks_size", 1000))
    if not isinstance(video_template, str):
        raise ValueError("video_path must be present for the ego-view ACT baseline")
    video_files = 0
    for episode_index in range(expected_episodes):
        video_path = root / video_template.format(
            episode_chunk=episode_index // chunk_size,
            video_key=manifest.image_key,
            episode_index=episode_index,
        )
        if not video_path.is_file():
            raise ValueError(f"missing video for episode {episode_index}: {video_path}")
        video_files += 1

    return DatasetInspectionReport(
        source=str(root),
        episodes=expected_episodes,
        frames=total_frames,
        fps=manifest.fps,
        action_dim=manifest.action_dim,
        video_files=video_files,
    )
