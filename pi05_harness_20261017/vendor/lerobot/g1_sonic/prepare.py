"""Prepare HUMANO's v2.1 G1 SONIC data for the official LeRobot v3 trainer."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.utils.constants import DEFAULT_FEATURES, IMAGENET_STATS

from .contract import (
    G1SonicActionSpec,
    G1SonicObservationSpec,
    compose_action,
    compose_state,
    validate_dataset_info,
    validate_episode_rows,
)


def action_feature(spec: G1SonicActionSpec = G1SonicActionSpec()) -> dict[str, Any]:
    """Return the v3 feature description for the exported 78D action."""

    names = (
        [f"motion_token_{index}" for index in range(spec.motion_token_dim)]
        + [f"left_hand_{index}" for index in range(spec.left_hand_dim)]
        + [f"right_hand_{index}" for index in range(spec.right_hand_dim)]
    )
    return {"dtype": "float32", "shape": [spec.total_dim], "names": names}


def state_feature(
    info: dict[str, Any],
    spec: G1SonicObservationSpec = G1SonicObservationSpec(),
) -> dict[str, Any]:
    """Return metadata for the GR00T-aligned 46D model state."""

    joint_names = info["features"][spec.state_key].get("names")
    if not isinstance(joint_names, list) or len(joint_names) != spec.joint_state_dim:
        joint_names = [f"joint_state_{index}" for index in range(spec.joint_state_dim)]
    gravity_names = info["features"][spec.projected_gravity_key].get("names")
    if not isinstance(gravity_names, list) or len(gravity_names) != spec.projected_gravity_dim:
        gravity_names = ["gravity_x", "gravity_y", "gravity_z"]
    return {
        "dtype": "float32",
        "shape": [spec.total_dim],
        "names": [*joint_names, *gravity_names],
    }


def add_action_column(table: pd.DataFrame, spec: G1SonicActionSpec = G1SonicActionSpec()) -> pd.DataFrame:
    """Return a copy of a v2.1 frame table with an explicit ``action`` column."""

    actions = [compose_action(row, spec) for row in table.to_dict(orient="records")]
    converted = table.copy()
    converted["action"] = actions
    return converted


def add_model_state_column(
    table: pd.DataFrame,
    spec: G1SonicObservationSpec = G1SonicObservationSpec(),
) -> pd.DataFrame:
    """Return a copy with the 43D joints and 3D projected gravity composed as model state."""

    states = [compose_state(row, spec) for row in table.to_dict(orient="records")]
    converted = table.copy()
    converted[spec.state_key] = states
    return converted


def _vector_stats(
    values: np.ndarray,
    expected_dim: int,
    feature: str,
) -> dict[str, list[float] | list[int]]:
    if values.ndim != 2 or values.shape[1] != expected_dim:
        raise ValueError(f"{feature} must have shape (N, {expected_dim}), got {values.shape}")
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def _action_stats(actions: np.ndarray) -> dict[str, list[float] | list[int]]:
    return _vector_stats(actions, G1SonicActionSpec().total_dim, "actions")


def augment_v30_dataset(root: Path) -> None:
    """Add the GR00T-aligned state, stable action, and normalization statistics."""

    root = Path(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"expected converted v3.0 dataset, got {info.get('codebase_version')!r}")
    validate_dataset_info({**info, "codebase_version": "v2.1"})

    parquet_paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise ValueError(f"no v3 data parquet files found under {root / 'data'}")

    contract_feature_keys = {
        "observation.images.ego_view",
        "observation.state",
        *DEFAULT_FEATURES,
    }
    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    rows_for_validation: list[dict[str, Any]] = []
    for parquet_path in parquet_paths:
        table = pd.read_parquet(parquet_path)
        converted = add_action_column(table)
        converted = add_model_state_column(converted)
        parquet_columns = [column for column in converted.columns if column in contract_feature_keys]
        converted = converted[[*parquet_columns, "action"]]
        converted.to_parquet(parquet_path, index=False)
        all_actions.extend(converted["action"].tolist())
        all_states.extend(converted["observation.state"].tolist())
        rows_for_validation.extend(table[["episode_index", "frame_index", "index"]].to_dict(orient="records"))

    validate_episode_rows(rows_for_validation)
    action_array = np.stack(all_actions).astype(np.float32, copy=False)
    state_array = np.stack(all_states).astype(np.float32, copy=False)
    info_features = {
        key: feature for key, feature in info["features"].items() if key in contract_feature_keys
    }
    info_features["observation.state"] = {
        **state_feature(info),
        "fps": int(info["fps"]),
    }
    info_features["action"] = {**action_feature(), "fps": int(info["fps"])}
    updated_info = {**info, "features": info_features}
    info_path.write_text(json.dumps(updated_info, indent=4) + "\n")

    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats = {key: value for key, value in stats.items() if key in info_features}
    image_stats = stats.get("observation.images.ego_view") or dict(IMAGENET_STATS)
    stats["observation.images.ego_view"] = {
        "min": [[[0.0]]],
        "max": [[[1.0]]],
        **image_stats,
        "count": [int(state_array.shape[0])],
    }
    stats["action"] = _action_stats(action_array)
    stats["observation.state"] = _vector_stats(
        state_array,
        G1SonicObservationSpec().total_dim,
        "observation.state",
    )
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")


def prepare_dataset(source: Path, output: Path) -> None:
    """Convert v2.1 without modifying it, then add the aligned state and action."""

    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    source_info = json.loads((source / "meta" / "info.json").read_text())
    validate_dataset_info(source_info)

    scratch_parent = Path.cwd() / "tmp" / "g1_sonic_act"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="convert_", dir=scratch_parent) as temporary_dir:
        work = Path(temporary_dir) / source.name
        shutil.copytree(source, work)
        from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

        convert_dataset(
            repo_id=source.name,
            root=work,
            push_to_hub=False,
            force_conversion=True,
        )
        augment_v30_dataset(work)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work), str(output))


def discover_dataset_sources(source_root: Path) -> tuple[Path, ...]:
    """Return sorted v2.1 dataset directories immediately under a collection root."""

    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"dataset collection does not exist: {source_root}")
    sources = tuple(sorted(path.parent.parent for path in source_root.glob("*/meta/info.json")))
    if not sources:
        raise ValueError(f"no dataset directories found under {source_root}")
    return sources


def prepare_dataset_collection(source_root: Path, output: Path) -> None:
    """Prepare every source dataset, then aggregate them into one official v3 dataset."""

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    sources = discover_dataset_sources(source_root)
    scratch_parent = Path.cwd() / "tmp" / "g1_sonic_collection"
    scratch_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="prepare_", dir=scratch_parent) as temporary_dir:
        work_root = Path(temporary_dir)
        prepared_roots = tuple(work_root / "sources" / source.name for source in sources)
        for source, prepared_root in zip(sources, prepared_roots, strict=True):
            prepare_dataset(source, prepared_root)

        aggregated_root = work_root / output.name
        from lerobot.datasets.aggregate import aggregate_datasets

        aggregate_datasets(
            repo_ids=[f"g1_sonic/{source.name}" for source in sources],
            roots=list(prepared_roots),
            aggr_repo_id=f"g1_sonic/{output.name}",
            aggr_root=aggregated_root,
            concatenate_videos=False,
            concatenate_data=False,
        )
        from .training import validate_prepared_dataset

        validate_prepared_dataset(aggregated_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(aggregated_root), str(output))
