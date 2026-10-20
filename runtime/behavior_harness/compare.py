"""Audit a matched custom-OOD baseline/assisted pair; never infer missing success.

CLI exit codes: 0 = this pair demonstrates recovery, 1 = evidence does not
demonstrate recovery, 2 = required input files cannot be read/parsed. The JSON
report is emitted in every case. Video checks inspect saved metadata and file
presence, not decoding. A passing pair is not an official benchmark replication
or a success-rate estimate across tasks/seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


_RADIO_CHECKPOINT = "/root/download/models/pi05-BEHAVIOR-turning-on-radio/pi05_turn_on_the_radio"
_POSITION_TOLERANCE = 0.002
_ANGLE_TOLERANCE = 0.001
_FILES = ("config", "setup", "result", "initial_robot_state")
_CONFIG_KEYS = ("task_name", "instance_id", "seed", "mode", "max_steps", "head_mode", "ood")
_LANGUAGE_ASSISTANCE = "visual_localization_and_language_context"
_ROOM_FACING_OOD = "radio_on_tv_cabinet_facing_room"
_MATCHED_OPTIONAL_FLAGS = ("manipulation_proprio_enabled",)


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _vector(value, length):
    return isinstance(value, list) and len(value) == length and all(_finite(item) for item in value)


def _distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _read_episode(directory: Path, label: str, errors: list[str]):
    episode = {"directory": str(directory)}
    for name in _FILES:
        path = directory / f"{name}.json"
        try:
            def reject_constant(token):
                raise ValueError(f"nonfinite JSON constant {token}")

            value = json.loads(path.read_text(), parse_constant=reject_constant)
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            episode[name] = value
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: cannot read {name}.json ({type(exc).__name__}: {exc})")
            episode[name] = {}
    return episode


def _model(config):
    task = config.get("task_name")
    return {
        "policy_kind": config.get("policy_kind", "websocket"),
        "policy_checkpoint": config.get("policy_checkpoint", _RADIO_CHECKPOINT if task == "turning_on_radio" else None),
        "policy_config": config.get("policy_config", "pi05_b1k"),
        "policy_repo_id": config.get("policy_repo_id", task),
        "policy_robot": config.get("policy_robot", "b1k/R1Pro"),
        "policy_task": config.get("policy_task", f"b1k/{task}" if task else None),
        # These are fixed in launch.py, not per-episode planner choices.
        "control_mode": "receding_horizon",
        "action_horizon": 16,
    }


def _assistance_type(config):
    if config.get("language_context_enabled", False) is True:
        return _LANGUAGE_ASSISTANCE
    return {"baseline": "none", "assisted": "visual_localization"}.get(config.get("run_mode"), "unknown")


def _context_conditions(episode, label):
    config, result = episode["config"], episode["result"]
    enabled, recorded = config.get("language_context_enabled", False), result.get("language_context_enabled", False)
    reasons = []
    if type(enabled) is not bool or type(recorded) is not bool:
        reasons.append(f"{label}: language_context_enabled must be a boolean")
    elif enabled != recorded:
        reasons.append(f"{label}: result.language_context_enabled does not match config")
    if label == "baseline" and (enabled is not False or recorded is not False):
        reasons.append("baseline: language context invalidates a VLA-only baseline")
    assistance = result.get("assistance_type")
    if (assistance is not None or enabled is True) and assistance != _assistance_type(config):
        reasons.append(f"{label}: result.assistance_type does not match the configured intervention")
    count = result.get("context_handoffs", 0)
    if not _integer(count):
        reasons.append(f"{label}: context_handoffs must be a nonnegative integer")
    if enabled is not True and (count != 0 or result.get("last_visual_context") is not None):
        reasons.append(f"{label}: disabled language context has recorded context bindings")
    return reasons


def _read_jsonl(path):
    def reject_constant(token):
        raise ValueError(f"nonfinite JSON constant {token}")

    rows = [json.loads(line, parse_constant=reject_constant) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("audit rows must be JSON objects")
    return rows


def _context_evidence(episode, label):
    """A feature flag or a bound handoff alone never proves language was used."""
    from .context_policy_service import validate_context

    directory = Path(episode["directory"])
    enabled = episode["config"].get("language_context_enabled", False) is True
    audit_path = directory / "context_policy_audit.jsonl"
    summary = {"language_context_enabled": enabled, "assistance_type": _assistance_type(episode["config"]),
               "verified_context_inferences": 0, "language_context_applied": False}
    reasons = []
    if not enabled and not audit_path.exists():
        return summary, reasons
    try:
        policy_rows = _read_jsonl(audit_path)
        if not enabled:
            if any(row.get("context") is not None or row.get("language_context_active") is True for row in policy_rows):
                reasons.append(f"{label}: policy audit records language context while configuration disables it")
            return summary, reasons
        backend_rows = _read_jsonl(directory / "backend_audit.jsonl")
        handoffs = [row for row in backend_rows if row.get("event") == "handoff"]
        bindings = {}
        last_source = None
        for version, row in enumerate(handoffs, 1):
            context = validate_context(row.get("visual_context"))
            source_id, step = context["source_observation_id"], context["source_step"]
            source_order = (step, int(source_id.rsplit("-", 1)[1]))
            if (context["version"] != version or context["context_id"] in bindings
                    or (last_source is not None and source_order <= last_source)
                    or row.get("cache_reset") is not True or row.get("phase") != "vla"
                    or not _integer(row.get("step")) or row["step"] != step
                    or not _integer(episode["result"].get("steps"), step)):
                raise ValueError("context handoff version/source/reset evidence is inconsistent")
            observation_dir = directory / "observations" / source_id
            observation = json.loads((observation_dir / "observation.json").read_text())
            if (observation.get("observation_id") != source_id or observation.get("phase") != "gpt"
                    or not _integer(observation.get("step")) or observation["step"] != step):
                raise ValueError("context source observation does not match its saved ID/step")
            for role in ("head", "left_wrist", "right_wrist"):
                camera = observation.get("cameras", {}).get(role, {})
                image = Path(camera.get("path", ""))
                if (camera.get("source") != "robot_sensor" or camera.get("color_space") != "RGB"
                        or not _integer(camera.get("step")) or camera["step"] != step
                        or image.resolve().parent != observation_dir.resolve()
                        or not image.is_file() or image.stat().st_size == 0):
                    raise ValueError("context source lacks current saved robot RGB evidence")
            bindings[context["context_id"]] = context
            last_source = source_order
        count = episode["result"].get("context_handoffs")
        if not _integer(count, 1) or count != len(bindings):
            raise ValueError("result.context_handoffs does not match completed backend handoffs")
        if episode["result"].get("last_visual_context") != list(bindings.values())[-1]:
            raise ValueError("result.last_visual_context does not match the final handoff")
        for row in policy_rows:
            if row.get("event") != "infer_completed" or row.get("first_context_infer") is not True:
                continue
            context = validate_context(row.get("context"))
            prompt = row.get("original_prompt")
            if (bindings.get(context["context_id"]) != context or row.get("language_context_active") is not True
                    or row.get("assistance_type") != _LANGUAGE_ASSISTANCE
                    or not _integer(row.get("token_count"), 1) or row["token_count"] > 200
                    or not isinstance(prompt, str) or not prompt.strip()
                    or row.get("effective_prompt") != prompt + "\nCurrent visual context: " + context["text"]):
                raise ValueError("completed policy context inference does not match its handoff/prompt/token evidence")
            summary["verified_context_inferences"] += 1
        summary["language_context_applied"] = summary["verified_context_inferences"] > 0
        if not summary["language_context_applied"]:
            reasons.append(f"{label}: language context has no verified completed inference; enabling/binding is insufficient")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        reasons.append(f"{label}: invalid or missing language-context evidence ({type(exc).__name__}: {exc})")
        summary["language_context_applied"] = False
        summary["verified_context_inferences"] = 0
    return summary, reasons


def _compare_vector(left, right, key, length, tolerance, reasons, *, euclidean=False):
    a, b = left.get(key), right.get(key)
    if not _vector(a, length) or not _vector(b, length):
        reasons.append(f"initial conditions: {key} requires {length} finite values in both episodes")
        return
    difference = _distance(a, b) if euclidean else max(abs(x - y) for x, y in zip(a, b))
    if difference > tolerance + 1e-12:
        reasons.append(f"initial conditions: {key} differs by {difference:.8g}, tolerance {tolerance}")


def _compare_quaternion(left, right, key, reasons):
    a, b = left.get(key), right.get(key)
    if not _vector(a, 4) or not _vector(b, 4):
        reasons.append(f"initial conditions: {key} requires a finite xyzw quaternion in both episodes")
        return
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if abs(na - 1.0) > 0.001 or abs(nb - 1.0) > 0.001:
        reasons.append(f"initial conditions: {key} is not a unit quaternion")
        return
    # q and -q denote the same orientation. Compare angular distance in radians.
    dot = min(1.0, abs(sum(x * y for x, y in zip(a, b)) / (na * nb)))
    angle = 2 * math.acos(dot)
    if angle > _ANGLE_TOLERANCE + 1e-12:
        reasons.append(f"initial conditions: {key} differs by {angle:.8g} rad, tolerance {_ANGLE_TOLERANCE}")


def _condition_reasons(baseline, assisted):
    reasons = []
    configs = [episode["config"] for episode in (baseline, assisted)]
    for label, episode in (("baseline", baseline), ("assisted", assisted)):
        config, setup, result = episode["config"], episode["setup"], episode["result"]
        reasons.extend(_context_conditions(episode, label))
        for key in _MATCHED_OPTIONAL_FLAGS:
            if type(config.get(key, False)) is not bool:
                reasons.append(f"{label}: config.{key} must be a boolean")
        for key in _CONFIG_KEYS:
            value = config.get(key)
            valid = (_integer(value, 1 if key == "max_steps" else 0)
                     if key in {"instance_id", "seed", "max_steps"}
                     else isinstance(value, str) and bool(value))
            if not valid:
                reasons.append(f"{label}: missing or invalid config.{key}")
        if config.get("run_mode") != label or result.get("run_type") != label:
            reasons.append(f"{label}: config.run_mode and result.run_type must both be {label!r}")
        for config_key, result_key in (("task_name", "task"), ("instance_id", "instance_id"),
                                       ("max_steps", "max_steps"), ("head_mode", "head_mode"), ("ood", "ood")):
            if result_key not in result or result.get(result_key) != config.get(config_key):
                reasons.append(f"{label}: result.{result_key} does not match its config")
        if not isinstance(config.get("ood"), str) or config.get("ood") == "none" or setup.get("ood") != config.get("ood"):
            reasons.append(f"{label}: this audit requires a recorded custom OOD fixture matching config.ood")
        for key, expected in (("task_goal_modified", False), ("privileged_setup_only", True),
                              ("original_benchmark_initial_distribution", False),
                              ("settled_on_support", True), ("initial_toggled_on", False)):
            if setup.get(key) is not expected:
                reasons.append(f"{label}: setup.{key} must be {expected!r}")
        speed = setup.get("linear_speed_mps")
        if not _finite(speed) or not 0 <= speed <= 0.05:
            reasons.append(f"{label}: fixture settling was not verified (linear_speed_mps)")
        if config.get("ood") == _ROOM_FACING_OOD:
            for key in ("settled_facing_room", "settled_stable"):
                if setup.get(key) is not True:
                    reasons.append(f"{label}: setup.{key} must be True for the room-facing fixture")
            angular_speed = setup.get("angular_speed_rps")
            if not _finite(angular_speed) or not 0 <= angular_speed <= 0.1:
                reasons.append(f"{label}: fixture settling was not verified (angular_speed_rps)")
            # The legacy field is the requested placement pose in this mode.
            _compare_quaternion(setup, {"radio_orientation_xyzw": setup.get("requested_radio_orientation_xyzw")},
                                "radio_orientation_xyzw", reasons)
        for key, value in _model(config).items():
            if key == "action_horizon":
                continue
            if not isinstance(value, str) or not value:
                reasons.append(f"{label}: missing or invalid model setting {key}")
        if _model(config)["policy_kind"] != "websocket":
            reasons.append(f"{label}: local smoke-test policy cannot count as VLA evidence")
    for key in _CONFIG_KEYS:
        if configs[0].get(key) != configs[1].get(key):
            reasons.append(f"conditions differ: config.{key}")
    for key in _MATCHED_OPTIONAL_FLAGS:
        if configs[0].get(key, False) != configs[1].get(key, False):
            reasons.append(f"conditions differ: config.{key}")
    for key, value in _model(configs[0]).items():
        if value != _model(configs[1])[key]:
            reasons.append(f"conditions differ: model.{key}")
    for key in ("radio_name", "support_name"):
        a, b = baseline["setup"].get(key), assisted["setup"].get(key)
        if not isinstance(a, str) or not a or a != b:
            reasons.append(f"initial conditions: setup.{key} is missing or different")
    for key in ("requested_radio_position_world", "settled_radio_position_world"):
        _compare_vector(baseline["setup"], assisted["setup"], key, 3, _POSITION_TOLERANCE, reasons, euclidean=True)
    _compare_quaternion(baseline["setup"], assisted["setup"], "radio_orientation_xyzw", reasons)
    if any(config.get("ood") == _ROOM_FACING_OOD for config in configs):
        for key in ("requested_radio_orientation_xyzw", "settled_radio_orientation_xyzw"):
            _compare_quaternion(baseline["setup"], assisted["setup"], key, reasons)
    robots = [episode["initial_robot_state"] for episode in (baseline, assisted)]
    _compare_vector(*robots, "base_position", 3, _POSITION_TOLERANCE, reasons, euclidean=True)
    _compare_quaternion(*robots, "base_orientation_xyzw", reasons)
    _compare_vector(*robots, "trunk_qpos", 4, _ANGLE_TOLERANCE, reasons)
    _compare_vector(*robots, "head_gimbal", 2, _ANGLE_TOLERANCE, reasons)
    yaw_a, yaw_b = robots[0].get("base_yaw_rad"), robots[1].get("base_yaw_rad")
    if not _finite(yaw_a) or not _finite(yaw_b):
        reasons.append("initial conditions: base_yaw_rad is missing or nonfinite")
    elif abs(math.atan2(math.sin(yaw_a - yaw_b), math.cos(yaw_a - yaw_b))) > _ANGLE_TOLERANCE + 1e-12:
        reasons.append("initial conditions: base_yaw_rad differs beyond 0.001 rad")
    return reasons


def _outcome_reasons(episode, label):
    result = episode["result"]
    reasons = []
    success = result.get("success")
    if success is not (label == "assisted"):
        expected = "True" if label == "assisted" else "False"
        reasons.append(f"{label}: success must be the explicit boolean {expected}, observed {success!r}")
    if result.get("success_source") != "BehaviorTask.success":
        reasons.append(f"{label}: success_source is not BehaviorTask.success")
    if result.get("done") is not True:
        reasons.append(f"{label}: episode is not closed/terminal")
    steps, maximum = result.get("steps"), result.get("max_steps")
    valid_steps = _integer(steps, 1) and _integer(maximum, 1) and steps <= maximum
    if not valid_steps:
        reasons.append(f"{label}: invalid or missing actual steps/max_steps")
    if label == "baseline":
        if result.get("complete_environment_episode") is not True and result.get("host_budget_exhausted") is not True:
            reasons.append("baseline: incomplete rollout cannot establish VLA failure")
        if result.get("host_budget_exhausted") is True and (not valid_steps or steps != maximum):
            reasons.append("baseline: host_budget_exhausted contradicts actual step count")
    gpt_steps, vla_steps = result.get("gpt_control_steps"), result.get("vla_steps")
    if not _integer(gpt_steps) or not _integer(vla_steps) or not valid_steps or gpt_steps + vla_steps != steps:
        reasons.append(f"{label}: control-source step accounting is missing or inconsistent")
    elif label == "baseline" and gpt_steps != 0:
        reasons.append("baseline: GPT control invalidates a VLA-only baseline")
    elif label == "assisted" and (gpt_steps == 0 or vla_steps == 0):
        reasons.append("assisted: both active-perception motion and VLA steps are required to demonstrate recovery")
    if result.get("errors") != []:
        reasons.append(f"{label}: result.errors must be present and empty")
    videos = result.get("videos")
    if not isinstance(videos, dict):
        reasons.append(f"{label}: video metadata is missing")
        videos = {}
    for role in ("robot_cameras", "third_person"):
        if role not in videos:
            reasons.append(f"{label}: {role} video is missing")
    for role, video in videos.items():
        if not isinstance(video, dict):
            reasons.append(f"{label}: invalid {role} video metadata")
            continue
        if video.get("finalized") is not True:
            reasons.append(f"{label}: {role} video was not finalized")
        frames = video.get("frames")
        if not _integer(frames, 1) or not valid_steps or frames != steps:
            reasons.append(f"{label}: {role} video frames do not match actual steps")
        if video.get("errors", []) != [] or video.get("error") not in (None, ""):
            reasons.append(f"{label}: {role} video reports an error")
        path_value = video.get("path")
        if not isinstance(path_value, str) or not path_value:
            reasons.append(f"{label}: {role} video path is missing")
            continue
        path = Path(path_value)
        if not path.is_absolute():
            path = Path(episode["directory"]) / path
        try:
            if not path.is_file() or path.stat().st_size == 0:
                reasons.append(f"{label}: {role} video file is missing or empty")
        except OSError as exc:
            reasons.append(f"{label}: cannot inspect {role} video file ({type(exc).__name__})")
    return reasons


def _summary(episode, context_evidence):
    result = episode["result"]
    keys = ("run_type", "success", "success_source", "steps", "max_steps", "done",
            "termination_reason", "complete_environment_episode", "host_budget_exhausted",
            "gpt_control_steps", "vla_steps", "videos", "errors", "language_context_enabled",
            "assistance_type", "context_handoffs", "last_visual_context")
    config = {key: episode["config"].get(key) for key in _CONFIG_KEYS}
    config.update({key: episode["config"].get(key, False) for key in _MATCHED_OPTIONAL_FLAGS})
    return {"directory": episode["directory"], "config": config,
            **context_evidence, "effective_model": _model(episode["config"]),
            "result": {key: result.get(key) for key in keys}}


def compare_episodes(baseline_dir: str | Path, assisted_dir: str | Path) -> dict[str, Any]:
    """Read paired artifacts conservatively; incomplete evidence returns reasons."""
    input_errors = []
    baseline = _read_episode(Path(baseline_dir).resolve(), "baseline", input_errors)
    assisted = _read_episode(Path(assisted_dir).resolve(), "assisted", input_errors)
    conditions = _condition_reasons(baseline, assisted)
    outcomes = _outcome_reasons(baseline, "baseline") + _outcome_reasons(assisted, "assisted")
    context_evidence = {}
    for label, episode in (("baseline", baseline), ("assisted", assisted)):
        context_evidence[label], context_reasons = _context_evidence(episode, label)
        (outcomes if episode["config"].get("language_context_enabled", False) is True else conditions).extend(context_reasons)
    matched = not input_errors and not conditions
    demonstrated = matched and not outcomes
    return {
        "schema_version": 1,
        "evaluation_scope": "paired custom OOD; not an official benchmark replication or success-rate estimate",
        "official_benchmark_replication": False,
        "assistance_type": context_evidence["assisted"]["assistance_type"],
        "recovery_claim_scope": ("visual localization plus language context; does not isolate a view-only benefit"
                                 if context_evidence["assisted"]["language_context_enabled"]
                                 else "visual localization only; no language-context intervention"),
        "video_validation": "saved metadata and nonempty file presence only; decoding not verified",
        "tolerances": {"position_m": _POSITION_TOLERANCE, "joint_and_orientation_rad": _ANGLE_TOLERANCE},
        "matched_conditions": matched,
        "demonstrated_recovery": demonstrated,
        "reasons": input_errors + conditions + outcomes,
        "input_errors": input_errors,
        "summaries": {"baseline": _summary(baseline, context_evidence["baseline"]),
                      "assisted": _summary(assisted, context_evidence["assisted"])},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir")
    parser.add_argument("assisted_dir")
    parser.add_argument("--output", help="Write the same JSON report to this path")
    args = parser.parse_args(argv)
    report = compare_episodes(args.baseline_dir, args.assisted_dir)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized)
    print(serialized, end="")
    return 2 if report["input_errors"] else 0 if report["demonstrated_recovery"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
