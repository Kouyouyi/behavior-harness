"""Verify finalized rollout MP4s without inspecting their visual contents.

Codec / frame counting and complete CPU decoding are separate checks. Successful
video verification is evidence of recording integrity, never task success.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time


REQUIRED_ROLES = ("robot_cameras", "third_person")


class VideoVerificationError(RuntimeError):
    """Recording-integrity failure with serializable partial audit evidence."""

    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = deepcopy(evidence)


def _fail(evidence, message, *, role=None, stage=None, command=None,
          stderr=None, returncode=None):
    evidence.update(passed=False, status="failed", error=message)
    if role is not None:
        evidence["failed_role"] = role
    if stage is not None:
        evidence["failed_stage"] = stage
    if command is not None:
        evidence["command"] = command
    if stderr is not None:
        evidence["stderr"] = str(stderr)[-16000:]
    if returncode is not None:
        evidence["returncode"] = returncode
    raise VideoVerificationError(message, evidence)


def _run(command, timeout, evidence, role, stage):
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        _fail(evidence, f"{role}: {stage} timed out after {timeout:g} seconds",
              role=role, stage=stage, command=command, stderr=stderr or "")
    except OSError as exc:
        _fail(evidence, f"{role}: cannot run {stage}: {exc}", role=role,
              stage=stage, command=command, stderr=str(exc))
    # Some damaged files make ffprobe / ffmpeg log an error yet return zero.
    # -v error makes any remaining stderr a failed integrity check.
    if result.returncode != 0 or result.stderr.strip():
        _fail(evidence, f"{role}: {stage} failed (exit {result.returncode})",
              role=role, stage=stage, command=command,
              stderr=result.stderr, returncode=result.returncode)
    return result


def verify_videos(sim_result, expected_frames, *, ffprobe="/usr/bin/ffprobe",
                  ffmpeg="/usr/bin/ffmpeg", timeout_seconds=None):
    """Require and verify both complete rollout views when actions were executed.

    ``sim_result['videos']`` is the role -> writer metadata mapping produced by
    behavior_ap_sim's close(). Writer counters are checked first, then actual
    files are probed and fully decoded. The actual H.264 High / avc1 / yuv420p
    frame count must equal ``expected_frames`` for every advertised role.

    Zero expected frames return ``status='no_frames'`` and explicitly report
    that no playable video was verified. They do not fabricate empty videos.
    On failure, VideoVerificationError.evidence retains the failing command,
    stderr, stage, and any roles which have already passed.
    """
    if type(expected_frames) is not int or expected_frames < 0:
        raise ValueError("expected_frames must be a nonnegative integer")
    if not isinstance(sim_result, dict):
        raise ValueError("sim_result must be a dictionary")
    timeout = max(60.0, expected_frames / 30 * 5 + 60) if timeout_seconds is None else timeout_seconds
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("timeout_seconds must be finite and positive")
    evidence = {"schema_version": 1, "passed": False, "status": "checking",
                "playable_video_available": False, "expected_frames": expected_frames,
                "required_roles": list(REQUIRED_ROLES) if expected_frames else [],
                "recording_integrity_only_not_task_success": True,
                "videos": {}}
    videos = sim_result.get("videos", {})
    if not isinstance(videos, dict):
        _fail(evidence, "sim_result videos must be a role-to-metadata mapping", stage="metadata")
    if "steps" in sim_result and (type(sim_result["steps"]) is not int or sim_result["steps"] != expected_frames):
        _fail(evidence, "Expected frames differ from simulator action step count", stage="metadata")
    for role, meta in videos.items():
        if not isinstance(role, str) or not role or not isinstance(meta, dict):
            _fail(evidence, "Each video needs a nonempty role and metadata object", stage="metadata")
        if type(meta.get("frames")) is not int or meta["frames"] != expected_frames:
            _fail(evidence, f"{role}: writer frame count differs from {expected_frames}", role=role, stage="metadata")
    if expected_frames == 0:
        evidence.update(passed=True, status="no_frames",
                        reason="No action frames were expected; no playable video was verified")
        evidence["videos"] = {role: {"status": "no_frames", "declared_frames": 0,
                                    "path": meta.get("path"), "full_decode_passed": False,
                                    "playable_video_available": False}
                              for role, meta in videos.items()}
        return evidence
    missing = [role for role in REQUIRED_ROLES if role not in videos]
    if missing:
        evidence["missing_roles"] = missing
        _fail(evidence, f"Missing required rollout video roles: {', '.join(missing)}", stage="metadata")

    paths_seen = set()
    for role, meta in sorted(videos.items()):
        if meta.get("finalized") is not True:
            _fail(evidence, f"{role}: writer was not finalized", role=role, stage="metadata")
        for key in ("frames_submitted", "frames_encoded"):
            if key in meta and (type(meta[key]) is not int or meta[key] != expected_frames):
                _fail(evidence, f"{role}: {key} differs from {expected_frames}", role=role, stage="metadata")
        for key in ("queue_pending_frames", "frames_unencoded", "dropped_frames"):
            if key in meta and (type(meta[key]) is not int or meta[key] != 0):
                _fail(evidence, f"{role}: {key} is not zero", role=role, stage="metadata")
        for key in ("encoder_error", "encoder_cleanup_error"):
            if meta.get(key) is not None:
                _fail(evidence, f"{role}: {key}: {meta[key]}", role=role, stage="metadata")
        raw_path = meta.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            _fail(evidence, f"{role}: missing local video path", role=role, stage="metadata")
        path = Path(raw_path).expanduser().resolve()
        try:
            if not path.is_file() or path.stat().st_size == 0:
                _fail(evidence, f"{role}: video file is missing or empty: {path}", role=role, stage="file")
            before = path.stat()
        except OSError as exc:
            _fail(evidence, f"{role}: cannot read video: {exc}", role=role, stage="file", stderr=str(exc))
        if path in paths_seen:
            _fail(evidence, f"{role}: multiple roles point at the same video", role=role, stage="metadata")
        paths_seen.add(path)
        probe_command = [str(ffprobe), "-v", "error", "-threads", "1", "-count_frames",
                         "-select_streams", "v:0", "-show_entries",
                         "stream=codec_name,profile,codec_tag_string,pix_fmt,nb_read_frames,width,height,avg_frame_rate",
                         "-of", "json", str(path)]
        started = time.perf_counter_ns()
        probe = _run(probe_command, timeout, evidence, role, "ffprobe")
        try:
            streams = json.loads(probe.stdout)["streams"]
            if not isinstance(streams, list) or len(streams) != 1:
                raise ValueError("Expected one selected video stream")
            stream = streams[0]
            raw_count = stream["nb_read_frames"]
            if isinstance(raw_count, bool) or not isinstance(raw_count, (str, int)):
                raise ValueError("Decoded frame count must be an integer")
            frame_count = int(raw_count)
        except (ValueError, TypeError, KeyError) as exc:
            _fail(evidence, f"{role}: ffprobe did not return a readable frame count: {exc}",
                  role=role, stage="ffprobe_json", command=probe_command, stderr=probe.stderr)
        expected_codec = {"codec_name": "h264", "profile": "High",
                          "codec_tag_string": "avc1", "pix_fmt": "yuv420p"}
        mismatched = {key: {"expected": value, "actual": stream.get(key)}
                      for key, value in expected_codec.items() if stream.get(key) != value}
        if mismatched or frame_count != expected_frames:
            evidence["probe_stream"] = stream
            evidence["codec_mismatches"] = mismatched
            _fail(evidence, f"{role}: codec or decoded frame count mismatch (actual {frame_count}, expected {expected_frames})",
                  role=role, stage="probe_validation", command=probe_command)
        decode_command = [str(ffmpeg), "-nostdin", "-v", "error", "-xerror",
                          "-err_detect", "explode", "-threads", "1", "-i", str(path),
                          "-map", "0:v:0", "-an", "-f", "null", "-"]
        _run(decode_command, timeout, evidence, role, "full_decode")
        try:
            after = path.stat()
        except OSError as exc:
            _fail(evidence, f"{role}: video disappeared after decode: {exc}", role=role, stage="file")
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            _fail(evidence, f"{role}: video changed during verification", role=role, stage="file")
        evidence["videos"][role] = {
            "path": str(path), "bytes": after.st_size, "expected_frames": expected_frames,
            "actual_frames": frame_count, "writer_finalized": True, **expected_codec,
            "width": stream.get("width"), "height": stream.get("height"),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "ffprobe_passed": True, "full_decode_passed": True,
            "verification_wall_seconds": (time.perf_counter_ns() - started) / 1e9,
        }
    evidence.update(passed=True, status="verified", playable_video_available=True)
    return evidence
