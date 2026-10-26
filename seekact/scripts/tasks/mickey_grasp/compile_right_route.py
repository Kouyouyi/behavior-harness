"""Build a reviewable RGB-D route recipe from recorded successful movement jobs.

Offline only. Does not publish actions. Deliberately omits exploratory VLA attempts;
the newly composed route still requires its own unattended hardware acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def compile_route(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    events = [json.loads(line) for line in (run/"session/mcp.jsonl").read_text().splitlines()]
    observations, moves, submitted, prepares = [], [], {}, []
    for row in events:
        if (row["event"] == "mcp_request" and row["tool"] == "g1_vla"
                and row["arguments"].get("action") == "prepare"):
            prepares.append(row["time"])
        if row["event"] == "mcp_request" and row["tool"] == "g1_move":
            if row["arguments"].get("action") == "run":
                moves.append(row)
        if row["event"] != "mcp_result":
            continue
        result = json.loads(row["result"])
        if row["tool"] == "g1_observe" and "frame_id" in result:
            observations.append((row["time"], result["frame_id"]))
        if row["tool"] == "g1_move" and result.get("status") == "submitted":
            submitted[row["call_id"]] = result["job_id"]
    if len(moves) != 10:
        raise ValueError("Expected the reviewed ten-segment right-side route")
    # The final two turns followed a VLA attempt that changed the torso/camera.
    # Their start view differs from segment 8's endpoint. Do not
    # silently stitch discontinuous states into an allegedly verified replay.
    moves = moves[:8]
    assets = {}

    def copy_frame(ident):
        folder = run/"session/agent-workspace/runtime/frames"/ident
        destination = output/ident
        if not destination.exists():
            destination.mkdir()
            for name in ("rgb.png", "depth.png", "meta.json"):
                source = folder/name
                shutil.copyfile(source, destination/name)
                assets[f"{ident}/{name}"] = hashlib.sha256(source.read_bytes()).hexdigest()
        return ident

    steps = []
    for move in moves:
        args = move["arguments"]
        result = json.loads((run/"move_results"/(submitted[move["call_id"]]+".json")).read_text())
        if not result.get("completed") or not result.get("settled"):
            raise ValueError("Only completed, settled movement is a route ingredient")
        after = next(ident for when, ident in observations if when > move["time"])
        steps.append(dict(
            command={k: args[k] for k in ("direction", "duration_s", "yaw_rate_deg_s") if k in args},
            before=copy_frame(args["frame_id"]), after=copy_frame(after),
            measured_yaw_delta_deg=result["measured_yaw_delta_deg"],
            source_job_id=result["job_id"], source_unix=move["time"]))
    recipe = dict(
        id="mickey_right_side_v1", source_run=run.name, steps=steps, assets=assets,
        prepared_frame=copy_frame(next(ident for when, ident in observations if when > prepares[-1])),
        nominal_movement_seconds=sum(s["command"]["duration_s"] for s in steps),
        max_seconds=180, verification="draft_composition_not_hardware_verified",
        notes=[
            "Eight contiguous movement ingredients completed and settled in the source session.",
            "Two later turns are excluded: an intervening VLA changed torso/camera pose. "
            "The new unattended controller has NOT passed hardware acceptance.",
            "Historical frames are audit evidence only. Current controller does not compare "
            "them with live images. The final prepare returns a current grasp-review checkpoint.",
            "Starting scene, current empty hands, arm clearance and the entire right-side "
            "corridor require current UI review. Historical clearance is not reused.",
            "Current measurements do not certify clearance; camera-to-body extrinsics remain uncalibrated.",
            "No translation distance is inferred from commanded speed and duration.",
            "One fresh prepare follows the route. The same UI agent reviews current grasp "
            "geometry and continues bounded VLA using g1_task_step, without replaying the route."])
    (output/"recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2)+"\n")
    return recipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    recipe = compile_route(args.run, args.output)
    print(json.dumps({k: recipe[k] for k in ("id", "nominal_movement_seconds", "verification")}))
