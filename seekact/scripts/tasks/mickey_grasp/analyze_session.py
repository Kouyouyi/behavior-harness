"""Offline task timing audit. Reads only saved MCP/UI logs; no robot connection."""
import argparse
import collections
import datetime
import json
from pathlib import Path


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def analyze(root):
    root = Path(root)
    events = list(rows(root / "agent-events.jsonl"))
    tasks = [e for e in events if e.get("kind") == "task"]
    if not tasks:
        raise ValueError("No task events")
    begin, end = tasks[0]["time"], tasks[-1]["time"]
    counts, requests, calls = collections.Counter(), {}, []
    rpc_seconds = 0.
    for entry in rows(root / "session/mcp.jsonl"):
        if not begin <= entry["time"] <= end:
            continue
        if entry["event"] == "mcp_request":
            counts[entry["tool"]] += 1
            requests[entry["call_id"]] = entry
        elif entry["event"] == "mcp_result" and entry["call_id"] in requests:
            request = requests[entry["call_id"]]
            rpc_seconds += entry["time"] - request["time"]
            args = request["arguments"]
            if (request["tool"] in ("g1_move", "g1_vla", "g1_hand")
                    and args.get("action") not in ("status", "result")):
                result = entry.get("result")
                result = json.loads(result) if isinstance(result, str) else result
                calls.append(dict(time=request["time"], tool=request["tool"],
                                  arguments=args, response=result))
    move_s, steps, prepare_count = 0., 0, 0
    outcomes = []
    for call in calls:
        response = call["response"]
        if not isinstance(response, dict) or not response.get("job_id"):
            continue
        folder = "move_results" if call["tool"] == "g1_move" else "vla_results"
        file = root / folder / (response["job_id"] + ".json")
        if not file.exists():
            continue
        result = json.loads(file.read_text())
        outcomes.append(dict(time=call["time"], tool=call["tool"], result=result))
        if call["tool"] == "g1_move" and call["arguments"]["action"] == "run":
            move_s += call["arguments"]["duration_s"]
        elif result.get("action") == "run":
            steps += result.get("steps_published", 0)
        elif result.get("action") == "prepare" and result.get("completed"):
            prepare_count += 1
    zone = datetime.timezone(datetime.timedelta(hours=8))
    return dict(
        source=str(root), start=datetime.datetime.fromtimestamp(begin, zone).isoformat(),
        end=datetime.datetime.fromtimestamp(end, zone).isoformat(),
        task_window_seconds=end-begin, tool_requests=dict(counts),
        measured_rpc_seconds=rpc_seconds,
        commanded_movement_seconds=move_s, published_vla_steps=steps,
        nominal_vla_seconds=steps/50, completed_prepares=prepare_count,
        nominal_prepare_seconds=prepare_count*6,
        interpretation="RPC time excludes asynchronous execution. Commanded/nominal "
                       "durations exclude settling, observation, reasoning and operator waits. "
                       "Their difference from wall time is NOT measured LLM overhead.",
        outcomes=outcomes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "outcomes"},
                     ensure_ascii=False, indent=2))
