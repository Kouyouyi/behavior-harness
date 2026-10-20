"""Inspect persisted evidence even when Isaac's native shutdown exits with zero."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def validate_exit(directory, returncode):
    directory = Path(directory)
    if returncode:
        if directory.exists():
            progress_path = directory/'sim/progress.json'
            progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
            (directory/'launcher_failure.json').write_text(json.dumps({
                'process_returncode':returncode, 'last_persisted_progress':progress,
                'complete_episode':False, 'automatic_replay':False,
                'note':'An unacknowledged submitted action may have advanced physics partially.'},indent=2))
        return returncode
    error = directory/'error.json'
    if error.exists():
        print(error.read_text(), file=sys.stderr)
        return 1
    result = directory/'sim/result.json'
    if not result.exists():
        print('Missing simulator terminal record', file=sys.stderr)
        return 1
    record = json.loads(result.read_text())
    if record.get('errors') or record.get('success') is None:
        print('Simulator reported initialization/finalization errors', file=sys.stderr)
        return 1
    return 0

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', required=True, type=Path)
    known,_=parser.parse_known_args()
    result=subprocess.run([sys.executable,'-m','ap_harness.run',*sys.argv[1:]])
    return validate_exit(known.output,result.returncode)

if __name__=='__main__': sys.exit(main())
