"""Zip finished sessions and ship them off the robot so its disk never fills.

The launcher starts this after every run (`--session <root> --sweep`); it can also be run
by hand. For each finished session:

  1. zip it into <artifact root>/outbox/<session>__<status>__<UTC stamp>.zip (media stored,
     everything else deflated) with a .sha256 next to it;
  2. rsync the zip to RAX_SHIP_DEST (scp-style `host:/dir`; there is no default, and
     without it nothing ships) and verify the remote sha256 over ssh;
  3. only after a verified copy: delete the local zip, write <session>/.shipped.json, and
     delete the session directory. A run that was interrupted (status `stopped`) stays
     on disk so `start_agent.sh --resume` still works; it ships again when it ends.

`--sweep` retries every finished session that has not been shipped since it last
changed, then, while free space is below RAX_SHIP_MIN_FREE_GB, deletes the oldest
sessions that are already shipped and unchanged since.

    python -m services.runtime_ship --session <root> [--sweep] [--dry-run]
    python -m services.runtime_ship --sweep
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.storage_root import artifact_root  # noqa: E402

FINISHED = ('exited', 'stopped', 'failed')
KEEP_LOCAL = ('stopped',)              # resumable: ship a snapshot, keep the session
STORED = {'.jpg', '.jpeg', '.png', '.mp4', '.zip', '.gz', '.npz'}
MARKER = '.shipped.json'
# Credentials never leave the robot: the agent's Claude login state, relay config, tokens and keys.
SECRET_NAMES = {'.claude.json', '.credentials.json', 'model-relay.json', 'settings.json'}
SECRET_SUFFIXES = {'.token', '.key', '.pem'}


def secret(relative):
    return (relative.name in SECRET_NAMES or relative.suffix in SECRET_SUFFIXES
            or (relative.parts[:1] == ('claude-home',) and relative.name.startswith('.')))


def settings():
    return {'dest': os.environ.get('RAX_SHIP_DEST', ''),
            'min_free_gb': float(os.environ.get('RAX_SHIP_MIN_FREE_GB', '100')),
            'ssh': shlex.split(os.environ.get('RAX_SHIP_SSH', 'ssh -o BatchMode=yes -o ConnectTimeout=20'))}


def log(message):
    print(f"{datetime.now().isoformat(timespec='seconds')} {message}", flush=True)


def lifecycle(root):
    try:
        return json.loads((root/'agent-lifecycle.json').read_text())
    except (OSError, ValueError):
        return {}


def finished(root):
    """The run's terminal status, or None while it may still be running (a resume rewrites it)."""
    life = lifecycle(root)
    if life.get('status') not in FINISHED or not life.get('finished_at'):
        return None
    return life['status']


def last_change(root):
    return max((p.stat().st_mtime for p in root.rglob('*') if p.is_file() and p.name != MARKER), default=0.0)


def shipped_since_change(root):
    try:
        marker = json.loads((root/MARKER).read_text())
    except (OSError, ValueError):
        return False
    return marker.get('at', 0) >= last_change(root)


def make_zip(root, outbox, status):
    outbox.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = outbox/f'{root.name}__{status}__{stamp}.zip'
    partial = target.with_suffix('.zip.part')
    with zipfile.ZipFile(partial, 'w', allowZip64=True) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_symlink() or not path.is_file() or path.name == MARKER:
                continue  # agent-workspace/runtime/stream links back into the session
            if secret(path.relative_to(root)):
                continue
            method = zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            archive.write(path, Path(root.name)/path.relative_to(root), compress_type=method, compresslevel=6)
    digest = hashlib.sha256()
    with open(partial, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    os.replace(partial, target)
    target.with_suffix('.zip.sha256').write_text(f'{digest.hexdigest()}  {target.name}\n')
    return target, digest.hexdigest()


def upload(zip_path, digest, config):
    host, _, directory = config['dest'].partition(':')
    if not host or not directory:
        raise ValueError(f"RAX_SHIP_DEST must look like host:/dir, got {config['dest']!r}")
    ssh = config['ssh']
    subprocess.run([*ssh, host, f'mkdir -p {shlex.quote(directory)}'], check=True, timeout=60)
    sidecar = zip_path.with_suffix('.zip.sha256')
    subprocess.run(['rsync', '-a', '--partial', '-e', shlex.join(ssh), str(zip_path), str(sidecar),
                    f'{host}:{directory}/'], check=True)
    remote = subprocess.run([*ssh, host, f'sha256sum {shlex.quote(directory + "/" + zip_path.name)}'],
                            check=True, capture_output=True, text=True, timeout=3600).stdout.split()[0]
    if remote != digest:
        raise RuntimeError(f'remote checksum {remote} does not match {digest}')
    return f"{config['dest']}/{zip_path.name}"


def ship(root, config, *, dry_run=False, lock_wait=0.0):
    root = Path(root).resolve()
    status = finished(root)
    if status is None:
        log(f'skip {root.name}: not finished')
        return False
    if shipped_since_change(root):
        log(f'skip {root.name}: already shipped')
        return False
    with open(root/'launch.lock', 'a') as lock:  # the launcher holds this while a run is live
        deadline = time.monotonic() + lock_wait
        while True:  # the launcher that started us releases it as it exits
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    log(f'skip {root.name}: a launcher holds it')
                    return False
                time.sleep(1)
        if finished(root) is None or shipped_since_change(root):  # re-check under the lock
            return False
        size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        if dry_run:
            log(f"would ship {root.name} ({size / 1e9:.2f} GB, {status}) to {config['dest']}")
            return True
        started = time.monotonic()
        zip_path, digest = make_zip(root, artifact_root()/'outbox', status)
        remote = upload(zip_path, digest, config)
        zip_path.unlink()
        zip_path.with_suffix('.zip.sha256').unlink()
        marker = {'at': time.time(), 'remote': remote, 'sha256': digest, 'status': status, 'bytes': size}
        (root/MARKER).write_text(json.dumps(marker, indent=1))
        log(f'shipped {root.name} ({size / 1e9:.2f} GB, {status}) -> {remote} in {time.monotonic() - started:.0f} s')
        with open(artifact_root()/'shipped.jsonl', 'a') as ledger:
            ledger.write(json.dumps({'session': root.name, **marker}) + '\n')
        if status not in KEEP_LOCAL:
            shutil.rmtree(root)
            log(f'deleted local {root.name}')
    return True


def free_gb():
    return shutil.disk_usage(artifact_root()).free / 1e9


def sessions():
    return sorted((p.parent for p in (artifact_root()/'sessions').glob('*/agent-lifecycle.json')),
                  key=lambda p: p.stat().st_mtime)


def sweep(config, *, dry_run=False):
    for root in sessions():
        try:
            ship(root, config, dry_run=dry_run)
        except Exception as exc:
            log(f'FAILED {root.name}: {type(exc).__name__}: {exc}')
    for root in sessions():  # oldest first
        if free_gb() >= config['min_free_gb']:
            break
        if finished(root) and shipped_since_change(root):
            log(f"{'would evict' if dry_run else 'evicting'} shipped {root.name} (free {free_gb():.0f} GB)")
            if not dry_run:
                shutil.rmtree(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', type=Path, help='Session root to ship')
    parser.add_argument('--sweep', action='store_true', help='Retry unshipped sessions and free disk space')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not (args.session or args.sweep):
        parser.error('give --session and/or --sweep')
    config = settings()
    with open(artifact_root()/'.ship.lock', 'a') as lock:  # one shipper at a time
        fcntl.flock(lock, fcntl.LOCK_EX)
        failed = False
        if args.session:
            try:
                ship(args.session, config, dry_run=args.dry_run, lock_wait=300)
            except Exception as exc:
                failed = True
                log(f'FAILED {args.session.name}: {type(exc).__name__}: {exc}')
        if args.sweep:
            sweep(config, dry_run=args.dry_run)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
