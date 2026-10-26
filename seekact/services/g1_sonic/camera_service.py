"""Continuous RGB-D from existing g1_video; owns its PICO camera request, never motors.

Run on the workstation. It sends OPEN_CAMERA/CLOSE_CAMERA only. Do not run alongside
a PICO camera client. Fixed calibration is for ZED Mini serial 12234465 only.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import struct
import threading
import time
import uuid

import av
import cv2

from .rgbd import StereoDepth


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name('.'+path.name)
    tmp.write_text(json.dumps(data, allow_nan=False, indent=2)+'\n')
    os.replace(tmp, path)


def packet(command, payload=b''):
    command = command.encode()
    body = struct.pack('<i', len(command))+command+struct.pack('<i', len(payload))+payload
    return struct.pack('>I', len(body))+body


def recv_exact(sock, length):
    chunks = bytearray()
    while len(chunks) < length:
        part = sock.recv(length-len(chunks))
        if not part:
            raise EOFError('Camera stream closed')
        chunks.extend(part)
    return bytes(chunks)


def run(args):
    import fcntl
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # One PICO request per robot, including different output directories.
    lock = open('/tmp/rax-g1-camera-'+args.robot.replace('.', '_')+'.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stereo = StereoDepth()
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    latest = queue.Queue(maxsize=1)
    stats = dict(received_packets=0, decoded_frames=0, published_frames=0, skipped_frames=0,
                 source='live_camera', robot_commands_sent=0, source_exposure_timestamp_available=False)
    errors = []
    control = connection = None
    worker = None
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.settimeout(10)
    started = time.monotonic()
    folders = deque()

    def receive():
        codec = av.CodecContext.create('h264', 'r')
        previous_stamp = None
        try:
            while not stop.is_set():
                size = struct.unpack('>I', recv_exact(connection, 4))[0]
                if not 0 < size < 16*1024*1024:
                    raise ValueError('Invalid H264 payload length')
                data = recv_exact(connection, size)
                now = time.monotonic()
                stats['received_packets'] += 1
                # Parser can retain one access unit; use the earlier packet arrival.
                stamp = previous_stamp if previous_stamp is not None else now
                previous_stamp = now
                for encoded in codec.parse(data):
                    for frame in codec.decode(encoded):
                        stats['decoded_frames'] += 1
                        item = (stats['decoded_frames'], stamp, frame.to_ndarray(format='bgr24'))
                        try:
                            latest.put_nowait(item)
                        except queue.Full:
                            try:
                                latest.get_nowait()
                                stats['skipped_frames'] += 1
                            except queue.Empty:
                                pass
                            latest.put_nowait(item)
        except Exception as exc:
            if not stop.is_set():
                errors.append(f'{type(exc).__name__}: {exc}')
            stop.set()

    try:
        listener.bind((args.receiver_ip, args.receiver_port))
        listener.listen(1)
        control = socket.create_connection((args.robot, 13579), timeout=5)
        address = args.receiver_ip.encode('ascii')
        payload = b'\xca\xfe\x01'+struct.pack('<7i', 2560, 720, 30, 12000000, 0, 0, args.receiver_port)
        payload += b'\x02VR'+bytes([len(address)])+address
        control.sendall(packet('OPEN_CAMERA', payload))
        connection, peer = listener.accept()
        if peer[0] != args.robot:
            raise ValueError('Unexpected video sender')
        connection.settimeout(2)
        worker = threading.Thread(target=receive, daemon=True)
        worker.start()
        while not stop.is_set() and (args.seconds == 0 or time.monotonic()-started < args.seconds):
            try:
                sequence, received, bgr = latest.get(timeout=.2)
            except queue.Empty:
                continue
            if time.monotonic()-received > 1:
                stats['skipped_frames'] += 1
                continue
            rgb, depth, mask = stereo.compute(bgr)
            if time.monotonic()-received > 1:
                stats['skipped_frames'] += 1
                continue
            frame_id = uuid.uuid4().hex
            tmp = output/('.'+frame_id)
            tmp.mkdir()
            for name, im in [('rgb.png', rgb), ('depth.png', depth), ('valid.png', mask)]:
                if not cv2.imwrite(str(tmp/name), im):
                    raise OSError('Failed writing '+name)
            meta = dict(stereo.metadata(), frame_id=frame_id, sequence=sequence,
                        received_monotonic=received, published_monotonic=time.monotonic(),
                        published_unix=time.time(), host_boot_id=boot_id,
                        timestamp_basis='workstation_packet_receive_not_exposure',
                        source='live_camera', valid_fraction=float((mask > 0).mean()))
            atomic_json(tmp/'meta.json', meta)
            os.replace(tmp, output/frame_id)
            atomic_json(output/'latest.json', meta)
            stats['published_frames'] += 1
            folders.append((received, frame_id))
            while folders and received-folders[0][0] > args.keep_seconds:
                _, old = folders.popleft()
                shutil.rmtree(output/old)
            atomic_json(output/'health.json', dict(stats, running=True, errors=errors,
                                                   updated_monotonic=time.monotonic()))
    finally:
        stop.set()
        if control:
            try:
                control.sendall(packet('CLOSE_CAMERA'))
            except OSError:
                pass
            control.close()
        if connection:
            connection.close()
        listener.close()
        if worker:
            worker.join(timeout=3)
        elapsed = time.monotonic()-started
        report = dict(stats, running=False, errors=errors, elapsed_s=elapsed,
                      published_fps=stats['published_frames']/elapsed,
                      updated_monotonic=time.monotonic())
        atomic_json(output/'health.json', report)
        lock.close()
        print(json.dumps(report, indent=2))
    if errors or not stats['published_frames']:
        raise RuntimeError('No healthy RGB-D stream; inspect health.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', default='192.168.123.164')
    p.add_argument('--receiver-ip', required=True)
    p.add_argument('--receiver-port', type=int, default=12345)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=0, help='0: run until SIGINT/SIGTERM')
    p.add_argument('--keep-seconds', type=float, default=60)
    args = p.parse_args()
    if not 0 <= args.seconds < float('inf') or not 5 <= args.keep_seconds <= 600:
        p.error('Invalid duration or retention')
    run(args)


if __name__ == '__main__':
    main()
