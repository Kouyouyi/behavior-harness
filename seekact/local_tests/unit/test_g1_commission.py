import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import zmq

from services.g1_sonic.vendor import wire_module


def test_vendor_wire_has_native_header_and_reference_order():
    wire = wire_module()
    fields = dict(joint_pos=np.arange(29, dtype=np.float32)[None],
                  joint_vel=np.zeros((1, 29), np.float32),
                  body_quat=np.array([[1, 0, 0, 0]], np.float32), frame_index=np.array([5], np.int64))
    data = wire.pack_pose_message(fields, version=1)
    header = json.loads(data[4:1284].rstrip(b'\0'))
    assert header['v'] == 1 and header['fields'][0]['shape'] == [1, 29]
    assert np.frombuffer(data[1284:1284+116], '<f4').tolist() == list(range(29))


def test_operator_owner_waits_for_init_then_only_sends_idle(tmp_path):
    # Random local port, no DDS participant and no native robot controller.
    context = zmq.Context()
    receiver = context.socket(zmq.SUB)
    receiver.setsockopt(zmq.SUBSCRIBE, b'')
    port = receiver.bind_to_random_port('tcp://127.0.0.1')
    receiver.close(0)
    receiver = context.socket(zmq.SUB)
    receiver.setsockopt(zmq.SUBSCRIBE, b'')
    receiver.connect(f'tcp://127.0.0.1:{port}')
    env = dict(os.environ)
    process = subprocess.Popen([sys.executable, '-m', 'services.g1_sonic.commission',
        '--run', str(tmp_path), '--start-policy', '--port', str(port)], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        start = time.monotonic()
        packets = []
        while time.monotonic()-start < 1.5:
            if receiver.poll(100):
                packets.append(receiver.recv())
        assert packets
        for data in packets:
            if data.startswith(b'command'):
                assert data[7+1280:] == bytes([0, 0, 1])
        (tmp_path/'controller.log').write_text('Init Done\n')
        seen_start = False
        deadline = time.monotonic()+2
        while time.monotonic() < deadline:
            if receiver.poll(100):
                data = receiver.recv()
                if data.startswith(b'command'):
                    start_bit, stop_bit, planner_bit = data[7+1280:]
                    assert stop_bit == 0 and planner_bit == 1
                    seen_start |= start_bit == 1
        assert seen_start
    finally:
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
        receiver.close(0); context.term()
    assert process.returncode == 0, stderr.decode()
    assert b'upper_owner_exiting' in stdout
