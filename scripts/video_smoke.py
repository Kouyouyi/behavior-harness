from pathlib import Path
import numpy as np
from ap_harness.paths import ROOT
from behavior_harness.backend import _VideoWriter
frame = np.zeros((448,672,3), dtype=np.uint8)
path = ROOT / 'outputs/video_smoke.mp4'
writer = _VideoWriter(path, frame)
for i in range(8):
    frame[:] = i*25
    writer.write(frame)
writer.close()
print(writer.metadata())
