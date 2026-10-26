"""Production recorder smoke: two controller owners, actual H264 encode/decode."""
import tempfile,unittest,sys
from pathlib import Path
import cv2,numpy as np,av
import os
if os.environ.get('SEEKACT_SIM_ROOT'):sys.path.insert(0,os.environ['SEEKACT_SIM_ROOT'])
from behavior_ap_sim.backend import SimulationBackend
from services.behavior_ap.video import verify_videos

class LiveRecordingTests(unittest.TestCase):
 def test_real_recorder_labels_current_owner_without_changing_sensor_images(self):
  with tempfile.TemporaryDirectory()as temporary:
   root=Path(temporary);(root/'videos').mkdir()
   b=SimulationBackend({'output_dir':str(root),'render_profile':'fast'})
   b.np=np;b.cv2=cv2
   sensor=np.full((448,448,3),70,dtype=np.uint8);original=sensor.copy()
   b._frames=lambda:{'head':sensor,'left_wrist':sensor,'right_wrist':sensor}
   b._third_person_frame=lambda **kw:np.full((720,1280,3),90,dtype=np.uint8)
   for index,owner in enumerate(['ap_vla','champion_vla']):
    b.steps=index+1;b.action_owner={'controller':owner,'job_id':'job_'+str(index)};b._record_frame()
   metadata={}
   for role,writer in b._writers.items():writer.close();metadata[role]=writer.metadata()
   verified=verify_videos({'steps':2,'videos':metadata},2)
   self.assertTrue(verified['passed']);self.assertTrue(np.array_equal(sensor,original))
   for role in ('robot_cameras','third_person'):
    with av.open(metadata[role]['path'])as container:frames=[f.to_ndarray(format='rgb24')for f in container.decode(video=0)]
    a=frames[0][8:44,0:4].mean(axis=(0,1));c=frames[1][8:44,0:4].mean(axis=(0,1))
    self.assertGreater(a[2]-a[0],100);self.assertGreater(c[0]-c[2],100)
    self.assertEqual(frames[0].shape[0],504 if role=='robot_cameras' else 776)
