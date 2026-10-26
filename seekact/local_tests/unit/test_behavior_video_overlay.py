"""Provenance/frame alignment and recording-only image annotations."""
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import os
if os.environ.get('SEEKACT_SIM_ROOT'):sys.path.insert(0,os.environ['SEEKACT_SIM_ROOT'])
from behavior_ap_sim.controller_overlay import add_controller_ribbon,HEIGHT
from scripts.annotate_behavior_controller_videos import action_owners

class OverlayTests(unittest.TestCase):
 def test_badges_preserve_every_camera_pixel_and_distinguish_controllers(self):
  image=np.random.default_rng(1).integers(0,256,(448,672,3),dtype=np.uint8);copy=image.copy()
  a=add_controller_ribbon(image,{'controller':'ap_vla','job_id':'job_00001'},step=1)
  b=add_controller_ribbon(image,{'controller':'champion_vla','job_id':'job_00002'},step=2)
  self.assertEqual(a.shape,(448+HEIGHT,672,3));self.assertTrue(np.array_equal(a[HEIGHT:],copy))
  self.assertTrue(np.array_equal(image,copy));self.assertFalse(np.array_equal(a[:HEIGHT],b[:HEIGHT]))
 def timeline(self,events):
  path=Path(self.directory.name)/'events.jsonl';path.write_text(''.join(json.dumps(e)+'\n'for e in events));return path
 def setUp(self):self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
 def events(self):
  return [{'event':'submitted','sequence':0,'step_before':0,'owner':{'controller':'ap_vla'}},
   {'event':'ack','sequence':0,'step_after':1},
   {'event':'submitted','sequence':1,'step_before':1,'owner':{'controller':'champion_vla'}},
   {'event':'ack','sequence':1,'step_after':2}]
 def test_frame_zero_is_action_one_and_switch_has_no_frame_offset(self):
  owners=action_owners(self.timeline(self.events()),2)
  self.assertEqual([o['controller']for o in owners],['ap_vla','champion_vla'])
 def test_missing_ack_is_not_silently_labelled(self):
  with self.assertRaisesRegex(ValueError,'exactly one'):action_owners(self.timeline(self.events()[:-1]),2)
 def test_misaligned_step_is_rejected(self):
  events=self.events();events[-1]['step_after']=3
  with self.assertRaisesRegex(ValueError,'sequence mismatch'):action_owners(self.timeline(events),2)
 def test_unknown_source_is_not_guessed_as_champion(self):
  events=self.events();events[0]['owner']={}
  with self.assertRaisesRegex(ValueError,'ownership'):action_owners(self.timeline(events),2)
