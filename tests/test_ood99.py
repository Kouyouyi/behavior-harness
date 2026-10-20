import unittest
from unittest.mock import Mock
from ap_harness.ood99 import TargetSetScorer, match_fingerprint, mesh_pixels
import numpy as np

class OODTests(unittest.TestCase):
    def test_mesh_identity_does_not_match_name_prefix_collision(self):
        mask=np.array([[2]*16+[3]*16],dtype=np.uint32)
        labels={'2':'/World/scene_0/can_1/base/visual','3':'/World/scene_0/can_10/base/visual'}
        evidence=mesh_pixels(mask,labels,'/World/scene_0/can_1')
        self.assertEqual(evidence['pixel_count'],16)
        self.assertEqual(evidence['matched_renderer_ids'],[2])

    def test_missing_mesh_mapping_is_error_not_invisibility(self):
        with self.assertRaises(ValueError):mesh_pixels(np.full((4,4),7,dtype=np.uint32),{'2':'/World/scene_0/can'},'/World/scene_0/can')

    def test_visibility_and_distance_cannot_be_taken_from_different_objects(self):
        scorer=object.__new__(TargetSetScorer)
        a=Mock(return_value={'target_visible':True,'base_position':[0,0,0],'target_position':[2,0,0]})
        a.target.name='visible_far'
        b=Mock(return_value={'target_visible':False,'base_position':[0,0,0],'target_position':[.5,0,0]})
        b.target.name='hidden_near'
        scorer.scorers=[a,b]
        self.assertFalse(scorer()['find_and_approach_success'])

    def test_preflight_mismatch_rejects_before_policy_execution(self):
        old={'task_name':'task','instance_id':301,'objects':{'target':[[0,0,0],[0,0,0,1]]},'robot_pose':[[1,1,0],[0,0,0,1]]}
        current={**old,'objects':{'target':[[.1,0,0],[0,0,0,1]]}}
        with self.assertRaises(ValueError):match_fingerprint(current,old)

    def test_quaternion_sign_is_not_a_scene_change(self):
        old={'task_name':'task','instance_id':301,'objects':{'target':[[0,0,0],[0,0,0,1]]},'robot_pose':[[1,1,0],[0,0,0,1]]}
        current={**old,'objects':{'target':[[0,0,0],[0,0,0,-1]]}}
        match_fingerprint(current,old)

if __name__=='__main__':unittest.main()
